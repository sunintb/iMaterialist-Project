## General modules
import numpy as np;
import pandas as pd;
import copy as cp;
import json;
import bz2;
import os;
from tqdm.notebook import tqdm;  #v.4.45+ required (v. 4.31 doesn't work!)

## Computer vision / data vis:
import cv2;
from PIL import Image;
import matplotlib.pyplot as plt;

## Computational modules
import gc;
import joblib; #parallel
import pickle; #model saving

## Custom modules:
from helpers import *;
from CustomClasses import *;
from DataClasses import *;

import torch;
from torch.utils.data import DataLoader;
from torch.optim import Adam, lr_scheduler;

## Constants:
DATA_DIR = "../input/imaterialist-fashion-2020-fgvc7/";
MODEL_FILE_DIR = '../input/imaterialist2020-pretrain-models/'
LOCAL = True;
LEARN_RATE = 0.005;
if LOCAL:
    EPOCHS = 2;
    BATCH_SIZE = 2;
    DATA_DIR = "../" + DATA_DIR;
    MODEL_FILE_DIR = "../"  + MODEL_FILE_DIR;
else:
    EPOCHS = 8;
    BATCH_SIZE = 4;

IMG_DIMS_FOR_ATTR = (160, 160);
IMG_DIMS_FOR_ATTR = (40,40)
DIM_TRANS_3D = (2, 0, 1);
THRESH_CALLING = 0.50;

GPU = torch.cuda.is_available();
PRETRAINED_FLAG = not os.path.isfile(MODEL_FILE_DIR + "maskmodel_%d.model" % IMG_DIMS_FOR_ATTR[0]);
print(PRETRAINED_FLAG)

## Automatically detect number of CPUs available (which can vary!)
N_CORES = os.cpu_count() - 1;
print("Using %d workers" % N_CORES)

def get_last_ids_from_annot(annot, key_cls="ClassId", key_attr="AttributesIds"):
    """ Helper function to extract highest / last class & attribute labels """
    clsId_end = max(annot_train[key_cls]);
    attrId_end = 0; #largest so far
    for attr in annot[key_attr]:
        for s in str(attr).split(","):
            if s != "nan":
                s = int(s);
                if s > attrId_end: attrId_end = s;
    return clsId_end, attrId_end;

def get_attr_idx_and_counts(raw_num_cls, raw_num_attr, annot, key_cls="ClassId", key_attr="AttributesIds"):
    ## Boolean flag whether there exists an AttributeId for a ClassId being 2D array:
    num_cls = raw_num_cls + 1; #add background
    num_attr = raw_num_attr + 1; #add background
    
    joint_mat = np.zeros((num_cls, num_attr));
    clz_attrid2idx = [[] for _ in range(num_cls)];

    ## Compupte num of Attributes for every ClassId:
    for id_cls, ids_attrs in zip(annot[key_cls], annot[key_attr]):
        for id_attr in str(ids_attrs).split(","):
            if id_attr != "nan":
                id_attr = int(id_attr);
                joint_mat[id_cls, id_attr] = 1;
                if not id_attr in clz_attrid2idx[id_cls]:
                    clz_attrid2idx[id_cls].append(id_attr);

    ## Compupte num of Attributes for every ClassId
    cls_attr_count = joint_mat.sum(axis=1).astype(np.int32);
    return clz_attrid2idx, cls_attr_count;

def global_getitem(imgid, train_df, attr_image_size):
    df = train_df[train_df.ImageId==imgid];
    res = [];
    imag = cv2.imread(DATA_DIR+"train/"+str(imgid)+".jpg");
    for idx in range(len(df)):
        t = df.values[idx]
        cid = t[4]
        mask = rle_to_mask(t[1],t[2],t[3])
        attr = map(int,str(t[5]).split(",")) if str(t[5]) != 'nan' else []
        where = np.where(mask != 0)
        y1,y2,x1,x2 = 0,0,0,0
        if len(where[0]) > 0 and len(where[1]) > 0:
            y1,y2,x1,x2 = min(where[0]),max(where[0]),min(where[1]),max(where[1])
        if y2>y1+10 and x2>x1+10:
            X = cv2.resize(imag[y1:y2,x1:x2], attr_image_size)
            X = bz2.compress(pickle.dumps(X), 3);
        else:
            X = None
        mask = cv2.resize(mask, attr_image_size)
        mask = bz2.compress(pickle.dumps(mask), 3);
        res.append((cid, mask, attr, X));

    imag = cv2.resize(imag, attr_image_size)
    imag = bz2.compress(pickle.dumps(imag), 3);
    return res, imag, imgid;

print("OpenCV2 version: %s" % cv2.__version__)
print("Pillow version: %s" % Image.__version__)

# -------------------- Identify AttributesIds for each ClassId --------------------
annot_train = load_train_annot(DATA_DIR);
annot_train = annot_train.sample(n=BATCH_SIZE*2);

## Sample n for each class
def create_balanced_training_set(annot_train, n_per_class=5):
    idx = [];
    for clz in np.unique(annot_train["ClassId"]):
        df_sub = annot_train[annot_train.ClassId == clz];
        idx.extend(df_sub["ClassId"].sample(n_per_class));
    return annot_train.iloc[idx, :];

# annot_train = create_balanced_training_set(annot_train, 4);
print("Training data annotation: %d x %d" % annot_train.shape)

clsId_end, attrId_end = get_last_ids_from_annot(annot_train);
clz_attrid2idx, clz_attr_num = get_attr_idx_and_counts(clsId_end, attrId_end, annot_train);
clz_attr_num

data_cache = [];
for i in tqdm(list(set(annot_train["ImageId"]))):
    res, imag, imgid = global_getitem(i, annot_train, IMG_DIMS_FOR_ATTR);
    for cid, mask, attr, X in res:
        data_cache.append((cid, mask, attr, imag, X, imgid));
joblib.dump(data_cache, MODEL_FILE_DIR + "data_cache_%d" % IMG_DIMS_FOR_ATTR[0]);

# --------------------- Training: Attribute Classifier ---------------------
def train_attr(clzid, num_epochs=2, lr=LEARN_RATE):
    ## Instantiate data & dataloader object:
    data = AttributesDataset(clzid, clz_attr_num, clz_attrid2idx, data_cache);
    data = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True);

    ## Set up model:
    model = AttributesNetwork(clz_attr_num[clzid]);
    dp = torch.nn.DataParallel(model) #if torch.cuda.device_count() > 1 else model.parameters();
    params = [p for p in dp.parameters() if p.requires_grad];
    if GPU: model.cuda();
    optimizer = Adam(params, lr=lr);

    ## Define model configurations:
    loss_func = nn.BCELoss();
    lr_scheduler.StepLR(optimizer, step_size=8);

    ## Training for multiple epochs:
    progress = tqdm(list(range(num_epochs)));
    for epoch in progress:
        for i, (xx, y) in enumerate(data):
            if GPU:
                xx, y = xx.cuda(), y.cuda();

            xx = dp(xx);
            losses = loss_func(xx, y);
            # progress.set_description("loss:%05f" % losses)
            optimizer.zero_grad();
            losses.backward();
            optimizer.step();

        del xx, y, losses;
        if GPU: torch.cuda.empty_cache();
        gc.collect();
    return model;

for clzid in range(len(clz_attr_num)):
    fname_cls = MODEL_FILE_DIR + "attrmodel_%d-%d.model" % (IMG_DIMS_FOR_ATTR[0], clzid);
    if clz_attr_num[clzid] > 0 and not os.path.isfile(fname_cls):
        model = train_attr(clzid, EPOCHS);
        torch.save(model.state_dict(), fname_cls);
model = reset_cache();  # once saved, reset

## Save model in data_mask for later use:
data_mask = {};
while len(data_cache) > 0:
    cid, mask, _, imag, _, imgid = data_cache.pop();
    mask = decompress(mask);
    if imgid not in data_mask:
        imag = decompress(imag);
        data_mask[imgid] = [compress(imag.transpose(DIM_TRANS_3D).astype(np.float32)),
                            np.zeros(IMG_DIMS_FOR_ATTR, dtype=np.int32)];
    data_mask[imgid][1][mask != 0] = cid + 1;

## Compress model:
for k in data_mask.keys():
    data_mask[k][1] = compress(data_mask[k][1]);

gc.collect();

# ----------------------- Training the Masked Image -----------------------
## Phase I: Training
def train_mask(num_epochs=1, lr=LEARN_RATE, batch_size=BATCH_SIZE):
    """ Wrapper function to train the Mask """
    data = MaskTrainingSet(list(data_mask.keys()), data_mask);
    data = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=N_CORES);

    model = StackedHourglass(clsId_end+2);
    if GPU: model.cuda();

    dp = torch.nn.DataParallel(model);  # ATTENTION: Higher memory requirement
    params = [p for p in dp.parameters() if p.requires_grad];
    optimizer = Adam(params, lr=lr);
    loss_func = nn.CrossEntropyLoss();
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.9)

    for epoch in range(num_epochs):
        total_loss = [];
        progress = tqdm(data, total=len(data));
        for i, (imag, mask) in enumerate(progress):
            ## Learning problem:
            xx = imag.cuda() if GPU else imag;
            xx = dp(xx) # if torch.cuda.device_count() > 1 else xx;
            y = mask.cuda() if GPU else mask;

            ## Flatten to 1D:
            y = y.reshape((y.size(0), -1));
            y = y.reshape((y.size(0) * y.size(1),));
            xx = xx.reshape((xx.size(0), xx.size(1), -1));
            xx = torch.transpose(xx, 2, 1);
            xx = xx.reshape((xx.size(0) * xx.size(1), -1));

            ## Compute loss & backprop
            losses = loss_func(xx, y.long()); #debugged
            progress.set_description("loss:%05f" % losses);
            optimizer.zero_grad();
            losses.backward();
            optimizer.step();
            total_loss.append(losses.detach().cpu().numpy());

        del progress, xx, y, losses; #conserve memory
        torch.cuda.empty_cache();
        gc.collect();
    return model;

if PRETRAINED_FLAG:
    model = train_mask(EPOCHS);
    torch.save(model.state_dict(), MODEL_FILE_DIR + "maskmodel_%d.model" % IMG_DIMS_FOR_ATTR[0]);

## Phase II: Prediction on test set *Optional random subsetting*

## Phase III. Cut out mask image & store in saved model:
def load_trained_model(fname_model):
    ## Re-instantiate model:
    model = StackedHourglass(clsId_end+2);
    if GPU: model.cuda();
    model.load_state_dict(torch.load(fname_model));
    model.eval();
    return model;

def wrapper_show_mask():
    """ Wrapper to show predicted masks & associated images side-by-side"""
    for clzid in range(len(clz_attr_num)):
        if clz_attr_num[clzid] > 0 and os.path.isfile(
                MODEL_FILE_DIR + "attrmodel_%d-%d.model" % (IMG_DIMS_FOR_ATTR[0], clzid)):
            model = AttributesNetwork(clz_attr_num[clzid])
            if GPU:
                model.cuda()
            model.eval()
            model.load_state_dict(torch.load(MODEL_FILE_DIR + "attrmodel_%d-%d.model" % (IMG_DIMS_FOR_ATTR[0], clzid)))
            for i in range(len(predict_classid)):
                if predict_classid[i] == clzid:
                    imag = cv2.imread(DATA_DIR + "test/" + predict_imgeid[i] + ".jpg");
                    imag = scale_image(imag, 1024);
                    mask = cv2.resize(predict_mask[i], (imag.shape[1], imag.shape[0]), interpolation=cv2.INTER_NEAREST);
                    where = np.where(mask != 0);
                    y1, y2, x1, x2 = 0, 0, 0, 0;
                    if len(where[0]) > 0 and len(where[1]) > 0:
                        y1, y2, x1, x2 = min(where[0]), max(where[0]), min(where[1]), max(where[1])
                        if y2 > y1 + 80 and x2 > x1 + 80 and np.sum(mask) / 255 > 1000:
                            print("class id=", clzid);
                            plt.subplot(1, 2, 1);
                            plt.imshow(imag);
                            plt.subplot(1, 2, 2);
                            plt.imshow(mask);
                            plt.show();

def make_predictions(testset, model, PX_BASE=1, PX_BORDER=80):
    data_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_CORES);
    predict_imgeid, predict_mask, predict_rle, predict_classid, predict_attr = [], [], [], [], [];
    num_pred = 0;
    for X in tqdm(data_loader, total=len(data_loader)):
        if GPU: X = X.cuda();
        pred = model(X).detach().cpu().numpy();
        for i, mask in enumerate(pred):
            imgid = testset.imgids[num_pred];
            num_pred += 1;
            pred_id = mask.argmax(axis=0)-1;  # exclude background
            for clz in set(pred_id.reshape((-1,)).tolist()):
                if clz >= 0:
                    maskdata = (pred_id == clz).astype(np.uint8) * 255;
                    predict_imgeid.append(imgid);
                    predict_mask.append(maskdata);
                    predict_rle.append("");
                    predict_classid.append(clz);
                    predict_attr.append([]); #init.

    uses_index = [];
    for clzid in tqdm(range(len(clz_attr_num))):
        fname_attr = MODEL_FILE_DIR + "attrmodel_%d-%d.model" % (IMG_DIMS_FOR_ATTR[0],clzid);
        if clz_attr_num[clzid] > 0 and os.path.isfile(fname_attr):
            attr_model = AttributesNetwork(clz_attr_num[clzid]);
            if GPU: attr_model.cuda();
            attr_model.eval();
            attr_model.load_state_dict(torch.load(fname_attr));

            for i in range(len(predict_classid)):
                if predict_classid[i] == clzid:
                    imag = cv2.imread(DATA_DIR+"test/"+predict_imgeid[i]+".jpg");
                    imag = scale_image(imag, 1024);
                    mask = cv2.resize(predict_mask[i], (imag.shape[1], imag.shape[0]), interpolation=cv2.INTER_NEAREST)
                    imag[mask == 0] = PX_BASE;
                    where = np.where(mask != 0);
                    y1, y2, x1, x2 = 0, 0, 0, 0;  #init
                    if len(where[0]) > 0 and len(where[1]) > 0:
                        y1, y2, x1, x2 = min(where[0]), max(where[0]), min(where[1]), max(where[1]);  # min-max
                        if y2 > int(y1+PX_BORDER) and x2 > int(x1+PX_BORDER) and int(np.sum(mask)/PX_BASE) > 1024:
                            predict_rle[i] = mask_to_rle(mask);
                            X = cv2.resize(imag[y1:y2,x1:x2],IMG_DIMS_FOR_ATTR).transpose(DIM_TRANS_3D);
                            tensor = torch.tensor([X], dtype=torch.float32);
                            if GPU: tensor.cuda();
                            attr_preds = model(tensor);
                            attr_preds = attr_preds.detach().cpu().numpy()[0];
                            for j in range(len(attr_preds)):
                                if attr_preds[j] > THRESH_CALLING:
                                    uses_index.append(i);
                                    predict_attr[i].append(clz_attrid2idx[predict_classid[i]][j]);

    if GPU: torch.cuda.empty_cache();
    gc.collect();
    return uses_index, predict_imgeid, predict_classid, predict_attr, predict_mask, predict_rle;

model = load_trained_model(MODEL_FILE_DIR + "maskmodel_%d.model" % IMG_DIMS_FOR_ATTR[0]);
print(model)
testset = MaskTestSet(DATA_DIR+"test/", IMG_DIMS_FOR_ATTR, BATCH_SIZE*2); #TODO: delete subsampling
uses_index, predict_imgeid, predict_classid, predict_attr, predict_mask, predict_rle = make_predictions(testset, model, 1, 0);
wrapper_show_mask();

# ------------------------------- Export Submission CSV -------------------------------
predict_imgeid = [predict_imgeid[i] for i in set(uses_index)];
predict_mask = [predict_mask[i] for i in set(uses_index)];
predict_rle = [predict_rle[i] for i in set(uses_index)];
predict_classid = [predict_classid[i] for i in set(uses_index)];
predict_attr = [predict_attr[i] for i in set(uses_index)];

predict_attri_str = [",".join(list(map(str, predict_attr[i]))) for i in range(len(predict_classid))];
predict_attri_str = [predict_attri_str[i] for i in set(uses_index)];

## Export
submission = pd.DataFrame({
    "ImageId": predict_imgeid,
    "EncodedPixels": predict_rle,
    "ClassId": predict_classid,
    "AttributesIds": predict_attri_str
});
submission.to_csv("test_submission.csv", index=False);