#!/usr/bin/env python

from classes import *

import mrcnn.model as modellib
import tensorflow
from tensorflow.python.keras.engine import saving;
import numpy as np
import pandas as pd
import copy as cp
import itertools
import cv2
from tqdm import tqdm, tqdm_pandas #prone to error on GPU cluster

KEY = "ImageId";

keras.engine.saving = saving;
IS_HPC = hostname.endswith("dartmouth.edu");

def resize_image(image_path):
    """
    Helper function to resize images at inference time
    :reference: CustomDataset class instance method
    """
    img = cv2.imread(image_path);
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
    img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE), interpolation=cv2.INTER_AREA);
    return img;

def rle_conversion(bit_data):
    """
    Converts bit objects to RLE format
    :reference: Kaggle starter code iMaterialist 2019-20
    """
    rle, pos = [], 0;
    for bit, group in itertools.groupby(bit_data):
        group_list = list(group);
        if bit: rle.extend([pos, sum(group_list)]);
        pos += len(group_list);
    return rle;

def process_masks(masks):
    """
    Extracts mask from prediction object
    :reference: Kaggle starter code
    """
    patches = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0);
    idx_mask = np.argsort(patches); 
    joined_mask = np.zeros(masks.shape[:-1], dtype=bool);
    for m in idx_mask:
        masks[:,:,m] = np.logical_and(masks[:,:,m], np.logical_not(joined_mask));
        joined_mask = np.logical_or(masks[:,:,m], joined_mask);
    return masks;

def main():
    print("INFO: Instantiate inference-mode model object") #ref. MRCNN Shapes tutorial
    model_path = MODEL_NAME_CUSTOM + "_weights_" + MODE + "_" + BACKBONE +".h5";

    print("Model weight location: %s" % model_path)
    model = modellib.MaskRCNN(mode="inference", config=InferenceConfig(), model_dir=WORKING_DIR);
    model.load_weights(model_path, by_name=True);

    print("INFO: Making predictions on the test set")
    train_data = pd.read_csv(os.path.join(DATA_DIR,"train.csv"));
    _, _, images_test = train_val_test_split(train_data);
    images_test[KEY] = images_test.index;

    print("INFO: Iteratively make predictions for submission file") #ref.: Kaggle starter code
    sample_data = cp.deepcopy(images_test);
    submission_list = [];
    progress = sample_data.iterrows() if IS_HPC else tqdm(sample_data.iterrows(),total=sample_data.shape[0]);
    for i, row in progress:
        image = resize_image(str(DATA_DIR+"/train/"+row[KEY]) + ".jpg");
        preds = model.detect([image])[0];
        if preds["masks"].size > 0:
            masks = process_masks(preds["masks"]);
            for m in range(masks.shape[-1]):
                label = preds["class_ids"][m] - 1;
                submission_list.append([row[KEY],label]);
        else:
            submission_list.append([row[KEY], np.NaN]);

    submission_data = pd.DataFrame(submission_list, columns=[KEY,"ClassID"]);
    submission_data.to_csv(MODEL_NAME_CUSTOM + "_" + MODE + "_" + BACKBONE + ".csv");

main();