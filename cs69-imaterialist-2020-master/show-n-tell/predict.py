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

OUTPUT_PATH = "test_submission.csv";
KEY = "ImageId";
EMPTY_ATTR = "1 1"; #Kaggle required format

keras.engine.saving = saving;
IS_HPC = hostname.endswith("dartmouth.edu");

def resize_image(image_path):
    """
    Helper function to resize images at inference time
    :reference: CustomDataset class instance method
    """
    img = cv2.imread(image_path);
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA);
    return img;

def rle_conversion(bits):
    """
    Helper function to convert bit-objects to RLE format
    :reference: Kaggle starter code iMaterialist 2019-20
    """
    rle = [];
    pos = 0;
    for bit, group in itertools.groupby(bits):
        group_list = list(group);
        if bit:
            rle.extend([pos, sum(group_list)]);
        pos += len(group_list);
    return rle;

def process_masks(masks, rois):
    """
    Helper function to meet Kaggle challenge requirement
    :reference: Kaggle starter code
    """
    patches = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0);
    idx_mask = np.argsort(patches); 
    joined_mask = np.zeros(masks.shape[:-1], dtype=bool);
    for m in idx_mask:
        masks[:,:,m] = np.logical_and(masks[:, :, m], np.logical_not(joined_mask));
        joined_mask = np.logical_or(masks[:, :, m], joined_mask);
    for k in range(masks.shape[-1]):
        position = np.where(masks[:,:,k] == True);
        if np.any(position):
            x0, y0 = np.min(position, axis=1);
            x1, y1 = np.max(position, axis=1);
            rois[k, :] = [x0, y0, x1, y1];
    return masks, rois;

def main():
    print("INFO: Instantiate inference-mode model object") #ref. MRCNN Shapes tutorial
    model_path = "saved_head_model_weights.h5"; #manually copied/moved
    print("Model weight location: %s" % model_path)
    model = modellib.MaskRCNN(mode="inference", config=InferenceConfig(), model_dir=WORKING_DIR);
    model.load_weights(model_path, by_name=True);

    print("INFO: Iteratively make predictions for submission file") #ref.: Kaggle starter code
    sample_data = cp.deepcopy(sample_submission); #init
    submission_list = [];
    missing_count = 0;

    progress = sample_data.iterrows() if IS_HPC else tqdm(sample_data.iterrows(),total=sample_data.shape[0]);
    for i, row in progress:
        if IS_HPC: print("Now on image %d" % i)
        image = resize_image(str(DATA_DIR+"/test/"+row[KEY]) + ".jpg");
        preds = model.detect([image])[0];
        if preds["masks"].size > 0:
            masks, _ = process_masks(preds["masks"], preds["rois"]); 
            for m in range(masks.shape[-1]):
                mask = masks[:, :, m].ravel(order="F")
                rle = rle_conversion(mask); 
                label = preds["class_ids"][m] - 1;
                submission_list.append([row[KEY], ' '.join(list(map(str, rle))), label, np.NaN])
        else:
            submission_list.append([row[KEY], EMPTY_ATTR, 23, np.NaN]);
            missing_count += 1;

    print("INFO: Making CSV file with predictions...")
    submission_data = pd.DataFrame(submission_list, columns=sample_submission.columns.values)
    print("Total number of unique images: %d" % submission_data[KEY].nunique())
    print("Number of images with missingness: %d" % missing_count)
    submission_data.to_csv(OUTPUT_PATH, index=False);
    print("Inference Phase Complete!");

main();