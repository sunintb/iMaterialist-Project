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

keras.engine.saving = saving; #hacky solution
OUTPUT_PATH = "testset_submission.csv";
KEY = "ImageId";
IS_HPC = hostname.endswith("dartmouth.edu");

def resize_image(image_path):
    """ Helper function copied from CustomDataset class """
    img = cv2.imread(image_path);
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA);
    return img;

def rle_conversion(bits):
    """ Helper function to convert to RLE format """
    rle = [];
    pos = 0;
    for bit, group in itertools.groupby(bits):
        group_list = list(group);
        if bit:
            rle.extend([pos, sum(group_list)]);
        pos += len(group_list);
    return rle;

def process_masks(masks, rois):
    """ Helper function to meet Kaggle challenge requirement """
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0);
    mask_index = np.argsort(areas);
    union_mask = np.zeros(masks.shape[:-1], dtype=bool);
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask));
        union_mask = np.logical_or(masks[:, :, m], union_mask);
    for m in range(masks.shape[-1]):
        mask_position = np.where(masks[:, :, m] == True);
        if np.any(mask_position):
            y1, x1 = np.min(mask_position, axis=1);
            y2, x2 = np.max(mask_position, axis=1);
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois;

def main():
    ## Instantiate inference-mode model object :
    model_path = "saved_head_model_weights.h5"; #manually copied/moved
    print("Model weight location: %s" % model_path)
    model = modellib.MaskRCNN(mode="inference", config=InferenceConfig(), model_dir=WORKING_DIR);
    model.load_weights(model_path, by_name=True);

    ## Iteratively make predictions for submission file:
    sample_data = cp.deepcopy(sample_submission); #init
    submission_list = [];
    missing_count = 0;
    progress = sample_data.iterrows() if IS_HPC else tqdm(sample_data.iterrows(),total=sample_data.shape[0]);

    for i, row in progress:
        if IS_HPC: print("Now on iteration %d" % i)
        image = resize_image(str(DATA_DIR+"/test/"+row[KEY]) + ".jpg");
        result = model.detect([image])[0];
        if result["masks"].size > 0:
            masks, _ = process_masks(result["masks"], result["rois"])
            for m in range(masks.shape[-1]):
                mask = masks[:, :, m].ravel(order='F')
                rle = rle_conversion(mask)
                label = result["class_ids"][m] - 1
                submission_list.append([row[KEY], ' '.join(list(map(str, rle))), label, np.NaN])
        else:
            submission_list.append([row[KEY], '1 1', 23, np.NaN]);
            missing_count += 1;

    ## Export to CSV file:
    submission_data = pd.DataFrame(submission_list, columns=sample_submission.columns.values)
    print("Total number of unique images: %d" % submission_data[KEY].nunique())
    print("Number of images with missingness: %d" % missing_count)
    submission_data.head()
    submission_data.to_csv(OUTPUT_PATH, index=False);

main();