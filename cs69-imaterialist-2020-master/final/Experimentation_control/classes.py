#!/usr/bin/env python

## TODO: Specify the following to start an experiment:
MODE = "scratch"; #"pretrained" or "scratch"
BACKBONE = "resnet50"; #"resnet101"

import keras
from keras import callbacks
from mrcnn.model import data_generator, log
from mrcnn.config import Config
from mrcnn import utils
from sklearn.model_selection import train_test_split

import cv2
from imgaug import augmenters as iaa

import numpy as np
import pandas as pd
import json
import multiprocessing
import warnings
import os
import socket

# ------------------------------- Global Configurations & Reusable Objects -------------------------------
warnings.filterwarnings("ignore");
hostname = socket.gethostname();

## Objects with default values:
PRODUCTION = True; #TODO: Change this as necessary!
DATA_DIR = "input/imaterialist-fashion-2020-fgvc7/";
DATASET_NAME = "iMaterialist2020";
EPOCHS = 2;
N_SAMP = 128;
SEED = 2020;

## Override some constants based on host name & training mode:
print("Host Name: %s" % hostname)
if hostname in ["DavidChen", "SunintBindra", "local"]:
    DATA_DIR = "../" + DATA_DIR;
    WORKING_DIR = "../results/";
    MODEL_NAME_CUSTOM = "../saved_head_model";
elif hostname.endswith("hpcc.dartmouth.edu"):
    DATA_DIR = "../" + DATA_DIR;
    WORKING_DIR = "../results/";
    MODEL_NAME_CUSTOM = WORKING_DIR + "saved_head_model";
    if PRODUCTION:
        EPOCHS = 16;
        N_SAMP = None;
else:
    DATA_DIR = "/kaggle/" + DATA_DIR;
    WORKING_DIR = "/kaggle/working/";
    MODEL_NAME_CUSTOM = "/kaggle/working/saved_head_model";
    EPOCHS = 16;
    N_SAMP = None;

COCO_WEIGHTS_PATH = "mask_rcnn_coco.h5";
COCO_EXCLUDE_COLS = ["mrcnn_mask","mrcnn_class_logits","mrcnn_bbox","mrcnn_bbox_fc"];
LAYERS_REGEX = {"heads":r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)", "all":".*"}; #ref. Kaggle starter code
IMAGE_SIZE = 256;
LEARNING_RATE = 0.005;
LEARNING_RATE_TUNE = 0.0001;

## Static files or paths:
modelDir = os.path.join(WORKING_DIR, "logs");

sample_submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"));

with open(os.path.join(DATA_DIR, "label_descriptions.json"), "r") as fh:
    label_description = json.load(fh);
    fh.close();
n_classes = len(label_description['categories']);
n_attributes = len(label_description['attributes']);
print("Number of Classes: %d, Attributes: %d" % (n_classes, n_attributes))

DATA_AUG = iaa.Sequential([
    ## ref. Kaggle starter code
    ## Image rotation, translation, flipping:
    iaa.OneOf([
        iaa.Fliplr(0.2),
        iaa.Affine(
            scale = {"x": (0.99,1.01), "y":(0.98,1.03)},
            translate_percent = {"x": (-0.025,0.025), "y": (-0.05,0.05)}
        ),
    ]),
    ## Adjustment of contrast or resolution:
    iaa.OneOf([
        iaa.ContrastNormalization((0.75, 1.05)),
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
    ]),
]);

def train_val_test_split(data, seed=SEED, val_size=0.10, test_size=3200):
    """ Shared wrapper to split data """
    AGG_FUNC = lambda vec: list(vec);
    images_train = data.groupby("ImageId")['EncodedPixels','ClassId'].agg(AGG_FUNC);
    dimensions = data.groupby("ImageId")['Height','Width'].mean();

    images_train = images_train.join(dimensions, on="ImageId");
    images_train, images_test = train_test_split(images_train, test_size=test_size, random_state=seed);
    images_train, images_val = train_test_split(images_train, test_size=val_size, random_state=seed);
    print("n_train=%d, n_valid=%d, n_test_%d" % (len(images_train),len(images_val),len(images_test)))

    return images_train, images_val, images_test;

#------------------------------- Classes & templates -------------------------------
class MaskRCNN:
    """
    Main class for creating MaskRCNN model
    :reference: Kaggle starter code
    :reference: MaskRCNN shapes tutorial
    """
    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        assert self.mode == "training", "Create model in training mode."
        if layers in LAYERS_REGEX.keys():
            layers = LAYERS_REGEX[layers]

        ## Assemble data generators:
        train_generator = data_generator(
            train_dataset,
            self.config,
            augmentation = augmentation,
            batch_size = self.config.BATCH_SIZE,
            shuffle = True
        );
        val_generator = data_generator(
            val_dataset,
            self.config,
            batch_size = self.config.BATCH_SIZE,
            shuffle = True
        );

        ## Create log_dir if it does not exist:
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir);

        ## Implement custom callbacks:
        CALLBACKS = [
            callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True, write_images=False),
            callbacks.ModelCheckpoint(self.checkpoint_path, verbose=0, save_weights_only=True),
        ];
        if custom_callbacks:
            CALLBACKS += custom_callbacks;

        ## Train dense model:
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate));
        log("Checkpoint Path: {}".format(self.checkpoint_path));
        self.set_trainable(layers);
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM);

        workers = 0 if os.name is 'nt' else multiprocessing.cpu_count();
        print("Number of workers to use: %d" % workers)

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch = self.epoch,
            epochs = epochs,
            steps_per_epoch = self.config.STEPS_PER_EPOCH,
            validation_data = val_generator,
            validation_steps = self.config.VALIDATION_STEPS,
            max_queue_size = 100,
            callbacks = CALLBACKS,
            workers = 1,
            use_multiprocessing = False,
        );
        self.epoch = max(self.epoch, epochs);

class TrainingConfig(Config):
    """ MaskRCNN class for model training """
    NAME = DATASET_NAME;
    BACKBONE = BACKBONE;

    NUM_CLASSES = 1 + n_classes #add background
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = IMAGE_MIN_DIM; #square
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128);
    TRAIN_ROIS_PER_IMAGE = 16;

    GPU_COUNT = 1;
    IMAGES_PER_GPU = 3;
    STEPS_PER_EPOCH = 64;
    VALIDATION_STEPS = 4;

class InferenceConfig(TrainingConfig):
    """ Custom class to make predictions"""
    GPU_COUNT = 1;
    IMAGES_PER_GPU = 1;

class CustomDataset(utils.Dataset):
    """
    Custom data class for loading images
    :reference: Kaggle starter code
    """
    def __init__(self, data):
        super().__init__(self)
        self.IMAGE_SIZE = IMAGE_SIZE;
        self.DIMENSIONS = (IMAGE_SIZE, IMAGE_SIZE);

        for category in label_description["categories"]:
            self.add_class(DATASET_NAME, category.get("id"), category.get("name"));

        for i, row in data.iterrows():
            self.add_image(
                DATASET_NAME,
                image_id=row.name,
                path=str('{0}/train/{1}.jpg'.format(DATA_DIR, row.name)),
                labels=row['ClassId'],
                annotations=row['EncodedPixels'],
                height=row['Height'],
                width=row['Width']
            );

    def resize_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.DIMENSIONS, interpolation=cv2.INTER_AREA)
        return img;

    def load_image(self, image_id):
        img = self.image_info[image_id]['path']
        return self.resize_image(img);

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path'], [x for x in info['labels']];

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE, len(info['annotations'])), dtype=np.uint8);

        labels = [];
        for (m, (annotation, label)) in enumerate(zip(info['annotations'], info['labels'])):
            sub_mask = np.full(info['height'] * info['width'], 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')];

            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel + annotation[2 * i + 1]] = 1;

            sub_mask = sub_mask.reshape((info['height'], info['width']), order='F');
            sub_mask = cv2.resize(sub_mask, self.DIMENSIONS, interpolation=cv2.INTER_NEAREST);

            mask[:, :, m] = sub_mask;
            labels.append(int(label) + 1);
            
        return mask, np.array(labels);
