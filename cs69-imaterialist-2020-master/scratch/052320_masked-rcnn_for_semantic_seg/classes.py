#!/usr/bin/env python

## Models:
import keras
from mrcnn.model import data_generator, log
from mrcnn.config import Config
from mrcnn import utils

## Computer vision:
import cv2
from imgaug import augmenters as iaa

## General:
import numpy as np
import pandas as pd
import json
import multiprocessing
import warnings
import os
import socket

warnings.filterwarnings("ignore");
hostname = socket.gethostname();

## Objects with default values:
PRODUCTION = True; #TODO: Change this as necessary!
DATA_DIR = "input/imaterialist-fashion-2020-fgvc7/";
DATASET_NAME = "iMaterialist2020";
EPOCHS = 2;
N_SAMP = 128;

## Override some constants based on host name & training mode:
print("Host Name: %s" % hostname)
if hostname in ["DavidChen", "SunintBindra", "local"]:
    DATA_DIR = "../../" + DATA_DIR;
    WORKING_DIR = "../results/";
    MODEL_NAME_CUSTOM = "../../saved_head_model";
elif hostname.endswith("hpcc.dartmouth.edu"): #["discovery7.hpcc.dartmouth.edu","g0x.hpcc.dartmouth.edu"]:
    DATA_DIR = "../" + DATA_DIR;
    WORKING_DIR = "../results/";
    MODEL_NAME_CUSTOM = WORKING_DIR + "saved_head_model";
    if PRODUCTION:
        EPOCHS = 25;
        N_SAMP = None;
else:
    DATA_DIR = "/kaggle/" + DATA_DIR;
    WORKING_DIR = "/kaggle/working/";
    MODEL_NAME_CUSTOM = "/kaggle/working/saved_head_model";
    EPOCHS = 25;
    N_SAMP = None;

COCO_WEIGHTS_PATH = "mask_rcnn_coco.h5";
COCO_EXCLUDE_COLS = ["mrcnn_mask","mrcnn_class_logits","mrcnn_bbox","mrcnn_bbox_fc"];
LAYERS_REGEX = {
    "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
    "all": ".*",
    "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
    "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
    "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)"
};
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
    ## Image rotation, translation, flipping:
    iaa.OneOf([
        iaa.Fliplr(0.2),
        iaa.Affine(
            scale = {"x": (0.99,1.01), "y":(0.98,1.03)},
            translate_percent = {"x": (-0.025,0.025), "y": (-0.05,0.05)},
            rotate = (-3, 3),
        ),
    ]),
    ## Adjustment of brightness, contrast norm, or resolution:
    iaa.OneOf([
        iaa.ContrastNormalization((0.75, 1.05)),
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
    ]),
]);

## Classes & templates:
class TrainingConfig(Config):
    """ MaskRCNN class for model training """
    NAME = DATASET_NAME;
    BACKBONE = "resnet50"; #default: Resnet-101

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
    """ Custom data class """
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

class MaskRCNN:
    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        assert self.mode == "training", "Create model in training mode."
        if layers in LAYERS_REGEX.keys():
            layers = LAYERS_REGEX[layers]

        # Data generators
        train_generator = data_generator(
            train_dataset,
            self.config,
            shuffle = True,
            augmentation = augmentation,
            batch_size = self.config.BATCH_SIZE,
            no_augmentation_sources = no_augmentation_sources
        );
        val_generator = data_generator(
            val_dataset,
            self.config,
            shuffle = True,
            batch_size = self.config.BATCH_SIZE
        );

        ## Create log_dir if it does not exist:
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir);

        ## Implement custom callbacks:
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path, verbose=1, save_weights_only=True),
        ];
        if custom_callbacks:
            callbacks += custom_callbacks;

        ## Train dense model:
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate));
        log("Checkpoint Path: {}".format(self.checkpoint_path));
        self.set_trainable(layers);
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM);

        workers = 0 if os.name is 'nt' else multiprocessing.cpu_count();
        print("Number of workers to use: %d" % workers)

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size = 100,
            workers = 1,
            use_multiprocessing = False,
        );
        self.epoch = max(self.epoch, epochs);