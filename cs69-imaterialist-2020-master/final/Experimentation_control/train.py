#!/usr/bin/env python

from classes import *

import keras
import mrcnn.model as modellib
import tensorflow
from tensorflow.python.keras.engine import saving
import pickle

def main():
    print(tensorflow.__version__) #1.5 or 1.14 needed
    print(keras.__version__) #2.1.5 needed
    keras.engine.saving = saving; #failed to work on tensorflow1.5
    os.listdir(DATA_DIR)

    print("INFO: Splitting training-validation-testing data subsets...")
    train_data = pd.read_csv(os.path.join(DATA_DIR,"train.csv"));
    if N_SAMP is not None:
        train_data = train_data.sample(N_SAMP);

    images_train, images_val, _  = train_val_test_split(train_data);

    print("INFO: Preparing data for training-validation...")
    dataset_train = CustomDataset(images_train);
    dataset_train.prepare();

    dataset_val = CustomDataset(images_val);
    dataset_val.prepare();

    print("INFO: Configure / initialize model to train...")
    config = TrainingConfig();
    config.display()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=WORKING_DIR);

    if MODE == "pretrained":
        print("INFO: Fine-tune with pre-trained weights")
        model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=COCO_EXCLUDE_COLS);
    else:
        warnings.warn("INFO: No pre-training!");

    print("INFO: Begin training model!")
    model.train(
        dataset_train,
        dataset_val,
        learning_rate = LEARNING_RATE,
        epochs = EPOCHS,
        layers = "heads",
        augmentation = DATA_AUG
    );
    history = model.keras_model.history.history;

    print("INFO: Exporting model weights & epoch history...")
    model.keras_model.save_weights(MODEL_NAME_CUSTOM +"_weights_" + MODE + "_" + BACKBONE + ".h5");
    pickle.dump(history, open(MODEL_NAME_CUSTOM +"_history_" + MODE + "_" + BACKBONE + ".pkl", "ab"));
    print("Training Phase Complete!");


main();