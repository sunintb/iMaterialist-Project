#!/usr/bin/env python

# pip3 uninstall tensorflow
# pip3 install tensorflow==1.5 #python 3.6x; or in python3.7: pip3 install tensorflow==1.14.0
# pip3 uninstall keras
# pip3 install keras==2.1.5
# pip3 install mrcnn
# pip3 install imgaug
# wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

## If Kaggle limits, replace the following line with contents inside
from classes import *

import keras
import mrcnn.model as modellib
import tensorflow
from tensorflow.python.keras.engine import saving
import pickle
from sklearn.utils import shuffle

def main():
    print(tensorflow.__version__)  #1.5 or 1.14 needed
    print(keras.__version__)  #2.1.5 needed
    keras.engine.saving = saving;  #failed to work on tensorflow1.5
    os.listdir(DATA_DIR)

    print("INFO: Loading & preparing training data...")
    train_data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"));
    if N_SAMP is not None:
        train_data = train_data.sample(N_SAMP);

    images_data = train_data.groupby('ImageId')['EncodedPixels', 'ClassId'].agg(lambda vec: list(vec));
    dimensions_data = train_data.groupby('ImageId')['Height', 'Width'].mean();
    images_data = images_data.join(dimensions_data, on='ImageId');
    images_data.head()
    print("Total images: %d" % len(images_data))

    images_data_shuffled = shuffle(images_data);
    val_size = int(0.10 * len(images_data_shuffled['ClassId']));
    image_data_val = images_data_shuffled[:val_size];
    image_data_train = images_data_shuffled[val_size:];
    print("n_train=%d, n_valid=%d" % (len(image_data_train),len(image_data_val)))

    dataset_train = CustomDataset(image_data_train);
    dataset_train.prepare();

    dataset_val = CustomDataset(image_data_val);
    dataset_val.prepare();

    print("INFO: Configure / initialize model to train...")
    config = TrainingConfig();
    config.display()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=WORKING_DIR);

    print("INFO: Fine-tune with pre-trained weights")
    model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=COCO_EXCLUDE_COLS);

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
    model.keras_model.save_weights(MODEL_NAME_CUSTOM + "_weights.h5");
    pickle.dump(history, open(MODEL_NAME_CUSTOM+"_history.pkl","ab"));

main();