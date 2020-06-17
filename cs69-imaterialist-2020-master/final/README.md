# Finalized Code Repository

**Team: Sunint B. &amp; David C.**

## Overview

This sub-directory consists of code for Keras-implementation of Mask RCNN for object categorization. Each image in the training data has one and only one label (`ClassId`). Prior to any training attempts, we split a test set with ground-truth label for final evaluation purposes.

This _final_ repository has the theme of a set of **experiments**, using n=3200 labeled images as a _Simulated Unknown Test Set_:

* Compare ResNet-50 architecture with vs. without fine-tuning
* Compare ResNet-50 (shallow) vs. ResNet-101 (deeper) as backbone

Note that in each experiment, there is only ONE _experimental variable_; everything else, e.g. training-validating-testing images, are _identical_. We used multi-class accuracy as the metric for evaluation.


## Dependencies

```shell script
pip3 uninstall tensorflow
pip3 install tensorflow==1.5 #python 3.6x; or in python3.7: pip3 install tensorflow==1.14.0
pip3 uninstall keras
pip3 install keras==2.1.5
pip3 install mrcnn
pip3 install imgaug
wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
```

## Procedure

To satisfy requirement for using GPUs, we have abstracted our Masked RCNN as separate _.py_ scripts, each consisting of a `main()` procedure. _Click on any script to view details._

**Phase 1.** `train.py`

* Train-validate-test split, with meta-data saved to CSV
* Exports trained weights as _.h5_
* Saves _Keras history_ object to _.pkl_ file

**Phase 2.** `predict.py`

* Saves results to CSV file
* Currently, only the model for _ClassId_ were fully developed

Pre-defined custom classes &amp; templates, as well as reusable objects, are implemented in `classes.py` to be loaded within either or both steps,

```shell script
python3 train.py
python3 predict.py
```

Note that only _Python3.x_ and legacy versions of _Keras_ &amp; _Tensorflow_ are supported.


## Note on GPU Usage

We made use of Dartmouth Research Computing GPU. Specific details as to the CPU &amp; GPU server is specified in the `PBS` script. Here, our GPU repository has the following structure:

```
|-- classes.py
|-- Expt_Resnet50+pretrained.o1762065
|-- predict.py
|-- queue.pbs
```

To initiate the GPU computation, submit the PBS script `queue.pbs` ([official instructions from Dartmouth Research Computing](https://rc.dartmouth.edu/index.php/using-discovery/using-the-gpu-nodes/job-submission-template/)).

```shell script
## Step 1. Virtual environment to be activated in queue.pbs
## This step only needs to be invoked ONCE
conda create --name <environment-name>

## Step 2. Submit job (lines 34-35 most important)
qsub queue.pbs
```

Since our team is permitted to queue make use of 1 GPU Job per queue, we repeated the procedure multiple times for multiple experiments.


## Data Exploration &amp; Visualization on Our Local Machine (Jupyter Notebook)

We used _Jupyter Notebook_ for interactively making figures and plots for the presentation. Simply click on the following _.ipynb_ files to see the assets:

* [visualization_image_eda.ipynb](https://gitlab.cs.dartmouth.edu/ydavidchen/cs69-imaterialist-2020/blob/master/final/visualization_image_eda.ipynb)
* [visualization_experiments_trainval_losses.ipynb](https://gitlab.cs.dartmouth.edu/ydavidchen/cs69-imaterialist-2020/blob/master/final/visualization_experiments_trainval_losses.ipynb)
* [visualization_experiments_testset.ipynb](https://gitlab.cs.dartmouth.edu/ydavidchen/cs69-imaterialist-2020/blob/master/final/visualization_experiments_testset.ipynb)


## References

* Shapes starter code / tutorial using Mask RCNN
* Mask RCNN starter code on Kaggle iMaterialist 2019 &amp; 202
* Dartmouth Research Computing GPU &amp; Linux (CPU) tutorial
