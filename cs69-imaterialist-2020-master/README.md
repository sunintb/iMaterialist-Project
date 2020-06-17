# iMaterialist Kaggle Challenge 2020

Sunint Bindra and David Chen | COSC69.9/169.9 at Dartmouth

## Overview

This is the code repository for the course final project.

## File Structure

* `scratch/`: This is the main directory that shows intermediate progress and updates
* `show-n-tell`: Code repository for the procedure successfully tested and chosen
* `website`: HTML/CSS/Javascript source code for our overview web

## Timeline

* May 18-19, 2020
  - Reviewed [Regional Convolutional Neural Network](https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html#r-cnn) and associated papers
  - Explored the data (images) and labels for the training data set, as well as publicly available works/discussions done by fellow Kaggle competitors
  - Applied [matterport's implementation of Mask RCNN](https://github.com/matterport/Mask_RCNN), end-to-end
  - `TODO`: Still troubleshooting how to create submission file
  - `DEBUG`: Bounding boxes were not predicted correctly
* May 20-23
  - Attempted to address the pixel-by-pixel mask prediction problem
  - Explored use of PyTorch
  - Successfully employed GPU, but failed to complete the task within 48 hours
* May 24: Check-in with course director &amp; TA
* May 25-27: Revert to _Mask RCNN_

## Major Challenges

1. Mask prediction is labor- and computationally intensive
2. GPU usage on Kaggle is limited to _30 hours / month_. Countdown starts whenever a Jupyter notebook is tagged
  - To address this, we leveraged Dartmouth Research Computing community GPU, which supports real-time testing and job submission-based offline running
3. Kaggle working directory (`/kaggle/working`) and workspace memory are limited
4. _PyCUDA_ GPU memory is also limited.
5. Some packages (e.g. MaskRCNN) does not support multiple GPUs and parallel programming

## Note on GPU usage at Dartmouth

For code development and model hyperparameter optimization, we instead used GPU resources at Dartmouth. Specificially,

1. Virtual environment that allows us to bypass the painstaking permission issues associated with shared cluster usage
2. [High Performance Computing with GPU usage at Dartmouth](https://rc.dartmouth.edu/index.php/using-discovery/using-the-gpu-nodes/)

Our virtual environment setup (only Python 3.6x available on cluster):


```shell
ssh -Y <cluster-login-info>
# ssh g01 #g01 to g12 available

cd ~
mkdir imaterialist
cd imaterialist/
# module list
# module avail
module swap python/3.6-Miniconda python/3.6-GPU

# Create virtual environment
conda create --name kaggle
source activate kaggle

# conda package installations
# Note pip or pip3 install still have permission issues
conda install torch
conda install torchvision
conda install -c conda-forge tqdm
conda install pandas
conda install -c conda-forge matplotlib
conda install -c intel scikit-learn

# cv2 package requires special procedure
conda update freetype
conda install opencv -c conda-forge
# ...

# PyCuda
conda install -c lukepfister pycuda

# To execute code:
python3 <script-name.py>

## To use Kaggle API:
pip install kaggle
kaggle competitions download imaterialist-fashion-2020-fgvc7
# put kaggle.json into .kaggle/


# When finish:
source deactivate kaggle
```
