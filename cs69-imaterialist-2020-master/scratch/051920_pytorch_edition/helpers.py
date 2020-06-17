import numpy as np;
import pandas as pd;
import cv2;
import bz2;
import pickle;
import gc;
import math;
import torch;

def load_train_annot(data_dir, fname="train.csv"):
    return pd.read_csv(data_dir+fname);

def compress(raw_obj):
    return bz2.compress(pickle.dumps(raw_obj), 3);

def decompress(comp_obj):
    return pickle.loads(bz2.decompress(comp_obj));

def reset_cache():
    gc.collect();
    try:
        torch.cuda.empty_cache();
    except:
        pass;
    return None;

def rle_to_mask(rle_string,height,width):
    rows, cols = height, width
    if rle_string == -1:
        return np.zeros((height, width))
    else:
        rleNumbers = [int(numstring) for numstring in rle_string.split(' ')]
        rlePairs = np.array(rleNumbers).reshape(-1,2)
        img = np.zeros(rows*cols,dtype=np.uint8)
        for index,length in rlePairs:
            index -= 1
            img[index:index+length] = 255
        img = img.reshape(cols,rows)
        img = img.T
        return img

def mask_to_rle(mask):
    pixels = mask.T.flatten()
    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle -= 1;
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return ' '.join(str(x) for x in rle);

def scale_image(img, long_size):
    ## Compute new size:
    if img.shape[0] < img.shape[1]:
        scale = img.shape[1] / long_size;
        new_size = (long_size, math.floor(img.shape[0] / scale));
    else:
        scale = img.shape[0] / long_size;
        new_size = (math.floor(img.shape[1] / scale), long_size);

    img = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST);
    return img;