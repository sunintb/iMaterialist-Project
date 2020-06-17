
import os;
import cv2;
from helpers import *;

class AttributesDataset(object):
    def __init__(self, id_cls, clz_attr_num, clz_attrid2idx, caches):
        self.id_cls = id_cls;
        self.clz_attrid2idx = clz_attrid2idx;
        self.clz_attr_num = clz_attr_num;
        self.caches = [cd for cd in caches if cd[0] == id_cls];

    def __getitem__(self, idx, dtype=np.float32):
        cid, mask, attr, img, X, id_img = self.caches[idx];
        mask, img = decompress(mask), decompress(img);
        X = img if X is None else decompress(X);
        y = np.zeros(self.clz_attr_num[self.id_cls]);
        for a in attr:
            y[self.clz_attrid2idx[self.id_cls].index(a)] = 1;
        return X.transpose((2,0,1)).astype(dtype), y.astype(dtype);

    def __len__(self):
        return len(self.caches);

class MaskTrainingSet(object):
    def __init__(self, keys, mask):
        self.keys = keys;
        self.mask = mask;

    def __getitem__(self, idx):
        k = self.keys[idx];
        return decompress(self.mask[k][0]), decompress(self.mask[k][1]);

    def __len__(self):
        return len(self.keys);

class MaskTestSet(object):
    def __init__(self, subdir, img_dim, n_sample=None):
        self.subdir = subdir;
        self.img_dim = img_dim;
        self.imgids = [f.split(".")[0] for f in os.listdir(subdir)];
        if n_sample is not None:
            print("Testing on a subset of n=%d" % n_sample)
            self.imgids = np.random.choice(self.imgids, n_sample);

    def __getitem__(self, idx):
        imag = cv2.imread(self.subdir + self.imgids[idx] + ".jpg");
        imag = cv2.resize(imag, self.img_dim);
        return imag.transpose((2, 0, 1)).astype(np.float32);

    def __len__(self):
        return len(self.imgids);