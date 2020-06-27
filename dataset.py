import numpy as np
from os.path import join as pjoin
import torch
from sklearn.model_selection import train_test_split
import struct


class Mnist(object):
    def __init__(self, root, mode='train', ratio=0.2):
        self.n_class = 10
        if mode == 'train':
            img_path = pjoin(root, 'train-images-idx3-ubyte')
            lab_path = pjoin(root, 'train-labels-idx1-ubyte')
            with open(lab_path, 'rb') as lbpath:
                magic, n = struct.unpack('>II', lbpath.read(8))
                train_y = np.fromfile(lbpath, dtype=np.uint8)
            with open(img_path, 'rb') as imgpath:
                magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
                train_x = np.fromfile(imgpath,dtype=np.uint8).reshape(len(train_y), 1, 28, 28)
            train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=ratio, random_state=2)
            train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
            train_y = torch.from_numpy(train_y).type(torch.LongTensor)
            self.train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
            val_x = torch.from_numpy(val_x).type(torch.FloatTensor)
            val_y = torch.from_numpy(val_y).type(torch.LongTensor)
            self.val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
        else:
            img_path = pjoin(root, 't10k-images-idx3-ubyte')
            lab_path = pjoin(root, 't10k-labels-idx1-ubyte')
            with open(lab_path, 'rb') as lbpath:
                magic, n = struct.unpack('>II', lbpath.read(8))
                test_y = np.fromfile(lbpath, dtype=np.uint8)
            with open(img_path, 'rb') as imgpath:
                magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
                test_x = np.fromfile(imgpath, dtype=np.uint8).reshape(len(test_y), 1, 28, 28) 
            test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
            test_y = torch.from_numpy(test_y).type(torch.LongTensor)
            self.test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
            