'''
Author: Minho Kim, Doyoung Jeong
Contact: mhk93@snu.ac.kr

Based on: https://github.com/ChunpingQiu/benchmark-on-So2SatLCZ42-dataset-a-simple-tour/blob/master/lr.py

'''

from tensorflow.python.keras.utils.data_utils import Sequence
import h5py
import numpy as np

def generator(h5path, batchSize=32, num=None):

    db = h5py.File(h5path, "r")

    indices=np.arange(num)

    while True:

        np.random.shuffle(indices)
        for i in range(0, len(indices), batchSize):

            batch_indices = indices[i:i+batchSize]
            batch_indices.sort()

            by = db["label"][batch_indices,:]
            bx = db["sen2"][batch_indices,:,:,:]

            yield (bx,by)

def alltime_generator(h5path, batchSize=32, num=None, extract_ratio=0.1):

    db = h5py.File(h5path, "r")

    indices=np.arange(num)
    
    while True:
        
        np.random.shuffle(indices)
        for i in range(0, round(len(indices)*extract_ratio), batchSize):

            batch_indices = indices[i:i+batchSize]
            batch_indices.sort()

            by = db["label"][batch_indices,:]
            
            x1 = db["0"][batch_indices,:,:,:]
            x2 = db["1"][batch_indices,:,:,:]
            x3 = db["2"][batch_indices,:,:,:]
            bx = np.concatenate([x1,x2,x3], axis=-1)

            yield (bx,by)
            
class MyGenerator(Sequence):

    def __init__(self, h5path, batch_size=96, augmentations=None, shuffle=True):
        
        db = h5py.File(h5path, "r")
        
        self.x = db['sen2']
        self.y = db['label']
        self.batch_size = batch_size
        self.augment = augmentations
        self.shuffle = shuffle
        self.on_epoch_end()
        
        self.n = 0
        self.max = self.__len__()
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        inds = sorted(self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size])
        # inds = sorted(inds)
        batch_x = self.x[inds]
        batch_y = self.y[inds]
        
        return np.stack([
            self.augment(image=x)["image"] for x in batch_x
        ], axis=0), np.array(batch_y)
    
    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result