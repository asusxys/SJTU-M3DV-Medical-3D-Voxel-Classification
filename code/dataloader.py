import random
import os
import numpy as np
import pandas as pd
from collections.abc import Sequence
from mylib.utils.misc import rotation, reflection, crop, random_center, _triple

# define the directory of train_val.csv
path = "./dataset/"
TRAIN = pd.read_csv(os.path.join(path, 'train_val.csv'))
TEST = pd.read_csv(os.path.join(path, 'test.csv'))

'''
Load train and validation dataset
'''
class ClfDataset(Sequence):
    def __init__(self, subset=[0, 1, 2, 3]):
        #self.index = tuple(TRAIN.index)
        index = []
        for sset in subset:
            index += list(TRAIN[TRAIN['subset'] == sset].index)
        self.index = tuple(sorted(index))  # the index in the info
        
        self.label = np.array(TRAIN.loc[self.index, 'label']) # label : 0/1
        #self.transform = Transform(crop_size, move)  # 变换
        #self.cutflag = cut

    def __getitem__(self, item):
        name = TRAIN.loc[self.index[item], 'name']
        
        # 加载npz数据
        with np.load(os.path.join("./dataset/train_val", '%s.npz' % name)) as npz:
            #voxel = self.transform(npz['voxel'])
            voxel = npz['voxel']

        label = self.label[item]
        return voxel, label

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        for x, y in data:
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

# include segmentation
class ClfSegDataset(ClfDataset):
    def __getitem__(self, item):
        name = TRAIN.loc[self.index[item], 'name']
        with np.load(os.path.join("./dataset/train_val", '%s.npz' % name)) as npz:
            #voxel, seg = self.transform(npz['voxel'], npz['seg'])
            voxel, seg = npz['voxel'], npz['seg']
            
        label = self.label[item]
        return voxel, (label, seg)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        segs = []
        for x, y in data:
            xs.append(x)
            ys.append(y[0])
            segs.append(y[1])
        return np.array(xs), {"clf": np.array(ys), "seg": np.array(segs)}
        
        
class ClfvalDataset(Sequence):
    def __init__(self, crop_size=32, move=3, subset=[0, 1, 2, 3]):
        #self.index = tuple(TRAIN.index)
        index = []
        for sset in subset:
            index += list(TRAIN[TRAIN['subset'] == sset].index)
        self.index = tuple(sorted(index))  # the index in the info
        
        self.label = np.array(TRAIN.loc[self.index, 'label']) # label : 0/1
        self.transform = Transform(crop_size, move)  # 变换

    def __getitem__(self, item):
        name = TRAIN.loc[self.index[item], 'name']
        
        # 加载npz数据
        with np.load(os.path.join("./dataset/train_val", '%s.npz' % name)) as npz:
            voxel = self.transform(npz['voxel'])

        label = self.label[item]
        return voxel, label

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        for x, y in data:
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

# include segmentation
class ClfvalSegDataset(ClfvalDataset):
    def __getitem__(self, item):
        name = TRAIN.loc[self.index[item], 'name']
        with np.load(os.path.join("./dataset/train_val", '%s.npz' % name)) as npz:
            voxel, seg = self.transform(npz['voxel'], npz['seg'])
            
        label = self.label[item]
        return voxel, (label, seg)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        segs = []
        for x, y in data:
            xs.append(x)
            ys.append(y[0])
            segs.append(y[1])
        return np.array(xs), {"clf": np.array(ys), "seg": np.array(segs)}
'''
Load test dataset
'''
class ClfTestDataset(Sequence):
    def __init__(self, crop_size=32, move=3):
        self.index = tuple(TEST.index)
        self.label = np.array(TEST.loc[self.index, 'label']) # label : 0/1
        self.transform = Transform(crop_size, move)  # 变换

    def __getitem__(self, item):
        name = TEST.loc[self.index[item], 'name']
        
        # 加载npz数据
        with np.load(os.path.join("./dataset/test", '%s.npz' % name)) as npz:
             voxel = self.transform(npz['voxel'])
        label = self.label[item]
        return voxel, label

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        for x, y in data:
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

# include segmentation
class ClfSegTestDataset(ClfTestDataset):
    def __getitem__(self, item):
        name = TEST.loc[self.index[item], 'name']
        with np.load(os.path.join("./dataset/test", '%s.npz' % name)) as npz:
            voxel, seg = self.transform(npz['voxel'], npz['seg'])
        label = self.label[item]
        return voxel, (label, seg)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        segs = []
        for x, y in data:
            xs.append(x)
            ys.append(y[0])
            segs.append(y[1])
        return np.array(xs), {"clf": np.array(ys), "seg": np.array(segs)}

'''
dataloader
'''
def get_loader_inorder(dataset, batch_size):
    total_size = len(dataset)
    print('Size', total_size)
    index_generator = order_iterator(range(total_size))
    while True:
        data = []
        for _ in range(batch_size):
            idx = next(index_generator)
            '''
            print(idx)
            names = TEST.loc[idx, 'name']
            name1 = [[names]]
            dt = pd.DataFrame(data = name1,columns=['Id'])
            dt.to_csv('Submission.csv',mode ='a',index=False,header=False)
            '''
            data.append(dataset[idx])
        yield dataset._collate_fn(data)
    

def get_loader(dataset, batch_size):
    total_size = len(dataset)
    print('Size', total_size)
    index_generator = shuffle_iterator(range(total_size))
    while True:
        data = []
        # 0,1,2...31
        for _ in range(batch_size):
            idx = next(index_generator)
            data.append(dataset[idx])
        yield dataset._collate_fn(data)
'''
use mixup method
'''
def get_mixup_loader(dataset, batch_size, alpha=1.0):
    total_size = len(dataset)
    print('Size', total_size)
    index_generator = shuffle_iterator(range(total_size))
    transform = Transform([32,32,32], 3)
    
    while True:
        data = []
        lam = np.random.beta(alpha, alpha, batch_size)
        #print(lam)
        for i in range(batch_size):
            X_l = lam[i]
            y_l = lam[i]
            
            idx = next(index_generator)

            idx1 = next(index_generator)
            
            data0 = dataset[idx]
            data1 = dataset[idx1]
            
            newdata = data0[0] * X_l + data1[0] * (1 - X_l)
            
            lb = y_l*data0[1][0]+(1-y_l)*data1[1][0]
            
            seg = X_l*data0[1][1]+(1-X_l)*data1[1][1]
            
            newdata = transform(newdata)
            seg = transform(seg)
            
            datanew = (newdata,(lb,seg))
            data.append(datanew)
        yield dataset._collate_fn(data)
        
'''
data augmentation
'''

'''
@author: duducheng
'''
class Transform:
    '''The online data augmentation, including:
    1) random move the center by `move`
    2) rotation 90 degrees increments
    3) reflection in any axis
    '''

    def __init__(self, size, move):
        self.size = _triple(size)
        self.move = move

    def __call__(self, arr, aux=None):
        shape = arr.shape
        if self.move is not None:
            center = random_center(shape, self.move)
            arr_ret = crop(arr, center, self.size)
            angle = np.random.randint(4, size=3)
            arr_ret = rotation(arr_ret, angle=angle)
            axis = np.random.randint(4) - 1
            arr_ret = reflection(arr_ret, axis=axis)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = rotation(aux_ret, angle=angle)
                aux_ret = reflection(aux_ret, axis=axis)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret
        else:
            center = np.array(shape) // 2
            arr_ret = crop(arr, center, self.size)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret

'''
@author: https://github.com/hysts/pytorch_cutout
'''
def cutout(image,mask_size=12, p=0.5, cutout_inside=False, mask_color=0):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0
    image = np.asarray(image).copy()

    if np.random.random() > p:
        return image

    h, w = image.shape[:2]

    if cutout_inside:
        cxmin, cxmax = mask_size_half, w + offset - mask_size_half
        cymin, cymax = mask_size_half, h + offset - mask_size_half
    else:
        cxmin, cxmax = 0, w + offset
        cymin, cymax = 0, h + offset

    cx = np.random.randint(cxmin, cxmax)
    cy = np.random.randint(cymin, cymax)
    xmin = cx - mask_size_half
    ymin = cy - mask_size_half
    xmax = xmin + mask_size
    ymax = ymin + mask_size
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)
    image[ymin:ymax, xmin:xmax, :] = mask_color
    return image
 
'''
@author: duducheng
'''
def shuffle_iterator(iterator):
    # iterator should have limited size
    index = list(iterator)   # 0~464
    total_size = len(index)
    i = 0
    random.shuffle(index)    # 随机打乱
    
    while True:
        yield index[i]
        i += 1
        if i >= total_size:
            i = 0
            random.shuffle(index)


def order_iterator(iterator):
    index = list(iterator)   # 0~464
    total_size = len(index)
    i = 0
    '''
    random.shuffle(index)
    names = TEST.loc[index, 'name']
    name1 = np.array(names)
    dt = pd.DataFrame(data = name1,columns=['Id'])
    dt.to_csv('Submission.csv',index=False)
    '''

    while True:
        yield index[i]
        #print(i)
        i += 1
        if i >= total_size:
            i = 0
            '''
            random.shuffle(index)
            names = TEST.loc[index, 'name']
            name1 = np.array(names)
            dt = pd.DataFrame(data = name1,columns=['Id'])
            dt.to_csv('Submission.csv',mode ='a',index=False,header=False)
            '''
