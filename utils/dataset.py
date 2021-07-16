import os
import json
import random
import numpy as np
import jittor as jt
from jittor.dataset import Dataset
from PIL import Image
from os.path import join


def isImageFile(filename):
    IMAGE_EXTENSIONS = ['.jpg','.png','.bmp','.tif','.tiff','.jpeg']
    return any([filename.lower().endswith(extension) for extension in IMAGE_EXTENSIONS])

def isNumpyFile(filename):
    return filename.lower().endswith('.npy')

class SingleDataset(Dataset):
    def __init__(self, json_dir, img_dir, mask_dir, batch_size=1, shuffle=False, train = True, logger=None, aug=None, img_suffix='.jpg', label_suffix='.npy'):
        super().__init__()

        self.train = train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aug = aug

        self.img_file_names = []
        self.mask_file_names = []
        img_dir_dict = json.load(open(json_dir, "r"))
        for person_num in img_dir_dict:
            person = img_dir_dict[person_num]
            for ct in person:
                self.img_file_names.extend([join(img_dir, file_name.replace(label_suffix, img_suffix)) for file_name in person[ct]["list"] if isNumpyFile(file_name)])
                self.mask_file_names.extend([join(mask_dir, file_name) for file_name in person[ct]["list"] if isNumpyFile(file_name)])

        if logger is not None:
            logger.info(f'Finished creating an instance of {self.__class__.__name__} with {len(self.img_file_names)} examples')
        else:
            print(f'Finished creating an instance of {self.__class__.__name__} with {len(self.img_file_names)} examples')

        assert(len(self.mask_file_names) == len(self.img_file_names))
        self.total_len = len(self.img_file_names)
        # this function must be called
        self.set_attrs(batch_size = self.batch_size, total_len=self.total_len, shuffle= self.shuffle)

    def __getitem__(self, index):
        img, mask = self.fetch(img_path = self.img_file_names[index], mask_path = self.mask_file_names[index])

        if self.aug is not None:
            img = self.aug(img)
        else:
            img = np.array(img).astype(np.float32)
            img /= 255.0

            img = np.array(img).astype(np.float32).transpose(2, 0, 1)
        img = jt.array(img)
        mask = jt.array(np.array(mask).astype(int))
        return img, mask

    # 读入数据
    def fetch(self, img_path, mask_path):
        with open(img_path, 'rb') as fp:
            img = Image.open(fp).convert('RGB')

        with open(mask_path, 'rb') as fp:
            mask = np.load(mask_path)

        return img, mask

    # 随机水平翻转
    def flip(self, img, mask):
        if random.random() < 0.4:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = np.flip(mask, axis=1)      # 水平翻转
        return img, mask

    # 随机正则化
    def normalize(self, img, mask):
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        img /= 255.0
        if random.random() < 0.4:
            mean = (0.485, 0.456, 0.40)
            std = (0.229, 0.224, 0.225)
            img -= mean
            img /= std
        return img, mask


class PaintContourDataset(Dataset):
    def __init__(self, json_dir, img_dir, mask_dir = None, batch_size=1, shuffle=False, logger=None, img_suffix='.jpg', label_suffix='.npy'):
        super().__init__()

        self.mask_flag = False if mask_dir == None else True
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.img_file_names = []
        self.mask_file_names = []
        img_dir_dict = json.load(open(json_dir, "r"))
        for person_num in img_dir_dict:
            person = img_dir_dict[person_num]
            for ct in person:
                self.img_file_names.extend([join(img_dir, file_name.replace(label_suffix, img_suffix)) for file_name in person[ct]["list"] if isNumpyFile(file_name)])
                if self.mask_flag:
                    self.mask_file_names.extend([join(mask_dir, file_name) for file_name in person[ct]["list"] if isNumpyFile(file_name)])
                    assert(len(self.mask_file_names) == len(self.img_file_names))

        if logger is not None:
            logger.info(f'Finished creating an instance of {self.__class__.__name__} with {len(self.img_file_names)} examples')
        else:
            print(f'Finished creating an instance of {self.__class__.__name__} with {len(self.img_file_names)} examples')

        self.total_len = len(self.img_file_names)
        # this function must be called
        self.set_attrs(batch_size = self.batch_size, total_len=self.total_len, shuffle= self.shuffle)

    def __getitem__(self, index):
        with open(self.img_file_names[index], 'rb') as fp:
            img = Image.open(fp).convert('RGB')
        img_ = np.array(img).astype(np.float32)
        img_ /= 255.0
        img_ = np.array(img_).astype(np.float32).transpose(2, 0, 1)
        img_ = jt.array(img_)

        if self.mask_flag:
            mask = np.load(self.mask_file_names[index])
            mask = jt.array(np.array(mask).astype(int))
            return self.img_file_names[index], img, img_, mask
        else:
            return self.img_file_names[index], img, img_


subs = {
        'train': 'train_label.json',
        'val': 'val_label.json',
        'test': 'test_label.json',
        'img': 'data',
        'mask': 'label'
    }


dataset_dirs = {
    'xh': 'data/data-lumber/data-XH',
    'zs_big': 'data/data-lumber/data-ZS/Big',
    'zs_small': 'data/data-lumber/data-ZS/Small',
    'hard': 'data/data-lumber/data-hard',
    'pancreas': 'data/data-pancreas'
}


def get_suffix(dataset_name):
    return '.png' if dataset_name == 'pancreas' else '.jpg'


dataset_choices = ['xh', 'xh_hard', 'zs_big', 'zs_small', 'pancreas']


def retrieve_sub_names(dataset_name):
    dir_name = dataset_dirs[dataset_name]
    sub_names = {}
    for keys, values in subs.items():
        sub_names[keys] = os.path.join(dir_name, values)
    return sub_names


def retrieve_data(args, split, aug, paint=False, mask=True):
    sub_names = retrieve_sub_names(args.dataset)
    assert split in ['train', 'val', 'test']
    if paint:
        loader = PaintContourDataset(
            json_dir = sub_names[split], img_dir = sub_names['img'], 
            mask_dir = sub_names['mask'] if mask else None, 
            batch_size = 1, shuffle = False, 
            img_suffix=get_suffix(args.dataset)
        )
    else:
        loader = SingleDataset(
            json_dir = sub_names[split], img_dir = sub_names['img'], mask_dir = sub_names['mask'], 
            batch_size = args.batch_size if split == 'train' else 1, 
            shuffle = True if split == 'train' else False, 
            train = True if split == 'train' else False, 
            aug=aug, img_suffix=get_suffix(args.dataset)
        )
    return loader
    
    
    