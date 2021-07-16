from PIL import ImageFilter
import random
from jittor import transform
from PIL import Image
import numpy as np
import json
import random
import numpy as np
from PIL import Image
from jittor.dataset import Dataset
from os.path import join
from utils import retrieve_sub_names, get_suffix


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
    
normalize = transform.ImageNormalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


augmentation = transform.Compose([
    transform.RandomCropAndResize((512, 512), scale=(0.2, 1.0)), 
    transform.RandomApply([transform.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), 
    transform.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5), 
    transform.RandomHorizontalFlip(), 
    transform.ToTensor(),
    normalize
])


aug_for_unet = transform.Compose([
    transform.RandomApply([transform.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), 
    transform.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5), 
    transform.ToTensor(),
])


aug_train = transform.Compose([
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5], std=[0.5])
])


def isImageFile(filename):
    IMAGE_EXTENSIONS = ['.jpg','.png','.bmp','.tif','.tiff','.jpeg']
    return any([filename.lower().endswith(extension) for extension in IMAGE_EXTENSIONS])


def isNumpyFile(filename):
    return filename.lower().endswith('.npy')


class AugDataset(Dataset):
    def __init__(self, json_dir, img_dir, mask_dir, batch_size=32, shuffle=False, aug=augmentation, img_suffix='.jpg', label_suffix='.npy'):
        super(AugDataset, self).__init__()
        self.batch_size = batch_size
        self.aug = aug
        self.shuffle = shuffle
        
        self.img_file_names = []
        self.mask_file_names = []
        img_dir_dict = json.load(open(json_dir, "r"))
        for person_num in img_dir_dict:
            person = img_dir_dict[person_num]
            for ct in person:
                self.img_file_names.extend([join(img_dir, file_name.replace(label_suffix, img_suffix)) for file_name in person[ct]["list"] if isNumpyFile(file_name)])
                self.mask_file_names.extend([join(mask_dir, file_name) for file_name in person[ct]["list"] if isNumpyFile(file_name)])
        assert(len(self.mask_file_names) == len(self.img_file_names))
        self.total_len = len(self.img_file_names)
        self.set_attrs(batch_size = self.batch_size, total_len=self.total_len, shuffle= self.shuffle)

    def __getitem__(self, index):
        img, mask = self.fetch(img_path = self.img_file_names[index], mask_path = self.mask_file_names[index])
        query, key = self.aug(img)
        return query, key, mask

    # 读入数据
    def fetch(self, img_path, mask_path):
        with open(img_path, 'rb') as fp:
            img = Image.open(fp).convert('RGB')
        with open(mask_path, 'rb') as fp:
            mask = np.load(mask_path)
        return img, mask


def retrieve_aug_data(args, split, aug):
    sub_names = retrieve_sub_names(args.dataset)
    assert split in ['train', 'val', 'test']
    loader = AugDataset(
        json_dir = sub_names[split], img_dir = sub_names['img'], mask_dir = sub_names['mask'], 
        batch_size = args.batch_size, 
        shuffle = True, 
        aug=TwoCropsTransform(aug), 
        img_suffix=get_suffix(args.dataset)
    )
    return loader