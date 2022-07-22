import os
import random
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils.weight_mask import Weight_Mask, align_face, corp_face
from PIL import Image
import numpy as np
import csv


class ImageData(Dataset):
    def __init__(self, data_path, img_size=224, train=False, rgb_means=(0.456, 0.456, 0.456),
                 std=(0.225, 0.225, 0.225)):
        if train:
            print('loading training data.....')
            csv_path = os.path.join(data_path, 'csv_file', 'training.csv')
            self.anno_data = self.load_anno(csv_path)
        else:
            print('loading validation data.....')
            self.anno_data = self.load_anno_ck()

        self.length = len(self.anno_data)
        self.data_path = data_path
        self.resize = transforms.Compose([transforms.Resize((img_size, img_size)), ])
        self.train = train
        self.img_size = img_size
        self.weight_mask = Weight_Mask()
        self.rgb_means = rgb_means
        self.std = std
        self.train_transform = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5), ])

    def __getitem__(self, index):
        anno_info = self.anno_data[index]
        img = cv2.imread(os.path.join(anno_info['path']))
        lmk = np.array(anno_info['facial_landmarks'])
        label = anno_info['expression']
        # align_face
        img, lmk = align_face(img, lmk)
        img, lmk = corp_face(img, lmk)

        # resize_face
        img_h, img_w, _ = img.shape
        lmk[:, 0] = lmk[:, 0] * self.img_size / img_w
        lmk[:, 1] = lmk[:, 1] * self.img_size / img_h
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        raw_img = img.copy()

        # augmentation
        if self.train:
            img, lmk = self.random_flip(img, lmk)
            img = self.random_zeros(img, self.img_size)
            img = Image.fromarray(img)
            img = self.train_transform(img)
            img = np.array(img)

        # weighted mask
        mask = self.weight_mask.get_weight_Mask(raw_img, lmk)
        img = img.astype(np.float32)
        w_m_img = img * mask

        # norm
        img = self.img_morm(img)
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img, dtype=np.float32)

        w_m_img = self.img_morm(w_m_img)
        w_m_img = w_m_img.transpose((2, 0, 1))
        w_m_img = np.ascontiguousarray(w_m_img, dtype=np.float32)

        mask = mask[..., 0]

        return torch.from_numpy(img), torch.from_numpy(w_m_img), torch.from_numpy(mask), label

    def img_morm(self, image):
        image = image.astype(np.float32)
        image /= 255.0
        if self.rgb_means is not None:
            image -= self.rgb_means
        if self.std is not None:
            image /= self.std
        return image

    def random_zeros(self, image, image_size):
        rect_width = 15  # 20
        rect_height = 25  # 30
        random_seed = np.random.randint(0, 5)
        rarray = np.random.randint(low=0, high=image_size - max(rect_height, rect_width), size=(random_seed, 2))
        if len(rarray) > 0:
            for i in range(len(rarray)):
                image[rarray[i][0]: rarray[i][0] + rect_width, rarray[i][1]: rarray[i][1] + rect_height] = 0
        return image

    def random_flip(self, img, lmk):
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            lmk[:, 0] = self.img_size - lmk[:, 0]
        return img, lmk

    def load_anno(self, csv_path):
        with open(csv_path, 'r') as fr:
            reader = csv.reader(fr)
            fieldnames = next(reader)
            csv_reader = csv.DictReader(fr, fieldnames=fieldnames)
            anno_data = list(csv_reader)
        fr.close()
        data = [dict(anno) for anno in anno_data]
        return data

    def load_anno_ck(self):
        img_list = []
        ck_path = './dataset/CK+/test_set'
        label_map = {'neutral': 0, 'happy': 1, 'sadness': 2, 'surprise': 3, 'fear': 4, 'disgust': 5, 'anger': 6}
        for root, dirs, files in os.walk(ck_path, topdown=False):
            for name in files:
                if name.endswith('.png'):
                    img_list.append(os.path.join(root, name))
        anno_data = []
        for img_path in img_list:
            one_anno = {}
            one_anno['path'] = img_path
            one_anno['expression'] = label_map[img_path.split('/')[-2]]
            one_anno['facial_landmarks'] = np.loadtxt(img_path.replace('.png', '_landmarks.txt')).tolist()
            anno_data.append(one_anno)
        return anno_data

    def __len__(self):
        return self.length
