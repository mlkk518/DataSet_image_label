import os
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.utils import data


class VOCDataSet(data.Dataset):
    def __init__(self, root, list_path, crop_size=(321, 321), mean=(104.008, 116.669, 122.675), mirror=True, scale=True,
                 ignore_label=255):
        super(VOCDataSet, self).__init__()
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.ignore_label = ignore_label
        self.mean = np.asarray(mean, np.float32)
        self.is_mirror = mirror
        self.is_scale = scale

        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        self.files = []
        for name in self.img_ids:
            img_file = os.path.join(self.root, "JPEGImages/%s.jpeg" % name)
            label_file = os.path.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        '''load the datas'''
        name = datafiles["name"]
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"]).convert('L')
        size_origin = image.size  # W * H

        '''random scale the images and labels'''
        if self.is_scale:  # 如果我在定义dataset时选择了scale=True，就执行本语句对尺度进行随机变换
            ratio = 0.5 + random.randint(0, 11) // 10.0  # 0.5~1.5
            out_h, out_w = int(size_origin[1] * ratio), int(size_origin[0] * ratio)
            # (H,W)for Resize
            image = transforms.Resize((out_h, out_w), Image.LANCZOS)(image)
            label = transforms.Resize((out_h, out_w), Image.NEAREST)(label)

        '''pad the inputs if their size is smaller than the crop_size'''
        pad_w = max(self.crop_w - out_w, 0)
        pad_h = max(self.crop_h - out_h, 0)
        img_pad = transforms.Pad(padding=(0, 0, pad_w, pad_h), fill=0, padding_mode='constant')(image)
        label_pad = transforms.Pad(padding=(0, 0, pad_w, pad_h), fill=self.ignore_label, padding_mode='constant')(label)
        out_size = img_pad.size

        '''random crop the inputs'''
        if (self.crop_h != 0 or self.crop_w != 0):
            # select a random start-point for croping operation
            h_off = random.randint(0, out_size[1] - self.crop_h)
            w_off = random.randint(0, out_size[0] - self.crop_w)
            # crop the image and the label
            image = img_pad.crop((w_off, h_off, w_off + self.crop_w, h_off + self.crop_h))
            label = label_pad.crop((w_off, h_off, w_off + self.crop_w, h_off + self.crop_h))

        '''mirror operation'''
        if self.is_mirror:
            if np.random.random() < 0.5:
                # 0:FLIP_LEFT_RIGHT, 1:FLIP_TOP_BOTTOM, 2:ROTATE_90, 3:ROTATE_180, 4:or ROTATE_270.
                image = image.transpose(0)
                label = label.transpose(0)

        '''convert PIL Image to numpy array'''
        I = np.asarray(image, np.float32) - self.mean
        I = I.transpose((2, 0, 1))  # transpose the  H*W*C to C*H*W
        L = np.asarray(np.array(label), np.int64)
        # print(I.shape,L.shape)
        return I.copy(), L.copy(), np.array(size_origin), name


# 这是一个测试函数,也即我的代码写好后,如果直接python运行当前py文件,就会执行以下代码的内容,以检测我上面的代码是否有问题,这其实就是方便我们调试,而不是每次都去run整个网络再看哪里报错
if __name__ == '__main__':
    DATA_DIRECTORY = './data/train_data/'
    DATA_LIST_PATH = './data/list/val.txt'
    Batch_size = 4
    MEAN = (104.008, 116.669, 122.675)
    dst = VOCDataSet(DATA_DIRECTORY, DATA_LIST_PATH, mean=(0, 0, 0))
    # just for test,  so the mean is (0,0,0) to show the original images.
    # But when we are training a model, the mean should have another value
    trainloader = data.DataLoader(dst, batch_size=Batch_size)
    plt.ion()
    for i, data in enumerate(trainloader):
        imgs, labels, _, _ = data
        if i % 1 == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = img.astype(
                np.uint8)  # change the dtype from float32 to uint8, because the plt.imshow() need the uint8
            img = np.transpose(img, (1, 2, 0))  # transpose the Channels*H*W to  H*W*Channels
            # img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            plt.pause(12)

            # input()
