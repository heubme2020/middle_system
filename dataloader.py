import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random

class SelfDataSet(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, '*.png'))


    def __getitem__(self, index):
        #读取图片和标签
        label_path = self.imgs_path[index]
        image_path = label_path.replace('out.png', 'in.png')
        image_path = image_path.replace('output', 'input')
        image = cv2.imread(image_path) #RGB 3通道图片
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(3, 256, 256)
        label = label.reshape(1, 256, 256)
        return image, label

    def __len__(self):
        return len(self.imgs_path)


if __name__ == '__main__':
    data_path = "train/output/"
    plate_dataset = SelfDataSet(data_path)
    print(len(plate_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=plate_dataset, batch_size=5, shuffle=True)
    for image,label in train_loader:
        print(label.shape)