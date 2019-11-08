import glob
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# from Models.CycleGAN import Generator
from Models.CycleGAN import Generator

city_img_root = '/home/ecust/zww/DANet/datasets/cityscapes/leftImg8bit/train/'
city_label_root = '/home/ecust/lx/Cityscapes/gtFine/train/*/*.png'

city_label = glob.glob(city_label_root)
city_img = [os.path.join(city_img_root, path.split(os.sep)[-2][1:],
                         os.path.basename(path)[:-24]+'leftImg8bit.png')
            for path in city_label]

model = Generator(3, 'bn').build()
model.load_weights('./bn_ckpt_ori200/')

for i in range(len(city_img)):
    ori_img = cv2.imread(city_img[i]).astype(np.float32)
    img = ori_img / 255
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)

    prediction = prediction[0] * 255

    if not os.path.exists(
            os.path.join('/home/ecust/lx/Cityscapes', 'new_imgs', 'train', city_img[i].split(
                os.sep)[-2])):
        os.makedirs(
            os.path.join('/home/ecust/lx/Cityscapes', 'new_imgs', 'train',
                         city_img[i].split(os.sep)[-2]))

    if not os.path.exists(
            os.path.join('/home/ecust/lx/Cityscapes', 'imgs', 'train', city_img[i].split(
                os.sep)[-2])):
        os.makedirs(
            os.path.join('/home/ecust/lx/Cityscapes', 'imgs', 'train',
                         city_img[i].split(os.sep)[-2]))

    cv2.imwrite(
        os.path.join(
            '/home/ecust/lx/Cityscapes', 'new_imgs', 'train', city_img[i].split(os.sep)[-2],
            os.path.basename(city_img[i])), prediction)

    cv2.imwrite(
        os.path.join(
            '/home/ecust/lx/Cityscapes', 'imgs', 'train', city_img[i].split(os.sep)[-2],
            os.path.basename(city_img[i])), ori_img)
