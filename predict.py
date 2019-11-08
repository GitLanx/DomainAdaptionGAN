import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# from Models.CycleGAN import Generator
from Models.CycleGAN import Generator

dir = r'/home/ecust/zww/DANet/datasets/cityscapes/leftImg8bit/train/bremen/'

data = glob.glob(dir + '*.png')
model = Generator(3, 'in').build()
model.load_weights('./in_ckpt_ori200/')

img = cv2.imread(data[0]).astype(np.float32) / 255
img = np.expand_dims(img, axis=0)
prediction = model.predict(img)

plt.figure(figsize=(30, 30))

    # target_image = tf.argmax(target_image, axis=-1)
    # prediction = tf.argmax(prediction, axis=-1)
img = img[0] * 255
prediction = prediction[0] * 255

plt.subplot(1, 2, 1)
plt.imshow(img.astype(np.uint8))
plt.subplot(1, 2, 2)
plt.imshow(prediction.astype(np.uint8))
plt.show()

# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.subplot(1, 2, 2)
# plt.imshow(prediction)
# plt.show()