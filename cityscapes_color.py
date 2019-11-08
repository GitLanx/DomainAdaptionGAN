import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2


dir = '/home/ecust/lx/Cityscapes/gtFine/'
newdir = '/home/ecust/lx/Cityscapes/color_gtFine/'
files = [
    os.path.join(looproot, filename)
    for looproot, _, filenames in os.walk(dir)
    for filename in filenames
]

cityspallete = [
    0, 0, 0,
    0, 0, 142,
    119, 11, 32,
    220, 20, 60,
    70, 130, 180,
    107, 142, 35,
    152, 251, 152,
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    190, 153, 153,
    220, 220, 0,
    153, 153, 153,
    # 250, 170, 30     # block
]

for i in range(len(files)):
    if not os.path.exists(
            os.path.join(newdir, 'train', files[i].split(
                os.sep)[-2])):
        os.makedirs(
            os.path.join(newdir, 'train',
                         files[i].split(os.sep)[-2]))
    img = Image.open(files[i])
    img = np.array(img)
# img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# b = img == 6

    b = Image.fromarray(img.astype('uint8'))
    b.putpalette(cityspallete)
    b.save(os.path.join(
            newdir, 'train', files[i].split(os.sep)[-2],
            os.path.basename(files[i])[:-4] + '.png'))
    # b.save(path[i][:-17] + 'color.png')
# plt.imshow(b)
# plt.show()
