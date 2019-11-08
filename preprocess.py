import os
import cv2

root = r'/home/ecust/zww/DANet/datasets/cityscapes/'
newroot = '/home/ecust/lx/Cityscapes/'
n_classes = 12
split = "train"
files = {}
images_base = os.path.join(root, "leftImg8bit", split)
annotations_base = os.path.join(root, "gtFine", split)
files[split] = [
    os.path.join(looproot, filename)
    for looproot, _, filenames in os.walk(images_base)
    for filename in filenames if filename.endswith(".png")
]
void_classes = [
    0, 1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 16, 18, 19, 27, 28, 29, 30, 31, -1
]
valid_classes = [7, 8, 11, 13, 17, 20, 21, 22, 23, 24, 25, 26, 32, 33]
class_names = [
    'unlabelled', 'car', 'ride tool', 'person', 'sky', 'tree', 'grass', 'road',
    'sidewald', 'building', 'fence', 'traffic_sign', 'pole'
]
ignore_index = 0
class_map = {
    7: 7,
    8: 8,
    11: 9,
    13: 10,
    17: 12,
    20: 11,
    21: 5,
    22: 6,
    23: 4,
    24: 3,
    25: 3,
    26: 1,
    32: 2,
    33: 2
}
label_path = [
    os.path.join(annotations_base,
                 x.split(os.sep)[-2],
                 os.path.basename(x)[:-15] + "gtFine_labelIds.png")
    for x in files[split]
]

for i in range(len(label_path)):
    if not os.path.exists(
            os.path.join(newroot, 'gtFine', split, label_path[i].split(
                os.sep)[-2])):
        os.makedirs(
            os.path.join(newroot, 'gtFine', split,
                         label_path[i].split(os.sep)[-2]))
    label = cv2.imread(label_path[i], cv2.IMREAD_GRAYSCALE)
    for _voidc in void_classes:
        label[label == _voidc] = ignore_index
    for _validc in valid_classes:
        label[label == _validc] = class_map[_validc]

    cv2.imwrite(
        os.path.join(
            newroot, 'gtFine', split, label_path[i].split(os.sep)[-2],
            os.path.basename(label_path[i])[:-19] +
            "gtFine_labelTrainIds.png"), label)
