import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt


def one_hot_encode(label, classes, palette):
    """change labels to one-hot encoding.

    :param label: labels
    :param palette: use self-defined label if specified
    :returns: one-hot encoded labels
    """
    if palette is None:
        label = tf.squeeze(label, axis=-1)
        one_hot_map = tf.one_hot(label, classes)
    else:
        one_hot_map = []
        for colour in palette:
            class_map = tf.reduce_all(tf.equal(label, colour), axis=-1)
            one_hot_map.append(class_map)

        one_hot_map = tf.stack(one_hot_map, axis=-1)
        one_hot_map = tf.cast(one_hot_map, tf.float32)

    return one_hot_map


def load_image(file_name, resized_shape):
    """load images.

    :param file_name: image file names
    :param resized_shape: resized_shape the images to proper size
    :returns: images
    """
    image = tf.read_file(file_name)
    image = tf.image.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, size=resized_shape)

    return image


def load_label(file_name, classes, resized_shape, palette):
    """load labels.

    :param file_name: label file names
    :param palette: label pixel
    :param resized_shape: resized_shapeze the labels to proper size
    :returns: one-hot encoded labels
    """
    label = tf.read_file(file_name)
    label = tf.image.decode_png(label)
    label = tf.image.resize_images(
        label,
        size=resized_shape,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = one_hot_encode(label, classes, palette)

    return label

def parse_function(images, labels, n_classes, resized_shape, palette):
    """
    function for parse images and labels
    """
    images = load_image(images, resized_shape)
    labels = load_label(labels, n_classes, resized_shape, palette)
    return images, labels

def get_dataset(target_img,
                target_label,
                city_img,
                city_label,
                classes,
                palette=None,
                resized_shape=[96, 96]):
    dataset = get_sub_dataset(
        target_img,
        target_label,
        classes,
        resized_shape=resized_shape,
        palette=palette)
    city_dataset = get_sub_dataset(
        city_img,
        city_label,
        classes,
        resized_shape=resized_shape,
        palette=palette)

    dataset = tf.data.Dataset.zip((city_dataset, dataset))
    return dataset.batch(4)


def get_sub_dataset(images,
                    labels,
                    n_classes,
                    resized_shape=[96, 96],
                    palette=None):
    """Use tf.data.Dataset to read image files.

    :param images: list of image file names
    :param labels: list of label file names
    :param palette: label pixel for each class
    :param resized_shape: rescale images to proper shape
    :returns: return a tf.data.Dataset
    """
    shuffle_size = len(images)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(shuffle_size, seed=1)
    dataset = dataset.map(
        lambda x, y: parse_function(x, y, n_classes, resized_shape, palette),
        num_parallel_calls=4)

    return dataset


def get_city_dataset(root, classes, split='train', resized_shape=[96, 96]):
    files = {}
    images_base = os.path.join(root, "leftImg8bit", split)
    annotations_base = os.path.join(root, 'gtFine', split)
    files[split] = [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(images_base)
        for filename in filenames if filename.endswith(".png")
    ]

    img_path = files[split]
    label_path = [
        os.path.join(annotations_base,
                     x.split(os.sep)[-2],
                     os.path.basename(x)[:-15] + "gtFine_labelTrainIds.png")
        for x in files[split]
    ]

    city_data = get_sub_dataset(img_path, label_path, classes)
    return city_data


def generate_images(model, input_image, target_image, plots=1):
    """plot input_image, target_image and prediction in one row, all with
    shape [batch_size, height, width, channels].

    :param model: trained model
    :param input_image: a batch of input images
    :param target_image: a batch of target images
    :param plots: numbers of image groups you want to plot, default 1
    """
    assert plots <= input_image.shape[
        0], "plots number should be less than batch size"

    prediction = model.predict(input_image)
    plt.figure(figsize=(50, 50))

    # target_image = tf.argmax(target_image, axis=-1)
    # prediction = tf.argmax(prediction, axis=-1)

    for i in range(plots):
        plt.subplot(plots, 3, i * 3 + 1)
        plt.imshow(tf.cast(input_image[i] * 255, dtype=tf.uint8))
        plt.subplot(plots, 3, i * 3 + 2)
        plt.imshow(tf.cast(target_image[i] * 255, dtype=tf.uint8))
        plt.subplot(plots, 3, i * 3 + 3)
        plt.imshow(tf.cast(prediction[i] * 255, dtype=tf.uint8))
    plt.show()

def gray2rgb(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb
