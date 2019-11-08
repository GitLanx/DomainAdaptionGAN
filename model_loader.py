from Models import FCN32
from Models import UNet
from Models import GAN

MODELS = ['FCN32', 'UNet']


def load_model(model_name, input_shape, classes):
    if model_name == 'FCN32':
        model = FCN32.FCN32(classes, input_shape).build()
    elif model_name == 'UNet':
        model = UNet.UNet(classes).build()
    elif model_name == 'GAN':
        model = GAN.Generator(classes)
    return model
