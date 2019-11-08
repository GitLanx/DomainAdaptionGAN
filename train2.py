import tensorflow as tf
import os
import argparse
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import get_dataset, generate_images
from metrics import mean_iou
from loss import categorical_loss, gan_loss, cycle_loss
from Models.GAN import Generator, Discriminator

tf.enable_eager_execution()

# parameters
batch_size = 1
epochs = 200
resized_shape = [96, 96]
plots = 3
lambda_para = 10
classes = 13
palette = None
# choose model to train for

netG_A = Generator(3, 'in').build()
netG_B = Generator(3, 'in').build()
netD_A = Discriminator('in').build()
netD_B = Discriminator('in').build()

# tf.keras.utils.plot_model(netG_A, to_file='cycle.png')

# dataset directory
city_img_root = '/home/ecust/zww/DANet/datasets/cityscapes/leftImg8bit/train/'
city_label_root = '/home/ecust/lx/Cityscapes/gtFine/train/ab*/*.png'
target_img_path = '/home/ecust/lx/数据库A(200)/IR_400x300/*.png'
target_label_path = '/home/ecust/lx/数据库A(200)/label_png/*.png'

# load dataset
target_img = glob.glob(target_img_path)
target_label = glob.glob(target_label_path)
city_label = glob.glob(city_label_root)
city_img = [os.path.join(city_img_root, path.split(os.sep)[-2][1:],
                         os.path.basename(path)[:-24]+'leftImg8bit.png')
            for path in city_label]

dataset = get_dataset(
    target_img,
    target_label,
    city_img,
    city_label,
    classes,
    resized_shape=resized_shape,
    palette=palette)

learning_rate = 0.0002

loss_G_A_log = []

optimizer_G = tf.train.AdamOptimizer(learning_rate)
optimizer_D = tf.train.AdamOptimizer(learning_rate)

for epoch in range(epochs):
    print(epoch)
    if epoch > 200 and epoch % 10 == 0:
        current_learning_rate = learning_rate * (1 - epoch / epochs) ** 0.9
        optimizer_G = tf.train.AdamOptimizer(current_learning_rate)
        optimizer_D = tf.train.AdamOptimizer(current_learning_rate)
    for ((real_src, real_src_label), (real_target, real_tar_label)) in dataset:
        with tf.GradientTape() as netG_A_tape, \
             tf.GradientTape() as netG_B_tape, \
             tf.GradientTape() as netD_A_tape, \
             tf.GradientTape() as netD_B_tape:

            fake_target = netG_A(real_src, training=True)
            rec_source = netG_B(fake_target, training=True)

            fake_src = netG_B(real_target, training=True)
            rec_target = netG_A(fake_src, training=True)

            disc_target_fake = netD_A(fake_target, training=True)
            disc_source_fake = netD_B(fake_src, training=True)
            disc_target_real = netD_A(real_target, training=True)
            disc_source_real = netD_B(fake_src, training=True)

            loss_G_A = gan_loss(disc_target_fake,
                                tf.ones_like(disc_target_fake))
            loss_G_B = gan_loss(disc_source_fake,
                                tf.ones_like(disc_source_fake))

            loss_cycle_A = cycle_loss(real_src, rec_source, lambda_para)
            loss_cycle_B = cycle_loss(real_target, rec_target, lambda_para)

            loss_G_A = loss_G_A + loss_cycle_A + loss_cycle_B
            loss_G_B = loss_G_B + loss_cycle_A + loss_cycle_B

            loss_D_A_real = gan_loss(disc_target_real,
                                     tf.ones_like(disc_target_real))
            loss_D_A_fake = gan_loss(disc_target_fake,
                                     tf.zeros_like(disc_target_fake))
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

            loss_D_B_real = gan_loss(disc_source_real,
                                     tf.ones_like(disc_source_real))
            loss_D_B_fake = gan_loss(disc_source_fake,
                                     tf.zeros_like(disc_source_fake))
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

        gradients_of_G_A = netG_A_tape.gradient(loss_G_A, netG_A.variables)
        gradients_of_G_B = netG_B_tape.gradient(loss_G_B, netG_B.variables)
        gradients_of_D_A = netD_A_tape.gradient(loss_D_A, netD_A.variables)
        gradients_of_D_B = netD_B_tape.gradient(loss_D_B, netD_B.variables)

        optimizer_G.apply_gradients(zip(gradients_of_G_A, netG_A.variables))
        optimizer_G.apply_gradients(zip(gradients_of_G_B, netG_B.variables))
        optimizer_D.apply_gradients(zip(gradients_of_D_A, netD_A.variables))
        optimizer_D.apply_gradients(zip(gradients_of_D_B, netD_B.variables))

    loss_G_A_log.append(loss_G_A.numpy())

plt.plot(loss_G_A_log)

netG_A.save_weights("./in_ckpt_ori200/")

for ((src_image, _), (tar_image, _)) in dataset.take(1):
    generate_images(netG_A, src_image, tar_image, plots=plots)
