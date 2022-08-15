import tensorflow as tf
import os
import numpy as np
import click
from skimage import img_as_ubyte
from skimage.io import imread, imsave
from skimage.morphology import binary_opening, disk
from tools import gdice_coef_loss, dice_coef, true_positive_rate, zero_IoU, IoU, agg_loss


fullres_model = tf.keras.models.load_model('model', custom_objects={
            "gdice_coef_loss": gdice_coef_loss,
            "dice_coef": dice_coef,
            "true_positive_rate": true_positive_rate
        })


def predict(img_p):
    c_img = imread(img_p)
    c_img = np.expand_dims(c_img, 0)/255.0
    cur_seg = fullres_model.predict(c_img)[0]
    cur_seg = binary_opening(cur_seg>1e3, np.expand_dims(disk(2), -1))
    img_n = img_p.split('/')[-1]
    return cur_seg, c_img, img_n


def pred_img(i_p, save_f):
    c_s, c_i, c_n = predict(i_p)
    if os.path.isdir(f"{save_f}") is False:
        os.mkdir(f"{save_f}")
    imsave(f"{save_f}/c_s_{c_n}", img_as_ubyte(c_s))
    imsave(f"{save_f}/c_i_{c_n}", c_i[0])


def main_f(f_path, stream, save_f):
    while True:
        imgs = os.listdir(f_path)
        for img_n in imgs:
            i_p = os.path.join(f_path, img_n).replace("\\", "/")
            pred_img(i_p, save_f)
        if stream == False:
            break


@click.command()
@click.argument('type', default='image')
@click.argument('path_t', default='Data/733337296.jpg')
@click.argument('save_f', default='Done')
def main(type, path_t, save_f):
    if type == "stream":
        main_f(path_t, True, save_f)
    elif type == "folder":
        main_f(path_t, False, save_f)
    elif type == 'image':
        pred_img(path_t, save_f)


if __name__ == '__main__':
   main()


