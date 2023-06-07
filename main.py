import tensorflow as tf
import os
import numpy as np
import click
from skimage.io import imread
from PIL import Image
from skimage.morphology import binary_opening, disk


fullres_model = tf.keras.models.load_model('model/fullres_model.h5')


def raw_prediction(path):
    c_img = imread(path)
    pre_img = np.expand_dims(c_img, 0)/255.0
    cur_seg = fullres_model.predict(pre_img)[0]
    return cur_seg, c_img


def smooth(cur_seg):
    return binary_opening(cur_seg>0.99, np.expand_dims(disk(2), -1))


def predict(path='Data/08cd21e12.jpg'):
    img_n = path.split('/')[-1]
    cur_seg, c_img = raw_prediction(path=path)
    return smooth(cur_seg), c_img, img_n


def pred_img(i_p, save_f):
    seg, img, name = predict(i_p)
    img = Image.fromarray(img)
    seg = Image.fromarray(seg[:, :, 0])
    if os.path.isdir(f"{save_f}") is False:
        os.mkdir(f"{save_f}")
    img.save(f"{save_f}/img_{name}")
    seg.save(f"{save_f}/seg_{name}")


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
@click.argument('path_t', default='Data/08cd21e12.jpg')
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


