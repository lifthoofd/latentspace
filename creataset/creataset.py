import os
import argparse
from random import randint, uniform
import cv2
import numpy as np
from sklearn.utils import shuffle


def get_string_labels(dirs):
    str_labels = []
    for dir in dirs:
        # label = dir.split('_')[1]
        label = dir
        str_labels.append(label)
    return str_labels


def rand_crop(im):
    h, w, _ = im.shape
    if h > w:
        h_offset = randint(0, (h - w) - 1)
        im = im[h_offset:h_offset + w, :]
    elif w > h:
        w_offset = randint(0, (w - h) - 1)
        im = im[:, w_offset:w_offset + h]
    return im


def rand_flip(im):
    if uniform(0, 1) < 0.5:
        im = cv2.flip(im, 1)
    return im


def creataset(indir, outdir='datasets', file_limit=None, name='dataset', img_size=(128, 72)):
    dirs = os.listdir(indir)
    dirs.sort()
    get_string_labels(dirs)

    ims = []
    labels = []
    str_labels = get_string_labels(dirs)

    for dir in dirs:
        for i, file in enumerate(os.listdir(os.path.join(indir, dir)), 0):
            if file.endswith('.jpg') or file.endswith('.png'):
                fp = os.path.join(indir, dir, file)
                print('adding ' + fp)
                im = cv2.imread(fp)
                im = rand_flip(im)
                # im = rand_crop(im)
                im = cv2.resize(im, img_size, interpolation=cv2.INTER_AREA)
                cv2.imshow('image', im)
                key = cv2.waitKey(50)  # pauses for 3 seconds before fetching next image
                if key == 27:  # if ESC is pressed, exit loop
                    cv2.destroyAllWindows()
                    break
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                label = str_labels.index(dir)
                ims.append(im)
                labels.append(label)

            if file_limit is not None and i == file_limit - 1:
                break

    ims, labels = shuffle(ims, labels, random_state=0)
    np.savez(os.path.join(outdir, '{}_{}x{}.npz'.format(name, img_size[0], img_size[1])),
             images=ims,
             labels=labels,
             str_labels=str_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creataset dataset creator')
    parser.add_argument('--indir', type=str, required=True, help='set the input directory with all images')
    parser.add_argument('--outdir', type=str, default='datasets', help='set the output directory, default is ./datasets')
    parser.add_argument('--file_limit', type=int, default=None, help='limit the amount of files per label')
    parser.add_argument('--name', type=str, default='dataset', help='output name')
    parser.add_argument('--size_x', type=int, default=128, help='set the x size of the images')
    parser.add_argument('--size_y', type=int, default=72, help='set the y size of the images')
    args = parser.parse_args()
    creataset(args.indir, args.outdir, args.file_limit, args.name, img_size=(args.size_x, args.size_y))
