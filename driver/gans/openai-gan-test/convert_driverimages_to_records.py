import math
from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
from utils import get_image, colorize

import tensorflow as tf
from glob import glob
import random


IMSIZE = 128



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(argv):
    # pattern = "/home/ian/imagenet/ILSVRC2012_img_train_t1_t2/n*/*JPEG"
    pattern = "C:\data\data\\train\\c*\\*.jpg"
    files = glob(pattern)
    random.shuffle(files)
    assert len(files) > 0

    dirs = glob("C:\data\data\\train\\c*")
    assert len(dirs) == 10, len(dirs)
    dirs = [d.split('/')[-1] for d in dirs]
    dirs = sorted(dirs)

    str_to_int = dict(zip(dirs, range(len(dirs))))

    outfile = 'imagenet_train_labeled_' + str(IMSIZE) + '.tfrecords'
    outfile_test = 'imagenet_train_labeled_forTest' + str(IMSIZE) + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(outfile)

    for i, f in enumerate(files):

        image = get_image(f, IMSIZE, is_crop=True, resize_w=IMSIZE)
        image = colorize(image)
        assert image.shape == (IMSIZE, IMSIZE, 3)
        image += 1.
        image *= (255. / 2.)
        image = image.astype('uint8')
        #print image.min(), image.max()
        # from pylearn2.utils.image import save
        # save('foo.png', (image + 1.) / 2.)
        image_raw = image.tostring()

        class_str = f.split('\\')[-2]

        label = str_to_int['C:\data\data\\train\\'+class_str]
        if i % 1 == 0:
            print (i, '\t',label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(IMSIZE),
            'width': _int64_feature(IMSIZE),
            'depth': _int64_feature(3),
            'image_raw': _bytes_feature(image_raw),
            'label': _int64_feature(label)
            }))
        writer.write(example.SerializeToString())

    writer.close()


if __name__ == "__main__":
    tf.app.run()
