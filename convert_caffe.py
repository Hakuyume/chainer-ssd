#!/usr/bin/env python3

import argparse
import numpy as np

import chainer
from chainer import serializers
from chainer.links.caffe import CaffeFunction

import config
from ssd import SSD300


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('target')
    parser.add_argument('--baseonly', action='store_true')
    parser.set_defaults(baseonly=False)
    args = parser.parse_args()

    caffe_model = CaffeFunction(args.source)
    model = SSD300(n_class=20, aspect_ratios=config.aspect_ratios)

    model.base.conv1_1.copyparams(caffe_model.conv1_1)
    model.base.conv1_2.copyparams(caffe_model.conv1_2)

    model.base.conv2_1.copyparams(caffe_model.conv2_1)
    model.base.conv2_2.copyparams(caffe_model.conv2_2)

    model.base.conv3_1.copyparams(caffe_model.conv3_1)
    model.base.conv3_2.copyparams(caffe_model.conv3_2)
    model.base.conv3_3.copyparams(caffe_model.conv3_3)

    model.base.conv4_1.copyparams(caffe_model.conv4_1)
    model.base.conv4_2.copyparams(caffe_model.conv4_2)
    model.base.conv4_3.copyparams(caffe_model.conv4_3)

    model.conv5_1.copyparams(caffe_model.conv5_1)
    model.conv5_2.copyparams(caffe_model.conv5_2)
    model.conv5_3.copyparams(caffe_model.conv5_3)

    model.conv6.copyparams(caffe_model.fc6)
    model.conv7.copyparams(caffe_model.fc7)

    if args.baseonly:
        model(chainer.Variable(
            np.empty((1, 3, model.insize, model.insize), dtype=np.float32),
            volatile=True))
    else:
        model.conv8_1.copyparams(caffe_model.conv6_1)
        model.conv8_2.copyparams(caffe_model.conv6_2)

        model.conv9_1.copyparams(caffe_model.conv7_1)
        model.conv9_2.copyparams(caffe_model.conv7_2)

        model.conv10_1.copyparams(caffe_model.conv8_1)
        model.conv10_2.copyparams(caffe_model.conv8_2)

        model.conv11_1.copyparams(caffe_model.conv9_1)
        model.conv11_2.copyparams(caffe_model.conv9_2)

        model.multibox.loc[0].copyparams(caffe_model.conv4_3_norm_mbox_loc)
        model.multibox.conf[0].copyparams(caffe_model.conv4_3_norm_mbox_conf)

        model.multibox.loc[1].copyparams(caffe_model.fc7_mbox_loc)
        model.multibox.conf[1].copyparams(caffe_model.fc7_mbox_conf)

        model.multibox.loc[2].copyparams(caffe_model.conv6_2_mbox_loc)
        model.multibox.conf[2].copyparams(caffe_model.conv6_2_mbox_conf)

        model.multibox.loc[3].copyparams(caffe_model.conv7_2_mbox_loc)
        model.multibox.conf[3].copyparams(caffe_model.conv7_2_mbox_conf)

        model.multibox.loc[4].copyparams(caffe_model.conv8_2_mbox_loc)
        model.multibox.conf[4].copyparams(caffe_model.conv8_2_mbox_conf)

        model.multibox.loc[5].copyparams(caffe_model.conv9_2_mbox_loc)
        model.multibox.conf[5].copyparams(caffe_model.conv9_2_mbox_conf)

    serializers.save_npz(args.target, model)
