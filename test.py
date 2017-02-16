#! /usr/bin/env python3

import argparse
import cv2
import numpy as np

import chainer
from chainer import cuda
from chainer import serializers

import config
from ssd import SSD300
from multibox import MultiBoxEncoder
import voc
from voc import VOCDataset


def dump_result(name, size, loc, conf):
    print(name)

    xp = cuda.get_array_module(loc)
    if xp is not np:
        loc = xp.asnumpy(loc)
        conf = xp.asnumpy(conf)

    boxes, conf = multibox_encoder.decode(loc, conf)
    nms = multibox_encoder.non_maximum_suppression(boxes, conf, 0.45)
    for box, cls, score in nms:
        box *= size

        filename = 'comp4_det_test_{:s}.txt'.format(voc.names[cls])
        with open(filename, mode='a') as f:
            print(
                name, score, box.left, box.top, box.right, box.bottom, file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='VOCdevkit')
    parser.add_argument('--test', action='append')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('model')
    args = parser.parse_args()

    model = SSD300(n_class=20, aspect_ratios=config.aspect_ratios)
    serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    multibox_encoder = MultiBoxEncoder(
        grids=model.grids,
        aspect_ratios=model.aspect_ratios,
        variance=config.variance)

    dataset = VOCDataset(args.root, [t.split('-') for t in args.train])

    info = list()
    batch = list()
    for i in range(len(dataset)):
        src = dataset.image(i)
        info.append((dataset.name(i), src.shape[1::-1]))
        x = cv2.resize(src, (model.insize, model.insize)).astype(np.float32)
        x -= config.mean
        x = x.transpose(2, 0, 1)
        batch.append(x)

        if len(batch) == args.batchsize:
            loc, conf = model(chainer.Variable(np.array(batch), volatile=True))
            for i, (name, size) in enumerate(info):
                dump_result(name, size, loc.data[i], conf.data[i])
            info = list()
            batch = list()

    if len(batch) > 0:
        loc, conf = model(chainer.Variable(np.array(batch), volatile=True))
        for i, (name, size) in enumerate(info):
            dump_result(name, size, loc.data[i], conf.data[i])
