#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F


class SSD300(chainer.Chain):

    def __init__(self):
        super(SSD300, self).__init__(
            base=L.VGG16Layers(),

            conv6=L.Convolution2D(None, 1024, 3, stride=1, pad=1),
            conv7=L.Convolution2D(None, 1024, 1, stride=1),

            conv8_1=L.Convolution2D(None, 256, 1, stride=1),
            conv8_2=L.Convolution2D(None, 512, 3, stride=2, pad=1),

            conv9_1=L.Convolution2D(None, 128, 1, stride=1),
            conv9_2=L.Convolution2D(None, 256, 3, stride=2, pad=1),

            conv10_1=L.Convolution2D(None, 128, 1, stride=1),
            conv10_2=L.Convolution2D(None, 256, 3, stride=1),

            conv11_1=L.Convolution2D(None, 128, 1, stride=1),
            conv11_2=L.Convolution2D(None, 256, 3, stride=1),
        )
        self.train = False

    def __call__(self, x):
        layers = ['conv4_2', 'conv5_2']
        h = self.base(x, layers=layers)
        h_conv4 = h['conv4_2']
        h_conv5 = h['conv5_2']

        h = F.relu(self.conv6(h_conv5))
        h_conv7 = F.relu(self.conv7(h))

        h = F.relu(self.conv8_1(h))
        h_conv8 = F.relu(self.conv8_2(h))

        h = F.relu(self.conv9_1(h_conv8))
        h_conv9 = F.relu(self.conv9_2(h))

        h = F.relu(self.conv10_1(h_conv9))
        h_conv10 = F.relu(self.conv10_2(h))

        h = F.relu(self.conv11_1(h_conv10))
        h_conv11 = F.relu(self.conv11_2(h))

        return h_conv4, h_conv7, h_conv8, h_conv9, h_conv10, h_conv11
