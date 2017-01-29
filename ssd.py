#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F


class MultiBoxHead(chainer.Chain):

    def __init__(self, n_class):
        super(MultiBoxHead, self).__init__(
            loc=chainer.ChainList(),
            conf=chainer.ChainList(),
        )

        self.n_class = n_class

        aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
        for aspect_ratio in aspect_ratios:
            n_anchor = (len(aspect_ratio) + 1) * 2
            self.loc.add_link(L.Convolution2D(
                None, n_anchor * 4, 3, stride=1, pad=1)
            )
            self.conf.add_link(L.Convolution2D(
                None, n_anchor * (self.n_class + 1), 3, stride=1, pad=1)
            )

    def __call__(self, xs):
        hs_loc = list()
        hs_conf = list()

        for i, x in enumerate(xs):
            h_loc = self.loc[i](x)
            h_loc = F.transpose(h_loc, (0, 2, 3, 1))
            h_loc = F.reshape(h_loc, (h_loc.shape[0], -1, 4))
            hs_loc.append(h_loc)

            h_conf = self.conf[i](x)
            h_conf = F.transpose(h_conf, (0, 2, 3, 1))
            h_conf = F.reshape(h_conf, (h_conf.shape[0], -1, self.n_class + 1))
            hs_conf.append(h_conf)

        hs_loc = F.concat(hs_loc, axis=1)
        hs_conf = F.concat(hs_conf, axis=1)
        return hs_loc, hs_conf


class SSD300(chainer.Chain):

    def __init__(self, n_class):
        super(SSD300, self).__init__(
            base=L.VGG16Layers(),

            conv6=L.DilatedConvolution2D(
                None, 1024, 3, stride=1, pad=6, dilate=6),
            conv7=L.Convolution2D(None, 1024, 1, stride=1),

            conv8_1=L.Convolution2D(None, 256, 1, stride=1),
            conv8_2=L.Convolution2D(None, 512, 3, stride=2, pad=1),

            conv9_1=L.Convolution2D(None, 128, 1, stride=1),
            conv9_2=L.Convolution2D(None, 256, 3, stride=2, pad=1),

            conv10_1=L.Convolution2D(None, 128, 1, stride=1),
            conv10_2=L.Convolution2D(None, 256, 3, stride=1),

            conv11_1=L.Convolution2D(None, 128, 1, stride=1),
            conv11_2=L.Convolution2D(None, 256, 3, stride=1),

            multibox=MultiBoxHead(n_class),
        )
        self.train = False

    def __call__(self, x):
        hs = list()

        layers = self.base(x, layers=['conv4_3', 'conv5_3'])
        hs.append(layers['conv4_3'])
        h = layers['conv5_3']
        h = F.max_pooling_2d(h, 3, stride=1, pad=1)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        hs.append(h)

        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        hs.append(h)

        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        hs.append(h)

        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        hs.append(h)

        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))
        hs.append(h)

        return self.multibox(hs)
