import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import initializers

from multibox import MultiBox


class Normalize(chainer.Link):

    def __init__(self, n_channel, initial=0, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.add_param(
            'scale',
            n_channel,
            initializer=initializers._get_initializer(initial))

    def __call__(self, x):
        norm = F.sqrt(F.sum(F.square(x), axis=1) + self.eps)
        norm = F.broadcast_to(norm[:, np.newaxis], x.shape)
        scale = F.broadcast_to(self.scale[:, np.newaxis, np.newaxis], x.shape)

        return x * scale / norm


class SSD300(chainer.Chain):

    insize = 300
    grids = (38, 19, 10, 5, 3, 1)

    def __init__(self, n_classes, aspect_ratios):
        init = {
            'initialW': initializers.GlorotUniform(),
            'initial_bias': initializers.Zero(),
        }
        super().__init__(
            conv1_1=L.Convolution2D(None, 64, 3, pad=1, **init),
            conv1_2=L.Convolution2D(None, 64, 3, pad=1, **init),

            conv2_1=L.Convolution2D(None, 128, 3, pad=1, **init),
            conv2_2=L.Convolution2D(None, 128, 3, pad=1, **init),

            conv3_1=L.Convolution2D(None, 256, 3, pad=1, **init),
            conv3_2=L.Convolution2D(None, 256, 3, pad=1, **init),
            conv3_3=L.Convolution2D(None, 256, 3, pad=1, **init),

            conv4_1=L.Convolution2D(None, 512, 3, pad=1, **init),
            conv4_2=L.Convolution2D(None, 512, 3, pad=1, **init),
            conv4_3=L.Convolution2D(None, 512, 3, pad=1, **init),
            norm4=Normalize(512, initial=initializers.Constant(20)),

            conv5_1=L.DilatedConvolution2D(None, 512, 3, pad=1, **init),
            conv5_2=L.DilatedConvolution2D(None, 512, 3, pad=1, **init),
            conv5_3=L.DilatedConvolution2D(None, 512, 3, pad=1, **init),

            conv6=L.DilatedConvolution2D(
                None, 1024, 3, pad=6, dilate=6, **init),
            conv7=L.Convolution2D(None, 1024, 1, **init),

            conv8_1=L.Convolution2D(None, 256, 1, **init),
            conv8_2=L.Convolution2D(None, 512, 3, stride=2, pad=1, **init),

            conv9_1=L.Convolution2D(None, 128, 1, **init),
            conv9_2=L.Convolution2D(None, 256, 3, stride=2, pad=1, **init),

            conv10_1=L.Convolution2D(None, 128, 1, **init),
            conv10_2=L.Convolution2D(None, 256, 3, **init),

            conv11_1=L.Convolution2D(None, 128, 1, **init),
            conv11_2=L.Convolution2D(None, 256, 3, **init),

            multibox=MultiBox(
                n_classes, aspect_ratios=aspect_ratios, init=init),
        )
        self.n_classes = n_classes
        self.aspect_ratios = aspect_ratios

    def __call__(self, x, t_loc=None, t_conf=None):
        hs = list()

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        hs.append(self.norm4(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
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

        h_loc, h_conf = self.multibox(hs)

        if t_loc is None or t_conf is None:
            return h_loc, h_conf

        loss_loc, loss_conf = self.multibox.loss(
            h_loc, h_conf, t_loc, t_conf)
        loss = loss_loc + loss_conf
        chainer.report(
            {'loss': loss, 'loc': loss_loc, 'conf': loss_conf},  self)
        return loss
