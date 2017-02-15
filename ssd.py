import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F

from multibox import MultiBox


def normalize_2d(x, eps=1e-05):
    norm = F.sqrt(F.sum(F.square(x), axis=1)) + eps
    norm = F.broadcast_to(norm[:, np.newaxis], x.shape)
    return x / norm


class SSD300(chainer.Chain):

    def __init__(self, n_class, n_anchors):
        super().__init__(
            base=L.VGG16Layers(pretrained_model=None),

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

            multibox=MultiBox(n_class, n_anchors=n_anchors),
        )
        self.train = False

    def __call__(self, x, t_loc=None, t_conf=None):
        hs = list()

        layers = self.base(x, layers=['conv4_3', 'conv5_3'])
        hs.append(normalize_2d(layers['conv4_3']) * 20)
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

        h_loc, h_conf = self.multibox(hs)

        if self.train:
            loss_loc, loss_conf = self.multibox.loss(
                h_loc, h_conf, t_loc, t_conf)
            loss = loss_loc + loss_conf
            chainer.report(
                {
                    'loss': loss,
                    'loc': loss_loc,
                    'conf': loss_conf},
                self)
            return loss
        else:
            return h_loc, h_conf
