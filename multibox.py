# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F


class MultiBox(chainer.Chain):

    def __init__(self, n_class, aspect_ratios):
        super(MultiBox, self).__init__(
            loc=chainer.ChainList(),
            conf=chainer.ChainList(),
        )

        self.n_class = n_class
        self.aspect_ratios = aspect_ratios

        for aspect_ratio in aspect_ratios:
            n_anchor = (len(aspect_ratio) + 1) * 2
            self.loc.add_link(L.Convolution2D(
                None, n_anchor * 4, 3, stride=1, pad=1))
            self.conf.add_link(L.Convolution2D(
                None, n_anchor * (self.n_class + 1), 3, stride=1, pad=1))

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
