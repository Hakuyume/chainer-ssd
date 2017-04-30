import numpy as np

import chainer
import chainer.functions as F


def _elementwise_softmax_cross_entropy(x, t):
    assert x.shape[:-1] == t.shape
    p = F.reshape(
        F.select_item(F.reshape(x, (-1, x.shape[-1])), F.flatten(t)),
        t.shape)
    return F.logsumexp(x, axis=-1) - p


def _mine_hard_negative(loss, pos, k):
    xp = chainer.cuda.get_array_module(loss)
    loss = chainer.cuda.to_cpu(loss)
    pos = chainer.cuda.to_cpu(pos)
    rank = (loss * (pos - 1)).argsort(axis=1).argsort(axis=1)
    hard_neg = rank < (pos.sum(axis=1) * k)[:, np.newaxis]
    return xp.array(hard_neg)


def multibox_loss(x_loc, x_conf, t_loc, t_conf, k):
    xp = chainer.cuda.get_array_module(t_conf.data)
    with chainer.cuda.get_device(t_conf.data):
        pos = t_conf.data > 0
        if xp.logical_not(pos).all():
            return 0, 0

    x_loc = F.reshape(x_loc, (-1, 4))
    t_loc = F.reshape(t_loc, (-1, 4))
    loss_loc = F.huber_loss(x_loc, t_loc, 1)
    loss_loc *= pos.flatten().astype(loss_loc.dtype)
    loss_loc = F.sum(loss_loc) / pos.sum()

    loss_conf = _elementwise_softmax_cross_entropy(x_conf, t_conf)
    hard_neg = _mine_hard_negative(loss_conf.data, pos, k)
    loss_conf *= xp.logical_or(pos, hard_neg).astype(loss_conf.dtype)
    loss_conf = F.sum(loss_conf) / pos.sum()

    return loss_loc, loss_conf
