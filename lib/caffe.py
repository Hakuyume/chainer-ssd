import numpy as np

import chainer.links.caffe.caffe_function as caffe

from lib.ssd import _Normalize


class _CaffeFunction(caffe.CaffeFunction):

    def __init__(self, model_path):
        super().__init__(model_path)

    @caffe._layer('Normalize', None)
    def _setup_normarize(self, layer):
        blobs = layer.blobs
        func = _Normalize(caffe._get_num(blobs[0]))
        func.scale.data[:] = np.array(blobs[0].data)
        self.add_link(layer.name, func)

    @caffe._layer('AnnotatedData', None)
    @caffe._layer('Flatten', None)
    @caffe._layer('MultiBoxLoss', None)
    @caffe._layer('Permute', None)
    @caffe._layer('PriorBox', None)
    def _skip_layer(self, _):
        pass


def load_caffe(model_path, model, base_only=False):
    caffe_model = _CaffeFunction(model_path)

    model.conv1_1.copyparams(caffe_model.conv1_1)
    model.conv1_2.copyparams(caffe_model.conv1_2)

    model.conv2_1.copyparams(caffe_model.conv2_1)
    model.conv2_2.copyparams(caffe_model.conv2_2)

    model.conv3_1.copyparams(caffe_model.conv3_1)
    model.conv3_2.copyparams(caffe_model.conv3_2)
    model.conv3_3.copyparams(caffe_model.conv3_3)

    model.conv4_1.copyparams(caffe_model.conv4_1)
    model.conv4_2.copyparams(caffe_model.conv4_2)
    model.conv4_3.copyparams(caffe_model.conv4_3)

    if not base_only:
        model.norm4.copyparams(caffe_model.conv4_3_norm)

    model.conv5_1.copyparams(caffe_model.conv5_1)
    model.conv5_2.copyparams(caffe_model.conv5_2)
    model.conv5_3.copyparams(caffe_model.conv5_3)

    model.conv6.copyparams(caffe_model.fc6)
    model.conv7.copyparams(caffe_model.fc7)

    if not base_only:
        model.conv8_1.copyparams(caffe_model.conv6_1)
        model.conv8_2.copyparams(caffe_model.conv6_2)

        model.conv9_1.copyparams(caffe_model.conv7_1)
        model.conv9_2.copyparams(caffe_model.conv7_2)

        model.conv10_1.copyparams(caffe_model.conv8_1)
        model.conv10_2.copyparams(caffe_model.conv8_2)

        model.conv11_1.copyparams(caffe_model.conv9_1)
        model.conv11_2.copyparams(caffe_model.conv9_2)

        model.loc[0].copyparams(caffe_model.conv4_3_norm_mbox_loc)
        model.conf[0].copyparams(caffe_model.conv4_3_norm_mbox_conf)

        model.loc[1].copyparams(caffe_model.fc7_mbox_loc)
        model.conf[1].copyparams(caffe_model.fc7_mbox_conf)

        model.loc[2].copyparams(caffe_model.conv6_2_mbox_loc)
        model.conf[2].copyparams(caffe_model.conv6_2_mbox_conf)

        model.loc[3].copyparams(caffe_model.conv7_2_mbox_loc)
        model.conf[3].copyparams(caffe_model.conv7_2_mbox_conf)

        model.loc[4].copyparams(caffe_model.conv8_2_mbox_loc)
        model.conf[4].copyparams(caffe_model.conv8_2_mbox_conf)

        model.loc[5].copyparams(caffe_model.conv9_2_mbox_loc)
        model.conf[5].copyparams(caffe_model.conv9_2_mbox_conf)
