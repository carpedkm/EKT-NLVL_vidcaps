from __future__ import absolute_import

import torch
from torch import nn
import sys
sys.path.append('/saat/3D-ResNets-PyTorch')
from models import resnet, pre_act_resnet, wide_resnet, resnext, densenet


def generate_model(opt):

    last_fc = False

    assert opt.model in ['resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet']

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            model = resnet.resnet10(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration)
        elif opt.model_depth == 18:
            model = resnet.resnet18(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = resnet.resnet34(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = resnet.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnet.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnet.resnet152(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = resnet.resnet200(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration)

    elif opt.model == 'wideresnet':
        assert opt.model_depth in [50]

        if opt.model_depth == 50:
            model = wide_resnet.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, k=opt.wide_resnet_k,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration)
    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]

        if opt.model_depth == 50:
            model = resnext.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnext.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                      sample_size=opt.sample_size, sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnext.resnet152(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                      sample_size=opt.sample_size, sample_duration=opt.sample_duration)
    elif opt.model == 'preresnet':
        assert opt.model_depth in [18, 34, 50, 101, 152, 200]

        if opt.model_depth == 18:
            model = pre_act_resnet.resnet18(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                            sample_size=opt.sample_size, sample_duration=opt.sample_duration
                                             )
        elif opt.model_depth == 34:
            model = pre_act_resnet.resnet34(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                            sample_size=opt.sample_size, sample_duration=opt.sample_duration
                                             )
        elif opt.model_depth == 50:
            model = pre_act_resnet.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                            sample_size=opt.sample_size, sample_duration=opt.sample_duration
                                             )
        elif opt.model_depth == 101:
            model = pre_act_resnet.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                             sample_size=opt.sample_size, sample_duration=opt.sample_duration
                                              )
        elif opt.model_depth == 152:
            model = pre_act_resnet.resnet152(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                             sample_size=opt.sample_size, sample_duration=opt.sample_duration
                                              )
        elif opt.model_depth == 200:
            model = pre_act_resnet.resnet200(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                             sample_size=opt.sample_size, sample_duration=opt.sample_duration
                                              )
    elif opt.model == 'densenet':
        assert opt.model_depth in [121, 169, 201, 264]

        if opt.model_depth == 121:
            model = densenet.densenet121(num_classes=opt.n_classes,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration
                                          )
        elif opt.model_depth == 169:
            model = densenet.densenet169(num_classes=opt.n_classes,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration
                                          )
        elif opt.model_depth == 201:
            model = densenet.densenet201(num_classes=opt.n_classes,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration
                                          )
        elif opt.model_depth == 264:
            model = densenet.densenet264(num_classes=opt.n_classes,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration
                                          )
    return model