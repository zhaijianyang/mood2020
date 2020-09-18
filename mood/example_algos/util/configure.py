import os
import sys

from .constant import *


TRAIN_DATASET_DIR = '/media/cxr/学习工作/mood/brain'

LOAD_MODEL_NAME = ''

# keywords
BASIC_KWS = {
    'logger': 'tensorboard',
    'log_dir': os.path.join(TRAIN_DATASET_DIR, 'log1'),
    'train_data_dir': os.path.join(TRAIN_DATASET_DIR, 'preprocessed_3'),

    'load': False,
    'load_path': os.path.join(TRAIN_DATASET_DIR, 'log', LOAD_MODEL_NAME, 'checkpoint'),
    'load_file_name': '21',
    'name': LOAD_MODEL_NAME,

    'print_every_iter': 100,
    # 'save_per_epoch': 1,
    'n_epochs': 25,
}

# 只包含了公共的参数，不同recipe对应的参数要手动添加
TRAIN_KWS = {
    'lr': 1e-4,
    'dataset': 'abdom',
    'batch_size': 16,
    'target_size': 128,
    'select_dim': 0,
}

OTHER_KWS = {
    'see_slice': 5,

    'data_augment_prob': 1,
    'mask_square_size': 15,

    'res_size': 32,
    'minus_low': False,
}

CONFIGURE_DICT = {
    'zunet': {},
    'zcae': {},

    'unet': {
        'base_num_features': 16,
        'num_classes': 1,
        'num_pool': 5,
        'num_conv_per_stage': 2,
        'feat_map_mul_on_downscale': 2,
        'conv_op': 'conv2d',
        'norm_op': 'batchnorm2d',
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True, 'momentum': 0.1},
        'dropout_op': 'dropout2d',
        'dropout_op_kwargs': {'p': 0.5, 'inplace': True},
        'nonlin': 'leakyrelu',
        'nonlin_kwargs': {'negative_slope': 1e-2, 'inplace': True},
        'deep_supervision': False,
        'dropout_in_localization': False,
        'final_nonlin': 'sigmoid',
        'weightInitializer': 'init_parameter',
        'pool_op_kernel_sizes': None,
        'conv_kernel_sizes': None,
        'upscale_logits': False,
        'convolutional_pooling': True,
        'convolutional_upsampling': True,
        'max_num_features': None,
        'seg_output_use_bias': False,
    },
}
