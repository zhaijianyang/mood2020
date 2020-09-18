import time
import os
import json

import torch

from trixi.logger import PytorchExperimentLogger

from .algorithm import Algorithm
from .cfunction import FuncFactory
from .constant import DEVICE


class AlgoFactory:
    def __init__(self):
        self.FF = FuncFactory()

    def getAlgo(self, run_mode, model_type=None, recipe=None, loss_type='l2', singleton=True, **kwargs):
        self.FF.singleton = singleton

        # 创建基本的algo
        # basic_kws永远是手动配置的，而其它两个只需要第一次手动配置，以后就会在路径中读取

        # 读取基本配置
        if 'basic_kws' in kwargs.keys():
            basic_kws = kwargs['basic_kws']
        else:
            basic_kws = None

        if basic_kws == None:
            from .configure import BASIC_KWS
            basic_kws = BASIC_KWS

        if run_mode in ['validate', 'statistics']:
            algo = Algorithm(basic_kws=basic_kws, train_kws={'model_type': model_type})
            setattr(algo, 'run', self.FF.getFunction('run', run_mode))
            return algo

        if not basic_kws['load']:
            from .configure import TRAIN_KWS, OTHER_KWS
            train_kws = TRAIN_KWS

            assert recipe is not None, '未指定recipe'
            self.FF.getFunction('modify_train_kws', OTHER_KWS, recipe)(train_kws)

            assert model_type is not None, '未指定model_type'
            train_kws['recipe'] = recipe
            train_kws['model_type'] = model_type
            train_kws['loss_type'] = loss_type
            need_to_save_config = True

        else:
            train_kws = AlgoFactory.load_config(os.path.join(basic_kws['load_path'], '../config/train_kws.json'))            # 读取训练配置
            loss_type = train_kws['loss_type']
            model_type = train_kws['model_type']
            if run_mode == 'train':
                need_to_save_config = True
            else:
                need_to_save_config = False

        algo = Algorithm(basic_kws=basic_kws, train_kws=train_kws)

        # if model_type == 'temmat':
        #     return algo

        # 为algo加入模型，在这里都是字符串，后面可能要转成对象，in_channels也在后面加
        if not basic_kws['load']:
            from .configure import CONFIGURE_DICT
            model_kws = CONFIGURE_DICT[model_type]
        else:
            model_kws = AlgoFactory.load_config(os.path.join(basic_kws['load_path'], '../config/model_kws.json'))

        if need_to_save_config:
            ex_dir = algo.tx.elog.work_dir
            AlgoFactory.save_config(data=train_kws, filename=os.path.join(basic_kws['log_dir'], ex_dir, 'config/train_kws.json'))
            AlgoFactory.save_config(data=model_kws, filename=os.path.join(basic_kws['log_dir'], ex_dir, 'config/model_kws.json'))

        self.FF.getFunction('modify_model_kws', train_kws)(model_kws)
        model = AlgoFactory.getModel(model_type=model_type, model_kws=model_kws).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_kws['lr'])
        algo.__setattr__('model', model)
        algo.__setattr__('optimizer', optimizer)

        if basic_kws['load']:
            model_path = os.path.join(basic_kws['load_path'], basic_kws['load_file_name'])
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'文件{model_path}不存在')
            algo.load_model(model_path)
            # PytorchExperimentLogger.load_model_static(model, model_path)
            time.sleep(2)
        else:
            algo.total_epoch = 0
            # algo.best_score = 0

        # 为algo设置函数
        dataset_functions, algo_functions = self.getFunctions(train_kws)
        setattr(algo, 'run', self.FF.getFunction('run', run_mode))
        algo.__setattr__('dataset_functions', dataset_functions)
        algo.__dict__.update(algo_functions)

        return algo


    @staticmethod
    def save_config(data, filename):
        with open(filename, 'w', encoding='utf8') as fp:
            json.dump(data, fp, ensure_ascii=False)


    @staticmethod
    def load_config(filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f'文件{filename}不存在')
        with open(filename, 'r', encoding='utf8') as fp:
            config_dict = json.load(fp)
        return config_dict


    @staticmethod
    def getModel(model_type, model_kws):
        if model_type == 'unet':
            from example_algos.nnunet.network_architecture.generic_UNet import Generic_UNet as Model
        elif model_type == 'zcae':
            from example_algos.model.encoders_decoders import CAE_pytorch as Model
        elif model_type == 'zunet':
            from example_algos.model.ZUNet import ZUNet as Model
        elif model_type == 'unet_3plus':
            from example_algos.model.UNet_3Plus import UNet_3Plus as Model
        return Model(**model_kws)


    def getFunctions(self, train_kws):
        dataset_functions = {
            'get_data_slice_num': None,
            'get_slices': None,
            'get_slice_data': None,
            # 'transforms': None,
        }
        algo_functions = {
            'calculate_loss': None,
            'get_input_label': None,
            'transpose': None,
            'revert_transpose': None,
            'get_pixel_score': None,
            'get_sample_score': None,
            'to_transforms': None,
        }
        
        for fn_name in dataset_functions.keys():
            dataset_functions[fn_name] = self.FF.getFunction(fn_name, train_kws)
        for fn_name in algo_functions.keys():
            algo_functions[fn_name] = self.FF.getFunction(fn_name, train_kws)

        return dataset_functions, algo_functions