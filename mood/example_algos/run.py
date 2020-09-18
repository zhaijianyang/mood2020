import os
import sys
from tqdm import tqdm
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse

from example_algos.util.factory import AlgoFactory
from util.constant import AFFINE
from util.nifti_io import ni_save

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='unet')
    parser.add_argument('--recipe', type=str, default='mask')
    parser.add_argument('--loss_type', type=str, nargs='+', default=['l2'])
    args = parser.parse_args()

    run_mode = args.run_mode
    model_type = args.model_type
    recipe = args.recipe                            # list
    loss_type = args.loss_type
    assert run_mode in ['train', 'predict', 'validate', 'statistics']
    assert model_type in ['unet', 'zcae', None]
    assert recipe in ['rec',  'rot', 'mask', 'res', 'canny', None]

    # 一个af对应一个algo，若使用多个algo则用多个af创造。
    af = AlgoFactory()
    algo = af.getAlgo(run_mode=run_mode, model_type=model_type, recipe=recipe,  loss_type=loss_type)
    algo.run(algo)


def auto_validate(args):
    from scripts.evalresults import eval_dir

    pred_dir = args.output
    label_dir = os.path.abspath(os.path.join(args.input, os.path.pardir))

    pixel_score = eval_dir(pred_dir=pred_dir, label_dir=os.path.join(label_dir, 'label', 'pixel'), mode='pixel', save_file=os.path.join(label_dir, 'pixel'))
    sample_score = eval_dir(pred_dir=pred_dir, label_dir=os.path.join(label_dir, 'label', 'sample'), mode='sample', save_file=os.path.join(label_dir, 'sample'))

    print('pixel score:', pixel_score)
    print('sample score:', sample_score)


def auto(args):
    # from example_algos.util.configure import TEST_DATASET_DIR, TRAIN_DATASET_DIR
    from example_algos.util.function import fuse, get_sample_score
    import os
    import shutil

    input_dir = args.input
    output_dir = args.output
    mode = args.mode

    model_dir = '/workspace'
    # model_dir = '/media/zhai/新加卷/mood_model_final/brain/eval'

    log_dir = os.path.join(model_dir, 'log')
    basic_kws = {
        'logger': 'tensorboard',
        'test_dir': input_dir,
        'output_dir': output_dir,
        'load': True,
        'log_dir': log_dir
    }

    if mode == 'abdom':
        config = [('unet_mask_0', 25), ('unet_mask_1', 22), ('unet_mask_2', 22), ('unet_canny_0', 24), ('unet_canny_1', 24), ('unet_canny_2', 22)]

        log_dir = os.path.join(log_dir, 'abdom')
    else:
        # config = [('unet_canny_0', 22), ('unet_canny_1', 24), ('unet_canny_2', 24)]
        # config = [('unet_mask_0', 23), ('unet_mask_1', 23), ('unet_mask_2', 23)]
        config = [('unet_mask_0', 23), ('unet_mask_1', 23), ('unet_mask_2', 23), ('unet_canny_0', 22), ('unet_canny_1', 24), ('unet_canny_2', 24)]
        log_dir = os.path.join(log_dir, 'brain')

    algos = []
    for dir_name, x in config:
        # print('load:{}'.format(dir_name))
        basic_kws['load_path'] = os.path.join(log_dir, dir_name, 'checkpoint')
        basic_kws['load_file_name'] = str(x)
        basic_kws['name'] = dir_name + '_' + str(x)
        algo = AlgoFactory().getAlgo(run_mode='predict', basic_kws=basic_kws)
        algos.append(algo)

    list_dir = os.listdir(input_dir)
    length = len(list_dir)
    handle = tqdm(enumerate(list_dir))
    for i, f_name in handle:
        handle.set_description_str(f'predict: {i+1}/{length}')
        path = os.path.join(input_dir, f_name)
        scores = []
        for algo in algos:
            score = algo.predict(path, return_sample_score=False)
            scores.append(score)
        pixel_score = fuse(mode, *scores)
        sample_score = get_sample_score(pixel_score)
        # Do something to save score
        ni_save(os.path.join(output_dir, f_name), pixel_score, AFFINE)

        with open(os.path.join(output_dir, f_name + ".txt"), "w") as target_file:
            target_file.write(str(sample_score))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/home/zhai/文档/docker_test/abdom/data')
    parser.add_argument('-o', '--output', type=str, default='/home/zhai/文档/docker_test/output/abdom')
    parser.add_argument('-r', '--mode', type=str, default='brain')

    args = parser.parse_args()

    main()
    # auto(args)
    # auto_validate(args)