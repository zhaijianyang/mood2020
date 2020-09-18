import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import random
from math import ceil
import time

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from util.nifti_io import ni_save, ni_load
from util.constant import *


def init_parameter(module):
    if hasattr(module, 'weight'):
        nn.init.constant_(module.weight, 1e-2)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, 1e-2)


# 将origin_dict中的值换成map中的值。
def transform_dict_data(origin_dict, map):
    map_keys = map.keys()
    for key, value in origin_dict.items():
        if value in list(map_keys):
            origin_dict[key] = map[value]


def cv2_canny(np_array, low, high):
    from cv2 import Canny
    low = int(low * 256)
    high = int(high * 256)
    temp = np_array * 256
    temp = temp.astype(np.uint8)
    temp = Canny(temp, low, high)
    np_array = temp.astype(np.float32) / 256
    return np_array

def cv2_canny_img(img, low, high):
    from cv2 import Canny
    low = int(low * 256)
    high = int(high * 256)
    img = Canny(img, low, high)
    return img

def array_to_img(np_array):
    temp = np_array * 256
    img = temp.astype(np.uint8)
    return img

def img_to_array(img):
    temp = img.astype(np.float32)
    array = temp / 256
    return array

def canny_ex():
    from cv2 import imwrite, medianBlur, blur
    save_path = '/home/cxr/Downloads'
    # path = '/home/cxr/Program_Datas/mood/validate/data/c4_00020_0_1.nii.gz'
    # path = '/home/cxr/Program_Datas/mood/brain_toy/data/toy_1.nii.gz'
    # path = '/home/cxr/Program_Datas/mood/validate/data/n2_00002_flair_13.nii.gz'
    path = '/home/cxr/Program_Datas/mood/validate_abdom/data/n13_00313_2.nii.gz'
    np_array, np_aff = ni_load(path)
    np_array = np_array[:,256,:]
    
    path = os.path.join(save_path, 'origin.png')
    origin_img = (np_array * 256).astype(np.uint8)
    imwrite(path, origin_img)

    kernel = 7
    smooth_img = blur(origin_img, (kernel, kernel))
    # img = medianBlur(img, 20)
    path = os.path.join(save_path, 'smooth.png')
    imwrite(path, smooth_img)

    highs = np.arange(0, 1, step=0.05)
    lows = np.arange(0, 1, step=0.05)
    for low in lows:
        for high in highs:
            if high <= low: continue
            file_name = f'{low:.2f}-{high:.2f}.png'
            path = os.path.join(save_path, file_name)
            # array = cv2_canny(np_array, low, high)
            canny_img = cv2_canny_img(smooth_img, low, high)

            # img = (img * 256).astype(np.uint8)
            imwrite(path, canny_img)

            path = os.path.join(save_path, 'a'+ file_name)
            new_new_img = np.copy(origin_img)
            new_new_img[canny_img == 0] = 0
            imwrite(path, new_new_img)


def save_images(pred_dir, f_name, ni_aff, result):
    ni_aff = ni_aff.astype(np.float64)

    if 'score' in result.keys():
        score_dir = os.path.join(pred_dir, 'score')
        if not os.path.exists(score_dir): os.mkdir(score_dir)
        score = result['score'].astype(np.float64)
        ni_save(os.path.join(score_dir, f_name), score, ni_aff)

    if 'rec' in result.keys():
        rec_dir = os.path.join(pred_dir, 'rec')
        if not os.path.exists(rec_dir): os.mkdir(rec_dir)
        rec = result['rec'].astype(np.float64)
        ni_save(os.path.join(rec_dir, f_name), rec, ni_aff)

    if 'input' in result.keys():
        input_dir = os.path.join(pred_dir, 'input')
        if not os.path.exists(input_dir): os.mkdir(input_dir)
        input = result['input'].astype(np.float64)
        ni_save(os.path.join(input_dir, f_name), input, ni_aff)


def clip_image(input_folder, output_folder):
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        if os.path.isdir(folder_path):
            if not os.path.exists(os.path.join(output_folder, folder_name)):
                os.mkdir(os.path.join(output_folder, folder_name))
            for f_name in os.listdir(folder_path):
                ni_file = os.path.join(folder_path, f_name)
                ni_data, ni_affine = ni_load(ni_file)
                ni_data = np.clip(ni_data, a_max=1.0, a_min=0.0)
                ni_save(os.path.join(output_folder, folder_name, f_name), ni_data, ni_affine)


def fuse_score(fuse_pixel_score_dir, fuse_sample_score_dir, mode, *pixel_score_dirs):
    from util.constant import AFFINE
    
    f_names = os.listdir(pixel_score_dirs[0])
    length = len(f_names)
    handle = tqdm(enumerate(f_names))
    for i, f_name in handle:
        handle.set_description_str(f'{i}/{length}')

        pixel_scores = []
        for pixel_score_dir in pixel_score_dirs:
            score, _ = ni_load(os.path.join(pixel_score_dir, f_name))
            pixel_scores.append(score)

        # pixel_score = np.mean(pixel_scores, axis=0)
        if mode == 'abdom':
            canny_w = 2
            mask_w = 8
            wei = (mask_w, mask_w, mask_w, canny_w, canny_w, canny_w)
            # wei = (1,1,1)
            pixel_score = np.average(pixel_scores, axis=0, weights=wei)
        else:
            canny_w = 1
            mask_w = 9
            wei = (mask_w, mask_w, mask_w, canny_w, canny_w, canny_w)
            # wei = (1,1,1)
            pixel_score = np.average(pixel_scores, axis=0, weights=wei)
        ni_save(os.path.join(fuse_pixel_score_dir, f_name), pixel_score, AFFINE)

        sample_score = get_sample_score(pixel_score)
        with open(os.path.join(fuse_sample_score_dir, f_name + '.txt'), 'w') as target_file:
            target_file.write(str(sample_score))




def fuse(mode, *scores):
    if len(scores) == 6:
        if mode == 'abdom':
            canny_w = 2
            mask_w = 8
            wei = (mask_w, mask_w, mask_w, canny_w, canny_w, canny_w)
        elif mode == 'brain':
            canny_w = 3
            mask_w = 7
            wei = (mask_w, mask_w, mask_w, canny_w, canny_w, canny_w)
        return np.average(scores, axis=0, weights=wei)
    elif len(scores) == 3:
        wei = (1, 1, 1)
        return np.average(scores, axis=0, weights=wei)
    else:
        return scores

# def get_attention(score, avg):



def statistics(test_dir, algo_name):
    print('statistics')
    import matplotlib.pyplot as plt

    predict_dir = os.path.join(test_dir, 'eval', algo_name, 'predict')
    assert os.path.exists(predict_dir), '先预测，再统计'
    statistics_dir = os.path.join(predict_dir, 'statistics')
    if not os.path.exists(statistics_dir):
        os.mkdir(statistics_dir)

    file_names = os.listdir(os.path.join(predict_dir, 'pixel', 'score'))
    length = len(file_names)
    handle = tqdm(enumerate(file_names))
    for i, file_name in handle:
        handle.set_description_str(f'{i}/{length}')

        prefix = file_name.split('.')[0]
        each_statistics_dir = os.path.join(statistics_dir, prefix)
        if not os.path.exists(each_statistics_dir): os.mkdir(each_statistics_dir)

        score, ni_aff = ni_load(os.path.join(predict_dir, 'pixel', 'score', file_name))
        flatten_score = score.flatten()

        # 整体打分直方图
        plt.hist(flatten_score, bins=50, log=False)
        plt.savefig(os.path.join(each_statistics_dir, 'whole_score_histogram'))
        plt.cla()

        with open(os.path.join(test_dir, 'label', 'sample', file_name + '.txt'), "r") as f:
            sample_label = f.readline()
        sample_label = int(sample_label)

        if sample_label == 1:
            # 异常区域打分直方图
            label, _ = ni_load(os.path.join(test_dir, 'label', 'pixel', file_name))
            abnormal_area_score = score[label == 1]
            plt.hist(abnormal_area_score, bins=50, log=False)
            plt.savefig(os.path.join(each_statistics_dir, 'abnormal_area_score_histogram'))
            plt.cla()

            abnormal_number = len(abnormal_area_score)
            # print(f'abnormal_number: {abnormal_number}')
        elif sample_label == 0:
            abnormal_number = 10000
        else: raise Exception(f'sample_label有问题: {sample_label}')

        # 高分区域打分直方图
        ordered_flatten_score = np.sort(flatten_score)[::-1]
        large_score = ordered_flatten_score[0: abnormal_number]
        plt.hist(large_score, bins=50, log=False)
        plt.savefig(os.path.join(each_statistics_dir, 'max_score_area_score_histogram'))
        plt.cla()

        max_score = large_score[0]
        img = score / max_score
        ni_save(os.path.join(each_statistics_dir, 'normalized'), img, ni_aff)


def fuse_ex(test_dir, *algo_names):
    from scripts.evalresults import eval_dir
    from scripts.evalresults_per_sample import eval_dir_per_sample
    score_dir, pred_pixel_dir, pred_sample_dir = init_validation_dir('fuse', test_dir)
    with open(os.path.join(test_dir, 'eval', 'fuse', 'readme'), 'w') as target_file:
        target_file.write(str(algo_names))

    pred_pixel_dir = os.path.join(pred_pixel_dir, 'score')
    if not os.path.exists(pred_pixel_dir): os.mkdir(pred_pixel_dir)

    pixel_score_dirs = []
    for algo_name in algo_names:
        pixel_score_dirs.append(os.path.join(test_dir, 'eval', algo_name, 'predict', 'pixel', 'score'))
    
    print('predict')
    fuse_score(pred_pixel_dir, pred_sample_dir, *pixel_score_dirs)

    # print('validate')
    # eval_dir_per_sample(pred_dir=pred_pixel_dir, label_dir=os.path.join(test_dir, 'label', 'pixel'), mode='pixel', save_file=os.path.join(score_dir, 'pixel'))
    # eval_dir(pred_dir=pred_sample_dir, label_dir=os.path.join(test_dir, 'label', 'sample'), mode='sample', save_file=os.path.join(score_dir, 'sample'))

def fuse_ex_new(test_dir, mode, *algo_names):
    pixel_score_dirs = []
    for algo_name in algo_names:
        pixel_score_dirs.append(os.path.join(test_dir, 'eval', algo_name, 'predict', 'pixel', 'score'))
    fuse_score(test_dir, test_dir, mode, *pixel_score_dirs)


def template_match_ex(test_dir): # 读进来的是nii.gz
    from util.configure import TRAIN_DATASET_DIR
    from scripts.evalresults import eval_dir

    score_dir, pred_pixel_dir, pred_sample_dir = init_validation_dir('temmat', test_dir)
    templates = load_array(os.path.join(TRAIN_DATASET_DIR, 'preprocessed'))

    print('predict')
    for f_name in os.listdir(os.path.join(test_dir, 'data')):
        print(f'f_name: {f_name}')
        np_array, ni_aff = ni_load(os.path.join(test_dir, 'data', f_name))

        score, rec = template_match(templates, np_array)
        save_images(pred_pixel_dir, f_name, ni_aff, score=score, rec=rec)

        sample_score = get_sample_score(score)
        with open(os.path.join(pred_sample_dir, f_name + ".txt"), "w") as target_file:
            target_file.write(str(sample_score))
    
    eval_dir(pred_dir=os.path.join(pred_pixel_dir, 'score'), label_dir=os.path.join(test_dir, 'label', 'pixel'), mode='pixel', save_file=os.path.join(score_dir, 'pixel'))
    eval_dir(pred_dir=pred_sample_dir, label_dir=os.path.join(test_dir, 'label', 'sample'), mode='sample', save_file=os.path.join(score_dir, 'sample'))


def template_match(imgs, np_array): # imgs 四维
    min_score_num = np.inf
    min_score_index = -1

    length = len(imgs)
    handle = tqdm(enumerate(imgs))
    for i, img in handle:
        score = (img - np_array) ** 2
        score_num = np.sum(score)
        if score_num < min_score_num:
            min_score_num = score_num
            min_score_index = i

        handle.set_description_str(f'{i+1}/{length}')

    rec = imgs[min_score_index]
    score = (rec - np_array) ** 2
    return score, rec


def get_sample_score(score):
    slice_scores = []
    for sli in score:
        slice_score = np.mean(sli)
        slice_scores.append(slice_score)
    return np.max(slice_scores)


def load_array(path):
    print(f'load_array')
    imgs = []
    handle = tqdm(os.listdir(path))
    for fname in handle:
        if fname.endswith('data.npy'):
            np_array = np.load(os.path.join(path, fname))
            imgs.append(np_array)
    return imgs


def init_validation_dir(algo_name, dataset_dir):
    eval_dir = os.path.join(dataset_dir, 'eval')
    if not os.path.exists(eval_dir):    os.mkdir(eval_dir)
    algo_dir = os.path.join(eval_dir, algo_name)
    if not os.path.exists(algo_dir):    os.mkdir(algo_dir)
    pred_dir = os.path.join(algo_dir, 'predict')
    if not os.path.exists(pred_dir):    os.mkdir(pred_dir)
    score_dir = os.path.join(algo_dir, 'score')
    if not os.path.exists(score_dir):   os.mkdir(score_dir)

    pred_pixel_dir = os.path.join(pred_dir, 'pixel')
    if not os.path.exists(pred_pixel_dir):  os.mkdir(pred_pixel_dir)
    pred_sample_dir = os.path.join(pred_dir, 'sample')
    if not os.path.exists(pred_sample_dir): os.mkdir(pred_sample_dir)

    return score_dir, pred_pixel_dir, pred_sample_dir

