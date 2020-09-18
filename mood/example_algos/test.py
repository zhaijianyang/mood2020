import numpy as np 
import os
import fnmatch
import tqdm

base_dir = '/home/zhai/文档/mood_data/abdom/preprocessed'
pattern = '*data.npy'
save_dir = '/media/zhai/新加卷/mood_data/abdom_train/preprocessed'

all_files = os.listdir(base_dir)
npy_files = fnmatch.filter(all_files, pattern)

for i, filename in enumerate(sorted(npy_files)):

    npy_file = os.path.join(base_dir, filename)
    numpy_array = np.load(npy_file, mmap_mode="r")

    file_len = numpy_array.shape[1]

    for j in range(file_len):

        np.save(os.path.join(save_dir, f'{i}_{j}_data.npy'), numpy_array[j])
        print(f'第{i}张{j}个')

    del numpy_array
    