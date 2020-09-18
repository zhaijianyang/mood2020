import os

import torch
import torch.nn as nn
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device('cpu')

SAVE_PATH = '/home/cxr/Downloads'

AFFINE = np.array([[0.69999999, 0.        , 0.        , 0.        ],
       [0.        , 0.78750002, 0.        , 0.        ],
       [0.        , 0.        , 0.69999999, 0.        ],
       [0.        , 0.        , 0.        , 1.        ]])