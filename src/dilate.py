# Machine Learning for Times Series
# DILATE Project
#
# Ben Kabongo & Martin Brosset
# M2 MVA


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *


def dilate(y_pred: torch.Tensor, y_true: torch.Tensor, alpha: float, gamma: float,
           Delta: torch.Tensor, Omega: torch.Tensor):
    pass
    