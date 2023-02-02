# -*- coding: utf-8

"""
@File       :   test.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

import numpy as np
import scipy.io as sio

data201 = sio.loadmat("data/data/ERP201.mat")["filtData"]

print(data201)
print(data201.shape)

# 640 试次
# 27 导联
# 750 时间点  250Hz  0.004s  (-1.5s  -   1.5s)     -0.5 - 1.5s