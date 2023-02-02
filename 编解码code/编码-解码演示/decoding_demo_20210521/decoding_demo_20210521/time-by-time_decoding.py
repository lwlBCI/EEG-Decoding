# -*- coding: utf-8

"""
@File       :   time-by-time_decoding.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

import scipy.io as sio
import numpy as np


# 被试id
subs = ["201", "202", "203", "204", "205"]

data = np.zeros([5, 640, 27, 500])
label = np.zeros([5, 640])

sub_index = 0
for sub in subs:  # 大循环是每一个受试者

    # 加载单个被试的ERP数据
    subdata = sio.loadmat("data/data/ERP" + sub + ".mat")["filtData"]

    # shape of subdata: 640*27*750
    # 640 - 试次数=16*40,16种条件，每种条件下40次 试次trials； 27 - 导联； 750 - 时间点  250Hz， 1 - 0.004s  3s [-1.5s至1.5s]
    # 采样率为250Hz，那么-1.5s-1.5s总共3s的数据就是750个点

    # 只关心-0.5s至1.5s
    subdata = subdata[:, :, 250:]  # 0.5s就是从250个点开始

    sublabel = np.loadtxt("data/labels/ori_" + sub + ".txt")[:, 1]  # 这地方ori可以改为pos，ori=方向，pos=位置

    data[sub_index] = subdata
    label[sub_index] = sublabel

    sub_index = sub_index + 1

from neurora.decoding import tbyt_decoding_kfold  # tbyt=time by time

accs = tbyt_decoding_kfold(data, label, n=16, navg=13, time_win=5, time_step=5, nfolds=3, nrepeats=10, smooth=True)
#  对于这些参数的解释：n=16为16种条件下，也就是分类的数目，13代表的是：对n(本数据中为40)个试次中的13个做一个平均，40/13=3余1
#  time_win=5 代表每五个点做一次平均，联合time_step=5可以这样来解释： 1 2 3 4 5 6 7 8 9 10，这10个点如果time_win=5而time_step=1
# 那么就是1 2 3 4 5 做平均，下一次2 3 4 5 6做平均，再下次3 4 5 6 7做平均
# 如果time_win=5而time_step=5，那么就是 1 2 3 4 5做平均，然后下一次6 7 8 9 10做平均
# nfolds=3代表交叉验证的次数为3，也就是说每次训练取2个样本作为训练，1个样本作为测试，nrepeats=10整个实验的大过程重复10次

np.savetxt("time-by-time_results.txt", accs)
print(accs.shape)  # shape=(5,100),五个受试者，100指的是每个受试者的500个数据每5个取了平均，最终为100个，每一个进行16分类得到一个准确率