# -*- coding: utf-8

"""
@File       :   plot_time-by-time_accs.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

from neurora.rsa_plot import plot_tbyt_decoding_acc
import numpy as np

nfolds=4

accs = np.loadtxt("time-by-time_results.txt")

plot_tbyt_decoding_acc(accs, start_time=-0.5, end_time=1.5, time_interval=0.02, chance=0.0625, p=0.05, cbpt=False,
                           stats_time=[0, 1.5], color='r', xlim=[-0.5, 1.5], ylim=[0.05, 0.15], figsize=[6.4, 3.6], x0=0,
                           fontsize=12, avgshow=False)
#  开始时刻，结束时刻，时间间隔，随机准确率，显著性水平，刺激呈现后的时间，颜色，x轴，y轴，图像大小，y轴在x轴的位置
#  画的是五个受试者的平均正确率