## 脑电信号解码的证据

### **多变量模式分析 MVPA**

:star:首先要介绍的是**[MVPA](http://www.360doc.com/content/21/0804/12/70369197_989471613.shtml)**这个专业名词，我们通过一个例子来进行介绍：

<img src="https://cdn.jsdelivr.net/gh/lwlBCI/EEG-Decoding/imagesimage-20230202110849098.png" alt="image-20230202110849098" style="zoom:50%;" />

:blue_heart:根据图中的信息我们可以这样理解：假设我们现在用条件1为瓶子，条件2为鞋子，在观看这两件物品后大脑中的反应来进行解码，图中蓝绿色的点分别看作神经活动在高维空间上的分布，**如果能找到可以一个线性分类器可以很好地把这两种信息分离开来，那么我们便可以认为大脑在解析或者说编码水瓶和鞋子这两种信息的时候是存在着两种模式的，那么也就可以认为大脑可以编码类别信息**

:purple_heart:总结而言便是：如果对于多种类别条件下的刺激，大脑可以加工这些类别的信息的话，那么我们可以用一个分类器或者说是切平面可以解码出这种差异也就可以反向地推导出大脑在编码这多种类别信息

## EEG层面的实现

<img src="https://cdn.jsdelivr.net/gh/lwlBCI/EEG-Decoding/imagesimage-20230202120857892.png" alt="image-20230202120857892" style="zoom:50%;" />

:heart:在EEG的层面是这样的，由于我们可以提供的导联(电极通道）是非常多的，因此可提供的脑电信息也是非常多通道的，假定在condition1情况下，在某一个固定时间点(例如任务开始后200ms这一时刻），得到了32个通道的脑电信息，如果我们可以用一个**切平面(高维空间中不再是线性分类所能分的开的)**将这些信息分开，得到一个在不同conditions下分类的准确率，这样就可以认为是**一个时间点对应一个准确率**，便可以画出 time-classifier accuracy这条曲线了，我们得到的那个切平面就可以看作是我们利用数据来训练的一个分类器

:green_heart:**在刺激未呈现之前，大脑所能编码信息的能力可以看作是随机的，也就是说如果编码两类信息的话，每类信息的准确率是0.5，如果是10类那么每类的随机准确率为1/10=0.1**

:exclamation:我们想做的内容就是逐个时间点进行eeg-decoding的操作，也就是得到上图中右下角这条曲线

### 逐时间点的EEG Decoding

<img src="https://cdn.jsdelivr.net/gh/lwlBCI/EEG-Decoding/imagesimage-20230202111807081.png" alt="image-20230202111807081" style="zoom: 67%;" />

:droplet:这是19年发表的[一篇文章](https://www.jneurosci.org/content/38/2/409.abstract)，里面所做的实验是对于16个Orientation和16个location信息进行编码，得到的最终结论为Alpha波段在编码Location的信息时非常有效，ERP在编码Orientation和16个location都有效

:droplet:并且在这里我们抛出一个问题：在基于ERP的解码模式下，**400ms处的解码模式和800ms的解码模式是否相同**，换句话而言，如果我们在以400ms处的信息训练一个分类器，:question:那么这个分类器是否能在800ms时有同样好的表现呢？这就牵扯到EEG的跨时域解码问题

### 跨时域的EEG Decoding

<img src="https://cdn.jsdelivr.net/gh/lwlBCI/EEG-Decoding/imagesimage-20230202112325565.png" alt="image-20230202112325565" style="zoom:80%;" />

:punch:在[上面这幅图](https://www.sciencedirect.com/science/article/pii/S105381191731087X)我们可以看到，横坐标为训练好的对应不同时刻的分类器，纵坐标为相同时间点，得到的值的含义为：**使用某一时刻的信息来训练另一时刻信息的准确率**，可以清晰地看到的是大概只有在对角线的位置上准确率才能够有比较高的显示。这句话的具体理解为：当我们用100ms的信息来检测200ms的时候，如图中所示是一块蓝色的区域，**这代表100ms的脑电解码并不能解码出200ms的信息**，这也就是说虽然大脑在任务开始后一直都对这项任务进行编码但是不同时间点的编码模式一直是在变化的

### 跨任务的EEG Decoding

在上面的介绍中我们探讨了跨时域的脑信号解码，那么我们可能会想是否可以从另一个角度：**从跨不同认知任务的角度训练分类器，在特定的一项任务中得到的分类器是否能够在其他任务中展示出较好的性能？**

![image-20230202113248584](https://cdn.jsdelivr.net/gh/lwlBCI/EEG-Decoding/imagesimage-20230202113248584.png)

:see_no_evil:在19年的[这篇文章](https://www.sciencedirect.com/science/article/pii/S1364661319300592)中提到了关于**视觉感知和图像的共享神经机制**，视觉感知任务和想象任务分别为让受试者观察不同房子和面孔，想象不同房子和面孔，以这样的实验目的为：探索在相同情景下，对于不同的任务方式EEG的解码方式是否相同，可以看到的是在0.3-1s的时间段内，大脑中 **观察和想象这两项工作** 对**房子和面孔的加工过程** 是同样的一种反应模式。

### 代码实现---基于2019那篇

:earth_asia:整个文件的代码被包含在"decoding_demo_20210521"当中，初次运行的时候需要先运行"prepare_data.py"获取5个受试者的数据，并且此时会新建一个data文件夹，里面包含data和label文件夹，data文件夹中包含"ERP201-ERP205"的数据，每个.mat文件的shape为"640 x 27 x 750"，其中640=16*40,16个条件(也就是说分类类别数为16)，每种条件下40个试次，27为导联数也就是通道电极数，750为采样点数，-1.5s --- 1.5s采样率为250Hz，一共750个采样点，事实上在论文中实际关心的是-0.5s---1.5s也就是500个采样点。

:earth_americas:关于labels数据需要解释的是：在论文中分为orientation和location两类任务，并且每一类任务都有16个分类类别，因此对于ori来说，把135°规定为数值6, 292.5°规定为数值13，以此类比...而在pos中同样的，把270规定为12， 225规定为10，以此类比...其实说白了就是原本的真实试验中的标签以角度为单位太过繁琐，将其替换为简单的数值更加通俗易懂也更容易训练。

😚 需要特别特别指出的是：本代码与数据的提供来自于[![LZT](https://img.shields.io/github/followers/ZitongLu1996?label=LZT&style=social)](https://github.com/ZitongLu1996)，本人仅进行了部分修改以及注释的编写

#### **time-by-time decoding**

:full_moon:时域解码的最重要一行代码(第三行)为：

```python
from neurora.decoding import tbyt_decoding_kfold  # tbyt=time by time
# 这里的neurora是路同学所在团队开发的一个开源库，可以非常方便的进行ERP层面的EEG 解码，这里向大家安利一下~
accs = tbyt_decoding_kfold(data, label, n=16, navg=13, time_win=5, time_step=5, nfolds=3, nrepeats=10, smooth=True)
"""
对于这些参数的解释：n=16为16种条件下，也就是分类的数目，13代表的是：对n(本数据中为40)个试次中的13个做一个平均，40/13=3余1
time_win=5 代表每五个点做一次平均，联合time_step=5可以这样来解释： 1 2 3 4 5 6 7 8 9 10，这10个点如果time_win=5而	 time_step=1
那么就是1 2 3 4 5 做平均，下一次2 3 4 5 6做平均，再下次3 4 5 6 7做平均
如果time_win=5而time_step=5，那么就是 1 2 3 4 5做平均，然后下一次6 7 8 9 10做平均
nfolds=3代表交叉验证的次数为3，也就是说每次训练取2个样本作为训练，1个样本作为测试，nrepeats=10整个实验的大过程重复10次
"""
np.savetxt("time-by-time_results.txt", accs)
print(accs.shape)  # shape=(5,100),五个受试者，100指的是每个受试者的500个数据每5个取了平均，最终为100个，每一个进行16分类得到一个准确率
```

:full_moon_with_face:代码为参考论文：[Dissociable Decoding of Spatial Attention and Working Memory from EEG Oscillations and Sustained Potentials 基于EEG振荡和持续电位对空间注意和工作记忆的分离解码](https://www.jneurosci.org/content/38/2/409.abstract) 所实现，因此数据与参数均与论文中所提到的一致

#### **plot_time_by_time_acc**

:bell:这里是将上面的准确率结果进行绘图：

```python
plot_tbyt_decoding_acc(accs, start_time=-0.5, end_time=1.5, time_interval=0.02, chance=0.0625, p=0.05, cbpt=False,stats_time=[0, 1.5], color='r', xlim=[-0.5, 1.5], ylim=[0.05, 0.15], figsize=[6.4, 3.6], x0=0,
fontsize=12, avgshow=False)
#  开始时刻，结束时刻，时间间隔，随机准确率，显著性水平，刺激呈现后的时间，颜色，x轴，y轴，图像大小，y轴在x轴的位置
#  画的是五个受试者的平均正确率
```

![image-20230202115920231](https://cdn.jsdelivr.net/gh/lwlBCI/EEG-Decoding/imagesimage-20230202115920231.png)

:strawberry:**可以看到从0.25s=250ms左右开始出现显著性，粉红色阴影区域代表显著性时刻，而显著性时刻代表此时大脑进行编码过程**

通常我们认为的ERP出现在300ms左右，这里也差不多其实

#### cross-temporal decoding

:peach:关键代码如下，事实上，跨时域解码的最重要目的就是寻找在某一时间点的训练模型是否在另一时间点同样适用，换句话说：我们想要探索的是，**对于同一任务下，大脑在这一时刻的解码方式是否同样适用于另一时刻**

```python
from neurora.decoding import ct_decoding_kfold
accs = ct_decoding_kfold(data, label, n=16, navg=13, time_win=5, time_step=5, nfolds=3, nrepeats=10, smooth=True)
# 这里导入的包和之前的不一样，这里是ct_decoding_kfold,意思为cross-time
# shape 5* 100 * 100，横纵轴分别都是100个点
accs = np.reshape(accs, [5, 10000])# 一个对应100，100个对应10000
np.savetxt("cross-temporal_results.txt", accs)
```

#### plot_cross_temporal_accs

![image-20230202120206135](https://cdn.jsdelivr.net/gh/lwlBCI/EEG-Decoding/imagesimage-20230202120206135.png)

:banana:最终出图的呈现效果如上，图中**灰色轮廓区域**代表的就是**显著性的位置**，也就是说在这些位置上的点可以表现出良好的跨时域特性

:melon:上述提到的那篇论文：《Dissociable Decoding of Spatial Attention and Working Memory from EEG Oscillations and Sustained Potentials》,并且在这篇论文里面，作者做了两种实验，分别是记忆水滴的位置和方向，说白了就是水滴按照一定的方向和位置排列构成一个圆形，让受试者来记忆，并且在1.5s后指出刚才水滴的位置，从刺激呈现到记忆这个刺激就是大脑参与编码解码的神经活动的过程，也就是在这个过程中出现了准确率的提升。

### Cite

Bae G Y, Luck S J. Dissociable decoding of spatial attention and working memory from EEG oscillations and sustained potentials[J]. Journal of Neuroscience, 2018, 38(2): 409-422.

Hogendoorn H, Burkitt A N. Predictive coding of visual object position ahead of moving objects revealed by time-resolved EEG decoding[J]. Neuroimage, 2018, 171: 55-61.

Dijkstra N, Bosch S E, van Gerven M A J. Shared neural mechanisms of visual perception and imagery[J]. Trends in cognitive sciences, 2019, 23(5): 423-434.

**再次感谢路同学提供的代码及数据支持：**

[![LZT](https://img.shields.io/github/followers/ZitongLu1996?label=LZT&style=social)](https://github.com/ZitongLu1996)

[![](https://img.shields.io/badge/%E8%B5%84%E6%BA%90%E5%BA%93-Neurora-brightgreen)](https://github.com/ZitongLu1996/NeuroRA)

