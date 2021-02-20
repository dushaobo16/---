# 赛题简介
本次新人赛是Datawhale与天池联合发起的零基础入门系列赛事第七场 —— 零基础入门语义分割之地表建筑物识别挑战赛。

赛题以计算机视觉为背景，要求选手使用给定的航拍图像训练模型并完成地表建筑物识别任务。为更好的引导大家入门，我们为本赛题定制了学习方案和学习任务，具体包括语义分割的模型和具体的应用案例。在具体任务中我们将讲解具体工具和使用和完成任务的过程。

通过对本方案的完整学习，可以帮助掌握语义分割基本技能。同时我们也将提供专属的视频直播学习通道。

新人赛的目的主要是为了更好地带动处于初学者阶段的新同学们一起玩起来，因此，我们鼓励所有选手，基于赛题发表notebook分享，内容包含但不限于对赛题的理解、数据分析及可视化、算法模型的分析以及一些核心的思路等内容。
#  赛题数据
遥感技术已成为获取地表覆盖信息最为行之有效的手段，遥感技术已经成功应用于地表覆盖检测、植被面积检测和建筑物检测任务。本赛题使用航拍数据，需要参赛选手完成地表建筑物识别，将地表航拍图像素划分为有建筑物和无建筑物两类。

如下图，左边为原始航拍图，右边为对应的建筑物标注。
![](https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/data-example.png)
赛题数据来源（Inria Aerial Image Labeling），并进行拆分处理。数据集报名后可见并可下载。赛题数据为航拍图，需要参赛选手识别图片中的地表建筑具体像素位置。
#数据标签
赛题为语义分割任务，因此具体的标签为图像像素类别。在赛题数据中像素属于2类（无建筑物和有建筑物），因此标签为有建筑物的像素。赛题原始图片为jpg格式，标签为RLE编码的字符串。

RLE全称（run-length encoding），翻译为游程编码或行程长度编码，对连续的黑、白像素数以不同的码字进行编码。RLE是一种简单的非破坏性资料压缩法，经常用在在语义分割比赛中对标签进行编码。
大家都知道在编码过程中尽量不要有大量冗余代码，这样不但会使代码变的可读性差，而且不易于管理。那么同样的如何对字符串进行冗余字符处理。其实非常简单，请看一下一个字符串：

 

JJJJJJAAAAVVVVAAAAAA

这个字符串可以用更简洁的方式来编码，那就是通过替换每一个重复的字符串为单个的实例字符加上记录重复次数的数字来表示，上面的字符串可以被编码为下面的形式：

6J4A4V6A
在这里，"6J"意味着6个字符J，"4A"意味着4个字符A，以此类推。这种字符串压缩方式称为"行程长度编码"方式，简称RLE。
RLE与图片之间的转换如下：
```
import numpy as np
import pandas as pd
import cv2

# 将图片编码为rle格式
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# 将rle格式进行解码为图片
def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')
```


   
参考文献
https://www.iteye.com/blog/zhaohongda33-1137937
