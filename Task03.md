# 语言分割模型发展

## 3 语义分割模型发展

### 3.1 什么是语义分割？

语义分割是从粗推理到精推理的自然步骤：

原点可以定位在分类，分类包括对整个输入进行预测。

下一步是本地化/检测，它不仅提供类，还提供关于这些类的空间位置的附加信息。

最后，语义分割通过对每个像素进行密集的预测、推断标签来实现细粒度的推理，从而使每个像素都被标记为其封闭对象矿石区域的类别。

![](https://img-blog.csdnimg.cn/20190102165307605.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDQ0NjU1Nw==,size_16,color_FFFFFF,t_70)

语义分割（全像素语义分割）作为经典的计算机视觉任务（图像分类，物体识别检测，语义分割）。其结合了图像分类、目标检测和图像分割，通过一定的方法将图像分割成具有一定语义含义的区域块，并识别出每个区域块的语义类别，实现从底层到高层的语义推理过程，最终得到一幅具有逐像素语义标注的分割图像。
### 3.2 语义分割的基础

也有必要回顾一些对计算机视觉领域做出重大贡献的标准深层网络，因为它们通常被用作语义分割系统的基础：

Alexnet:Toronto首创的Deep CNN，以84.6%的测试准确率赢得了2012年Imagenet竞赛。它由5个卷积层、最大池层、作为非线性的ReLUs、3个完全卷积层和dropout组成。

VGG-16：这款牛津型号以92.7%的准确率赢得了2013年的Imagenet竞争。它使用第一层中具有小接收场的卷积层堆栈，而不是具有大接收场的少数层。

GoogLeNet：这GoogLeNet赢得了2014年Imagenet的竞争，准确率为93.3%。它由22层和一个新引入的称为初始模块的构建块组成。该模块由网络层网络、池操作、大卷积层和小卷积层组成。

Resnet：这款微软的模型以96.4%的准确率赢得了2016年的Imagenet竞争。这是众所周知的，因为它的深度（152层）和残余块的引进。剩余的块通过引入标识跳过连接来解决培训真正深层架构的问题，以便层可以将其输入复制到下一层。
![](https://img-blog.csdnimg.cn/20190102170326280.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDQ0NjU1Nw==,size_16,color_FFFFFF,t_70)

### 3.3FCN
#### FCN原理及网络结构 

Fully Conv简称FCN为全连接网络，将将一幅RGB图像输入到卷积神经网络后，经过多次卷积以及池化过程得到一系列的特征图，然后利用反卷积层对最后一个卷积层得到的特征图进行上采样，使得上采样后特征图与原图像的大小一样，从而实现对特征图上的每个像素值进行预测的同时保留其在原图像中的空间位置信息，最后对上采样特征图进行逐像素分类，逐个像素计算softmax分类损失。

主要特点：

* 不含全连接层（FC）的全卷积（Fully Conv）网络。从而可适应任意尺寸输入。*

* 引入增大数据尺寸的反卷积（Deconv）层。能够输出精细的结果。

* 结合不同深度层结果的跳级（skip）结构。同时确保鲁棒性和精确性。

![](https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/fcn.jpg)

# 参考文献
https://blog.csdn.net/weixin_40446557/article/details/85624579
