# 数据读取与数据扩增方法
* 1.数据读取

由于赛题数据是图像数据，赛题的任务是识别图像中的字符。因此我们首先需要完成对数据的读取操作，在Python中有很多库可以完成数据读取的操作，比较常见的有Pillow和OpenCV。</br>
* 1.1 Pillow

 > Pillow是Python图像处理函式库(PIL）的一个分支。Pillow提供了常见的图像读取和处理的操作，而且可以与ipython notebook无缝集成，是应用比较广泛的库。
    
```
from PIL import Image
import matplotlib.pyplot as plt
im = Image.open(r'D:\opencv_data\aloeL.jpg')
```
应用效果：

<img src="https://github.com/dushaobo16/city-map-segment/blob/main/image/QQ%E6%88%AA%E5%9B%BE20210222202509.png?raw=true" width="200" height="200"/><br/>
* 2.1 OpenCV
> OpenCV是一个跨平台的计算机视觉库，最早由Intel开源得来。OpenCV发展的非常早，拥有众多的计算机视觉、数字图像处理和机器视觉等功能。OpenCV在功能上比Pillow更加强大很多，学习成本也高很多。




#参考文献
https://zhuanlan.zhihu.com/p/143946401
