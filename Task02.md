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

```
import cv2
import matplotlib.pyplot as plt
img = cv2.imread(r'D:\opencv_data\aloeL.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)   
plt.imshow(img)
```
>注意：通过OpenCV读入的图像数据是以BGR的形式读入，如果需要显示，则需要转换为RGB制式。

应用效果：
<img src="https://github.com/dushaobo16/city-map-segment/blob/main/image/QQ%E6%88%AA%E5%9B%BE20210222204048.png?raw=true" width="200" height="200"/><br/>

# 2 数据扩增
* 2.1 数据扩增的概念

数据扩增是指不实际增加原始数据，只是对原始数据做一些变换，从而创造出更多的数据。

* 2.2 数据扩增的目的

数据扩增的目的是增加数据量、丰富数据多样性、提高模型的泛化能力。

* 2.3 数据扩增的基本原则 

   >* 不能引入无关的数据

   >* 扩增总是基于先验知识的，对于不同的任务和场景，数据扩增的策略也会不同。

   >* 扩增后的标签保持不变

* 2.4 数据扩增的方法

数据扩增方法有很多：从颜色空间、尺度空间到样本空间，同时根据不同任务数据扩增都有相应的区别。

对于图像分类，数据扩增一般不会改变标签；对于物体检测，数据扩增会改变物体坐标位置；对于图像分割，数据扩增会改变像素标签。

以torchvision.transforms为例，首先整体了解数据扩增的方法，包括：
#2.4.1  裁剪

中心裁剪：transforms.CenterCrop；

随机裁剪：transforms.RandomCrop；

随机长宽比裁剪：transforms.RandomResizedCrop；

上下左右中心裁剪：transforms.FiveCrop；

上下左右中心裁剪后翻转: transforms.TenCrop。

#2.4.2  翻转和旋转

依概率p水平翻转：transforms.RandomHorizontalFlip(p=0.5)；

依概率p垂直翻转：transforms.RandomVerticalFlip(p=0.5)；

随机旋转：transforms.RandomRotation。

#2.4.3  随机遮挡

对图像进行随机遮挡: transforms.RandomErasing。

#2.4.4  图像变换

尺寸变换：transforms.Resize；

标准化：transforms.Normalize；

填充：transforms.Pad；

修改亮度、对比度和饱和度：transforms.ColorJitter；

转灰度图：transforms.Grayscale；

依概率p转为灰度图：transforms.RandomGrayscale；

线性变换：transforms.LinearTransformation()；

仿射变换：transforms.RandomAffine；

将数据转换为PILImage：transforms.ToPILImage；

转为tensor，并归一化至[0-1]：transforms.ToTensor；

用户自定义方法：transforms.Lambda。

#2.4.5  对transforms操作，使数据增强更灵活

transforms.RandomChoice(transforms): 从给定的一系列transforms中选一个进行操作；

transforms.RandomApply(transforms, p=0.5): 给一个transform加上概率，依概率进行操作；

transforms.RandomOrder: 将transforms中的操作随机打乱。

演示示例
```
from PIL import Image
from torchvision import transforms
img = Image.open(r'D:\opencv_data\aloeL.jpg')
#随机比例缩放
new_img = transforms.Resize((200,300))(img)
print(f'{img.size}---->{new_img.size}')
new_img.save('./1.jpg')
```

(1282, 1110)---->(300, 200)
![](https://github.com/dushaobo16/city-map-segment/blob/main/image/1.jpg?raw=true)
```
#随机位置裁剪
new_img = transforms.RandomCrop(500)(img)
new_img.save('./2_1.jpg')
new_img = transforms.CenterCrop(600)(img)
new_img.save('./2_2.jpg')
```

![](https://github.com/dushaobo16/city-map-segment/blob/main/image/2_1.jpg?raw=true) ![](https://github.com/dushaobo16/city-map-segment/blob/main/image/2_2.jpg?raw=true)
```
#随机水平/垂直翻转
new_img = transforms.RandomHorizontalFlip(p=1)(img)  #p表示图像翻转的概率,default p=0.5
new_img.save('./3_1.jpg')
new_img = transforms.RandomVerticalFlip(p=1)(img)
new_img.save('./3_2.jpg')
```

```
#随机角度旋转
new_img  = transforms.RandomRotation(45)(img) #表示 45度旋转
new_img.save('./4.jpg')
```

```
#色度、亮度、饱和度、对比度的变化
new_img = transforms.ColorJitter(brightness=1)(img)
# new_img = transforms.ColorJitter(contrast=1)(img)
# new_img = transforms.ColorJitter(saturation=0.5)(img)
# new_img = transforms.ColorJitter(hue=0.5)(img)
new_img.save('./5_1.jpg')
```

```
#进行随机的灰度化
new_img = transforms.RandomGrayscale(p=0.5)(img)
new_img.save('./6.jpg')
```

```
data_transform={'train':transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                    }
```


# 参考文献

https://zhuanlan.zhihu.com/p/143946401
https://zhuanlan.zhihu.com/p/71140833
https://blog.csdn.net/chenran187906/article/details/106836622/
https://github.com/pytorch/vision
