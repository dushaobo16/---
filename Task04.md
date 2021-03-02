
# 4评价函数与损失函数

## 4.1 评价指标

在机器学习领域中，用于评价一个模型的性能有多种指标，其中几项就是FP、FN、TP、TN、精确率(Precision)、召回率(Recall)、准确率(Accuracy)。

>  你这蠢货，是不是又把酸葡萄和葡萄酸弄“混淆“啦！！！

上面日常情况中的混淆就是：是否把某两件东西或者多件东西给弄混了，迷糊了

> 在机器学习中, 混淆矩阵是一个误差矩阵, 常用来可视化地评估监督学习算法的性能。混淆矩阵大小为 (n_classes, n_classes) 的方阵, 其中 n_classes 表示类的数量。

其中，这个矩阵的一行表示预测类中的实例（可以理解为模型预测输出，predict），另一列表示对该预测结果与标签（Ground Truth）进行判定模型的预测结果是否正确，正确为True，反之为False。

> 在机器学习中ground truth表示有监督学习的训练集的分类准确性，用于证明或者推翻某个假设。有监督的机器学习会对训练数据打标记，试想一下如果训练标记错误，那么将会对测试数据的预测产生影响，因此这里将那些正确打标记的数据成为ground truth。

此时，就引入FP、FN、TP、TN与精确率(Precision)，召回率(Recall)，准确率(Accuracy)。

以猫狗二分类为例，假定cat为正例-Positive，dog为负例-Negative；预测正确为True，反之为False。我们就可以得到下面这样一个表示FP、FN、TP、TN的表：
![](https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/Task4%EF%BC%9A%E8%AF%84%E4%BB%B7%E5%87%BD%E6%95%B0%E4%B8%8E%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0_image/%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png)
![](https://github.com/dushaobo16/city-map-segment/blob/main/image/Task04_img/1.png?raw=true)

```
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
sns.set()

f, (ax1,ax2) = plt.subplots(figsize = (, ),nrows=)
y_true = ["dog", "dog", "dog", "cat", "cat", "cat", "cat"]
y_pred = ["cat", "cat", "dog", "cat", "cat", "cat", "cat"]
C2= confusion_matrix(y_true, y_pred, labels=["dog", "cat"])
print(C2)
print(C2.ravel())
sns.heatmap(C2,annot=True)

ax2.set_title('sns_heatmap_confusion_matrix')
ax2.set_xlabel('Pred')
ax2.set_ylabel('True')
f.savefig('sns_heatmap_confusion_matrix.jpg', bbox_inches='tight')
```

![](https://github.com/dushaobo16/city-map-segment/blob/main/image/Task04_img/2.jpeg?raw=true)

这个时候我们还是不知道skearn.metrics.confusion_matrix做了些什么，这个时候print(C2)，打印看下C2究竟里面包含着什么。最终的打印结果如下所示：
```
[[1 2]
 [0 4]]
[1 2 0 4]
```

* cat为1-positive，其中真实值中cat有4个，4个被预测为cat，预测正确T，0个被预测为dog，预测错误F；
* dog为0-negative，其中真实值中dog有3个，1个被预测为dog，预测正确T，2个被预测为cat，预测错误F。

所以：TN=1、 FP=2 、FN=0、TP=4。

* TN=1：预测为negative狗中1个被预测正确了
* FP=2 ：预测为positive猫中2个被预测错误了
* FN=0：预测为negative狗中0个被预测错误了
* TP=4：预测为positive猫中4个被预测正确了

![](https://github.com/dushaobo16/city-map-segment/blob/main/image/Task04_img/3.png?raw=true)

这时候再把上面猫狗预测结果拿来看看，6个被预测为cat，但是只有4个的true是cat，此时就和右侧的红圈对应上了。

```
y_pred = ["cat", "cat", "dog", "cat", "cat", "cat", "cat"]

y_true = ["dog", "dog", "dog", "cat", "cat", "cat", "cat"]
```

## 精确率(Precision)、召回率(Recall)、准确率(Accuracy)
> 有了上面的这些数值，就可以进行如下的计算工作了

准确率(Accuracy)：这三个指标里最直观的就是准确率: 模型判断正确的数据(TP+TN)占总数据的比例
```
"Accuracy: "+str(round((tp+tn)/(tp+fp+fn+tn), ))
```

> 召回率(Recall)：针对数据集中的所有正例(TP+FN)而言,模型正确判断出的正例(TP)占数据集中所有正例的比例.FN表示被模型误认为是负例但实际是正例的数据.召回率也叫查全率,以物体检测为例,我们往往把图片中的物体作为正例,此时召回率高代表着模型可以找出图片中更多的物体!

```
"Recall: "+str(round((tp)/(tp+fn), ))
```

> 精确率(Precision)：针对模型判断出的所有正例(TP+FP)而言,其中真正例(TP)占的比例.精确率也叫查准率,还是以物体检测为例,精确率高表示模型检测出的物体中大部分确实是物体,只有少量不是物体的对象被当成物体
> 
```
"Precision: "+str(round((tp)/(tp+fp), ))
```

在语义分割任务中来，我们可以将语义分割看作是对每一个图像像素的的分类问题。根据混淆矩阵中的定义，我们亦可以将特定像素所属的集合或区域划分成TP、TN、 FP、FN四类。
![](https://github.com/datawhalechina/team-learning-cv/blob/master/AerialImageSegmentation/img/Task4%EF%BC%9A%E8%AF%84%E4%BB%B7%E5%87%BD%E6%95%B0%E4%B8%8E%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0_image/%E5%88%86%E5%89%B2%E5%AF%B9%E7%85%A7.png)

以上面的图片为例，图中左子图中的人物区域（黄色像素集合）是我们真实标注的#前景信息（target）#，其他区域（紫色像素集合）为背景信息。当经过预测之后，我们会得到的一张预测结果，图中右子图中的黄色像素为预测的前景（prediction），紫色像素为预测的背景区域。此时，我们便能够将预测结果分成4个部分：

* 预测结果中的黄色无线区域 → 真实的前景 → 的所有像素集合被称为真正例（TP）<预测正确>

* 预测结果中的蓝色斜线区域 → 真实的背景 → 的所有像素集合被称为假正例（FP）<预测错误>

* 预测结果中的红色斜线区域 → 真实的前景 → 的所有像素集合被称为假反例（FN）<预测错误>

* 预测结果中的白色斜线区域 → 真实的背景 → 的所有像素集合被称为真反例（TN）<预测正确>

## 4.2 Dice评价指标

### Dice系数

Dice系数（Dice coefficient）是常见的评价分割效果的方法之一，同样也可以改写成损失函数用来度量prediction和target之间的距离。
Dice系数定义如下：

$$Dice (T, P) = \frac{2 |T \cap P|}{|T| \cup |P|} = \frac{2TP}{FP+2TP+FN}$$

其中，$T$表示真实前景（target）,$P$表示预测前景（Prediction）。Dice系数取值范围为$[0,1]$,其中值 为1时代表预测与真实完全一致。仔细观察，Dice系数 与分类评价指标中的F1 Score很相似：
$$ \frac{1}{F1} = \frac{1}{Precision} + \frac{1}{Recall} \ F1 = \frac{2TP}{FP+2TP+FN} $$

# 参考文献

https://cloud.tencent.com/developer/article/1490456
