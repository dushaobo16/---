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

![](https://ask.qcloudimg.com/http-save/yehe-1651294/raeyvqul5r.png?imageView2/2/w/1620)

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

![](https://ask.qcloudimg.com/http-save/yehe-1651294/akj3vh1wn1.jpeg?imageView2/2/w/1620)

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

![](https://ask.qcloudimg.com/http-save/yehe-1651294/cfix7alidi.png?imageView2/2/w/1620)



# 参考文献

https://cloud.tencent.com/developer/article/1490456
