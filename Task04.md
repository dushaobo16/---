
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

![](https://github.com/dushaobo16/city-map-segment/blob/main/image/Task04_img/dice.png?raw=true)

$$
\Dice (T, P) = \frac{2 |T \cap P|}{|T| \cup |P|} = \frac{2TP}{FP+2TP+FN}
$$

其中，$T$表示真实前景（target）,$P$表示预测前景（Prediction）。Dice系数取值范围为$[0,1]$,其中值 为1时代表预测与真实完全一致。仔细观察，Dice系数 与分类评价指标中的F1 Score很相似：

![](https://github.com/dushaobo16/city-map-segment/blob/main/image/Task04_img/f1_score.png?raw=true)

$$ \frac{1}{F1} = \frac{1}{Precision} + \frac{1}{Recall} \ F1 = \frac{2TP}{FP+2TP+FN} $$

### Dice Loss

Dice Loss是在V-net模型中被提出应用的，是通过Dice系数转变而来，其实为了能够实现最小化的损失函数，以方便模型训练，以$1 - Dice$的形式作为损失函数：

![](https://pic1.zhimg.com/80/v2-3e7e3fc41f0ddf24bcbff2ddfeb0684c_1440w.png)

Laplace smoothing:

Laplace smoothing 是一个可选改动，即将分子分母全部加 1：

![](https://pic1.zhimg.com/80/v2-3511b4330023535ca029fdfa66d41dc4_1440w.jpg)

带来的好处：

（1）避免当|X|和|Y|都为0时，分子被0除的问题

（2）减少过拟合

## dice 系数计算

预测的分割图的 dice 系数计算，首先将 |X⋂Y| 近似为预测图与 GT 分割图之间的点乘，并将点乘的元素结果相加：
[1] - Pred 预测分割图与 GT 分割图的点乘：

![](https://img-blog.csdnimg.cn/20190727173958436.jpg)

[2] - 逐元素相乘的结果元素的相加和：

![](https://img-blog.csdnimg.cn/20190727174007745.jpg)

对于二分类问题，GT 分割图是只有 0, 1 两个值的，因此 |X⋂Y| 可以有效的将在 Pred 分割图中未在 GT 分割图中激活的所有像素清零. 对于激活的像素，主要是惩罚低置信度的预测，较高值会得到更好的 Dice 系数.

关于 |X| 和 |Y| 的量化计算，可采用直接简单的元素相加；也有采用取元素平方求和的做法：

![](https://img-blog.csdnimg.cn/20190727174018765.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0pNVV9NYQ==,size_16,color_FFFFFF,t_70)

**注**：dice loss 比较适用于样本极度不均的情况，一般的情况下，使用 dice loss 会对反向传播造成不利的影响，容易使训练变得不稳定.
## 代码实现

```
import numpy as np

def dice(output, target):
    '''计算Dice系数'''
    smooth = 1e-6 # 避免0为除数
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

# 生成随机两个矩阵测试
target = np.random.randint(0, 2, (3, 3))
output = np.random.randint(0, 2, (3, 3))

d = dice(output, target)
# ----------------------------
target = array([[1, 0, 0],
       			[0, 1, 1],
			    [0, 0, 1]])
output = array([[1, 0, 1],
       			[0, 1, 0],
       			[0, 0, 0]])
d = 0.5714286326530524

```
## 4.3 IoU评价指标

IoU 的全称为交并比（Intersection over Union），通过这个名称我们大概可以猜到 IoU 的计算方法。IoU 计算的是 “预测的边框” 和 “真实的边框” 的交集和并集的比值。
![](https://img-blog.csdn.net/20180922220708895?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQwNjE2MzA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

![](https://www.zhihu.com/equation?tex=IoU+%3D+%5Cfrac%7B%5Cleft%7C+A%5Ccap+B+%5Cright%7C%7D%7B%5Cleft%7C+A%5Ccup+B+%5Cright%7C%7D)


target      |     prediction
------------|----------------
![](https://github.com/datawhalechina/team-learning-cv/blob/master/AerialImageSegmentation/img/Task4%EF%BC%9A%E8%AF%84%E4%BB%B7%E5%87%BD%E6%95%B0%E4%B8%8E%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0_image/target.png)   | ![](https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/Task4%EF%BC%9A%E8%AF%84%E4%BB%B7%E5%87%BD%E6%95%B0%E4%B8%8E%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0_image/prediction.png)
Intersection($T\cap P$)   |  union($T\cup P$)
![](https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/Task4%EF%BC%9A%E8%AF%84%E4%BB%B7%E5%87%BD%E6%95%B0%E4%B8%8E%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0_image/intersection.png)  | ![](https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/Task4%EF%BC%9A%E8%AF%84%E4%BB%B7%E5%87%BD%E6%95%B0%E4%B8%8E%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0_image/union.png)

### 代码实现 

```
def iou_score(output, target):
    '''计算IoU指标'''
	  intersection = np.logical_and(target, output) 
    union = np.logical_or(target, output) 
    return np.sum(intersection) / np.sum(union)

# 生成随机两个矩阵测试
target = np.random.randint(0, 2, (3, 3))
output = np.random.randint(0, 2, (3, 3))

d = iou_score(output, target)
# ----------------------------
target = array([[1, 0, 0],
       			[0, 1, 1],
			    [0, 0, 1]])
output = array([[1, 0, 1],
       			[0, 1, 0],
       			[0, 0, 0]])
d = 0.4

```

## 4.4 BCE损失函数

BCE损失函数（Binary Cross-Entropy Loss）是交叉熵损失函数（Cross-Entropy Loss）的一种特例，BCE Loss只应用在二分类任务中。

![](https://pic3.zhimg.com/80/v2-4cea328253a1d17f8a53a8eb0513ba92_1440w.png)

pytorch还提供了已经结合了Sigmoid函数的BCE损失：torch.nn.BCEWithLogitsLoss()，相当于免去了实现进行Sigmoid激活的操作。

```
import torch
import torch.nn as nn

bce = nn.BCELoss()
bce_sig = nn.BCEWithLogitsLoss()

input = torch.randn(5, 1, requires_grad=True)
target = torch.empty(5, 1).random_(2)
pre = nn.Sigmoid()(input)

loss_bce = bce(pre, target)
loss_bce_sig = bce_sig(input, target)

# ------------------------
input = tensor([[-0.2296],
        		[-0.6389],
        		[-0.2405],
        		[ 1.3451],
        		[ 0.7580]], requires_grad=True)
output = tensor([[1.],
        		 [0.],
        		 [0.],
        		 [1.],
        		 [1.]])
pre = tensor([[0.4428],
        	  [0.3455],
        	  [0.4402],
        	  [0.7933],
        	  [0.6809]], grad_fn=<SigmoidBackward>)

print(loss_bce)
tensor(0.4869, grad_fn=<BinaryCrossEntropyBackward>)

print(loss_bce_sig)
tensor(0.4869, grad_fn=<BinaryCrossEntropyWithLogitsBackward>)


```

### 4.5 Focal Loss

Focal loss最初是出现在目标检测领域，主要是为了解决正负样本比例失调的问题。那么对于分割任务来说，如果存在数据不均衡的情况，也可以借用focal loss来进行缓解。
Focal loss是在交叉熵损失函数基础上进行的修改，首先回顾二分类交叉上损失：
![](https://images2018.cnblogs.com/blog/1055519/201808/1055519-20180818162755861-24998254.png)

![](https://images2018.cnblogs.com/blog/1055519/201808/1055519-20180818162835223-1945881125.png)是经过激活函数的输出，所以在0-1之间。可见普通的交叉熵对于正样本而言，输出概率越大损失越小。对于负样本而言，输出概率越小则损失越小。此时的损失函数在大量简单样本的迭代过程中比较缓慢且可能无法优化至最优。那么Focal loss是怎么改进的呢？

![](https://images2018.cnblogs.com/blog/1055519/201808/1055519-20180818174822290-765890427.png)

![](https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/Task4%EF%BC%9A%E8%AF%84%E4%BB%B7%E5%87%BD%E6%95%B0%E4%B8%8E%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0_image/FocalLoss.png)

简单来说：$α$解决样本不平衡问题，$γ$解决样本难易问题。

也就是说，当数据不均衡时，可以根据比例设置合适的$α$，这个很好理解，为了能够使得正负样本得到的损失能够均衡，因此对loss前面加上一定的权重，其中负样本数量多，因此占用的权重可以设置的小一点；正样本数量少，就对正样本产生的损失的权重设的高一点。

那γ具体怎么起作用呢？以图中$γ=5$曲线为例，假设$gt$类别为1，当模型预测结果为1的概率$p_t$比较大时，我们认为模型预测的比较准确，也就是说这个样本比较简单。而对于比较简单的样本，我们希望提供的loss小一些而让模型主要学习难一些的样本，也就是$p_t→ 1$则loss接近于0，既不用再特别学习；当分类错误时，$p_t → 0$则loss正常产生，继续学习。对比图中蓝色和绿色曲线，可以看到，γ值越大，当模型预测结果比较准确的时候能提供更小的loss，符合我们为简单样本降低loss的预期。

## 代码实现
```
import torch.nn as nn
import torch
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits	# 如果BEC带logits则损失函数在计算BECloss之前会自动计算softmax/sigmoid将其映射到[0,1]
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

# ------------------------

FL1 = FocalLoss(logits=False)
FL2 = FocalLoss(logits=True)

inputs = torch.randn(5, 1, requires_grad=True)
targets = torch.empty(5, 1).random_(2)
pre = nn.Sigmoid()(inputs)

f_loss_1 = FL1(pre, targets)
f_loss_2 = FL2(inputs, targets)

# ------------------------

print('inputs:', inputs)
inputs: tensor([[-1.3521],
        [ 0.4975],
        [-1.0178],
        [-0.3859],
        [-0.2923]], requires_grad=True)
    
print('targets:', targets)
targets: tensor([[1.],
        [1.],
        [0.],
        [1.],
        [1.]])
    
print('pre:', pre)
pre: tensor([[0.2055],
        [0.6219],
        [0.2655],
        [0.4047],
        [0.4274]], grad_fn=<SigmoidBackward>)
    
print('f_loss_1:', f_loss_1)
f_loss_1: tensor(0.3375, grad_fn=<MeanBackward0>)
    
print('f_loss_2', f_loss_2)
f_loss_2 tensor(0.3375, grad_fn=<MeanBackward0>)

```
## 4.6 Lovász-Softmax

论文提出了LovaszSoftmax，是一种基于IOU的loss，效果优于cross_entropy，可以在分割任务中使用。最终在Pascal VOC和 Cityscapes 两个数据集上取得了最好的结果。
cross_entropy loss:

![](https://img-blog.csdnimg.cn/20190118113533803.png)

Softmax 函数：

![](https://img-blog.csdnimg.cn/20190118113547280.png)

Jaccard index 

![](https://img-blog.csdnimg.cn/2019011811361294.png)

优化的IOU loss:

![](https://img-blog.csdnimg.cn/20190118113625479.png)

论文贡献：

* 结合Lovasz hinge 和Jaccard loss 解决2值图片的分割问题
* 提出了Lovasz-Softmax loss 对多个类别分割的参数设置
* 设计了一个基于batch的IOU作为基于dataset IOU的高效代理
* 分析和对比各种IOU测量方法
* 基于本文的loss,对经典的分割方法的分割效果做出很大的提升

定义1：

![](https://img-blog.csdnimg.cn/20190118113655600.png)

定义2：

![](https://img-blog.csdnimg.cn/20190118113710527.png)

Lovász-Softmax实现链接：https://github.com/bermanmaxim/LovaszSoftmax

```
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse
    
# --------------------------- MULTICLASS LOSSES ---------------------------
def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)

# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x
    
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

```





# 参考文献

https://cloud.tencent.com/developer/article/1490456
https://zhuanlan.zhihu.com/p/86704421
https://blog.csdn.net/h1239757443/article/details/108457082
https://blog.csdn.net/JMU_Ma/article/details/97533768
https://blog.csdn.net/u014061630/article/details/82818112
https://zhuanlan.zhihu.com/p/138592268
https://www.cnblogs.com/king-lps/p/9497836.html
