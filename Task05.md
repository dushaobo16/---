# 模型训练与验证

一个完善的深度学习模型应具有以下几步骤：
* 准备数据，将数据集划分为训练集和验证集
* 编写模型
* 训练模型
* 在训练模型过程中应当保存最优的权重，并读取权重；
* 记录下训练集和验证集的精度，便于调参

# 5 模型训练与验证

## 5.1 构造验证集

在训练模型过程中，随着神经网络模型层数的增加和训练步数的增加极容易导致模型过拟合的发生。

在模型的训练过程中，模型只能利用训练数据来进行训练，模型并不能接触到测试集上的样本。因此模型如果将训练集学的过好，模型就会记住训练样本的细节，导致模型在测试集的泛化效果较差，这种现象称为过拟合（Overfitting）。与过拟合相对应的是欠拟合（Underfitting），即模型在训练集上的拟合效果较差。

![](https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/loss.png)

如图所示：随着模型复杂度和模型训练轮数的增加，CNN模型在训练集上的误差会降低，但在测试集上的误差会逐渐降低，然后逐渐升高，而我们为了追求的是模型在测试集上的精度越高越好。
导致模型过拟合的情况有很多种原因，其中最为常见的情况是模型复杂度（Model Complexity ）太高，导致模型学习到了训练数据的方方面面，学习到了一些细枝末节的规律。
解决上述问题最好的解决方法：构建一个与测试集尽可能分布一致的样本集（可称为验证集），在训练过程中不断验证模型在验证集上的精度，并以此控制模型的训练。

在给定赛题后，赛题方会给定训练集和测试集两部分数据。参赛者需要在训练集上面构建模型，并在测试集上面验证模型的泛化能力。因此参赛者可以通过提交模型对测试集的预测结果，来验证自己模型的泛化能力。同时参赛方也会限制一些提交的次数限制，以此避免参赛选手“刷分”。

在一般情况下，参赛选手也可以自己在本地划分出一个验证集出来，进行本地验证。训练集、验证集和测试集分别有不同的作用：

* 训练集（Train Set）：模型用于训练和调整模型参数；
* 验证集（Validation Set）：用来验证模型精度和调整模型超参数；
* 测试集（Test Set）：验证模型的泛化能力。

因为训练集和验证集是分开的，所以模型在验证集上面的精度在一定程度上可以反映模型的泛化能力。在划分验证集的时候，需要注意验证集的分布应该与测试集尽量保持一致，不然模型在验证集上的精度就失去了指导意义。

既然验证集这么重要，那么如何划分本地验证集呢。在一些比赛中，赛题方会给定验证集；如果赛题方没有给定验证集，那么参赛选手就需要从训练集中拆分一部分得到验证集。验证集的划分有如下几种方式：

![](https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/%E9%AA%8C%E8%AF%81%E9%9B%86%E6%9E%84%E9%80%A0.png)

![](https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/%E9%AA%8C%E8%AF%81%E9%9B%86%E6%9E%84%E9%80%A0.png)

## 留出法（Hold-Out）

直接将训练集划分成两部分，新的训练集和验证集。这种划分方式的优点是最为直接简单；缺点是只得到了一份验证集，有可能导致模型在验证集上过拟合。留出法应用场景是数据量比较大的情况。

![](https://upload-images.jianshu.io/upload_images/1667471-9db53006d07c7d20.png)

## 交叉验证法（Cross Validation，CV）

将训练集划分成K份，将其中的K-1份作为训练集，剩余的1份作为验证集，循环K训练。这种划分方式是所有的训练集都是验证集，最终模型验证精度是K份平均得到。这种方式的优点是验证集精度比较可靠，训练K次可以得到K个有多样性差异的模型；CV验证的缺点是需要训练K次，不适合数据量很大的情况。

![](https://upload-images.jianshu.io/upload_images/1667471-7ddeb02e0be14b79.png)

k 折交叉验证通过对 k 个不同分组训练的结果进行平均来减少方差，因此模型的性能对数据的划分就不那么敏感。

* 第一步，不重复抽样将原始数据随机分为 k 份。
* 第二步，每一次挑选其中 1 份作为测试集，剩余 k-1 份作为训练集用于模型训练。
* 第三步，重复第二步 k 次，这样每个子集都有一次机会作为测试集，其余机会作为训练集。
* 在每个训练集上训练后得到一个模型，
* 用这个模型在相应的测试集上测试，计算并保存模型的评估指标，
* 第四步，计算 k 组测试结果的平均值作为模型精度的估计，并作为当前 k 折交叉验证下模型的性能指标。

## 自助采样法（BootStrap）

通过有放回的采样方式得到新的训练集和验证集，每次的训练集和验证集都是有区别的。这种划分方式一般适用于数据量较小的情况。

在本次赛题中已经划分为验证集，因此选手可以直接使用训练集进行训练，并使用验证集进行验证精度（当然你也可以合并训练集和验证集，自行划分验证集）。

当然这些划分方法是从数据划分方式的角度来讲的，在现有的数据比赛中一般采用的划分方法是留出法和交叉验证法。如果数据量比较大，留出法还是比较合适的。当然任何的验证集的划分得到的验证集都是要保证训练集-验证集-测试集的分布是一致的，所以如果不管划分何种的划分方式都是需要注意的。

这里的分布一般指的是与标签相关的统计分布，比如在分类任务中“分布”指的是标签的类别分布，训练集-验证集-测试集的类别分布情况应该大体一致；如果标签是带有时序信息，则验证集和测试集的时间间隔应该保持一致。

![](https://image.jiqizhixin.com/uploads/editor/62733eb3-e4a0-4554-a5b8-a5b1af3beb4d/1525376957137.jpg)


## 5.2 模型训练与验证

使用Pytorch来完成CNN的训练和验证过程，CNN网络结构与之前的章节中保持一致。我们需要完成的逻辑结构如下：

* 构造训练集和验证集；
* 每轮进行训练和验证，并根据最优验证集精度保存模型。

```
train_loader = torch.utils.data.DataLoader(
train_dataset,
batch_size = 10,
shuffle = True,
num_workers=10,
)

val_loader = torch.utils.data.DataLoader(
val_dataset,
batch_size=10,
shuffle=False,
num_workers=10,
)
model=Model()
criterion = nn.CrossEntropyLoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(),0.001)
best_loss = 1000.0
for epoch in range(20):
    print('Epoch:',epoch)
    train(train_loader, model, criterion, optimizer, epoch)
    val_loss = validate(val_loader, model, criterion)
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), './model.pt')

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    for i, (inputs, target) in enumerate(train_loader):
        # 正向传播
        outputs = model(inputs)
        
        # 计算Loss
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        optimizer.step()
        
def validate(val_loader, model, criterion):
    model.eval()
    val_loss = []
    with torch.no_grad():
         for i, (inputs, target) in enumerate(val_loader):
             outputs = model(inputs)
             loss = criterion(outputs, target)
             val_loss.append(loss)

```

## 5.3 模型保存与加载

在Pytorch中模型的保存和加载非常简单，比较常见的做法是保存和加载模型参数：
```
torch.save(model_object.state_dict(), 'model.pt')
model.load_state_dict(torch.load(' model.pt'))
```

## 5.4 模型调参流程

深度学习原理少但实践性非常强，基本上很多的模型的验证只能通过训练来完成。同时深度学习有众多的网络结构和超参数，因此需要反复尝试。训练深度学习模型需要GPU的硬件支持，也需要较多的训练时间，如何有效的训练深度学习模型逐渐成为了一门学问。


在参加本次比赛的过程中，我建议大家以如下逻辑完成：

* 初步构建简单的CNN模型，不用特别复杂，跑通训练、验证和预测的流程；
* 简单CNN模型的损失会比较大，尝试增加模型复杂度，并观察验证集精度；
* 在增加模型复杂度的同时增加数据扩增方法，直至验证集精度不变。

![](https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/%E8%B0%83%E5%8F%82%E6%B5%81%E7%A8%8B.png)




# 参考文献

https://www.cnblogs.com/sddai/p/8379452.html
https://github.com/datawhalechina/team-learning-cv/blob/master/AerialImageSegmentation/Task5%EF%BC%9A%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E4%B8%8E%E9%AA%8C%E8%AF%81.md
