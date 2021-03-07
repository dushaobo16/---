# 模型集成

## 1 前言

> 集成学习是指通过构建并结合多个学习器来完成学任务的分类系统

在机器学习中可以通过Stacking、Bagging、Boosting等常见的集成学习方法来提高预测精度，而在深度学习中，可以通过交叉验证的方法训练多个CNN模型，然后对这些训练好的模型进行集成就可以得到集成模型，从而提高字符识别的精度。如下图：

![](https://img-blog.csdnimg.cn/20200602185149275.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM3ODMzMTQy,size_16,color_FFFFFF,t_70#pic_center)

以上通过10折交叉验证，可训练得到10个CNN模型，集成方法有：

* 平均法：将10个模型预测结果的概率取平均值，然后解码为具体字符
* 投票法：对10个模型预测结果进行投票，得到最终字符

## 2 Pytorch实现模型集成

### 2.1 Dropout

Dropout作为一种DL训练技巧，是在每个Batch中，通过随机让一部分的节点停止工作减少过拟合，同样可以在预测模型时增加模型的精度，在训练模型中添加Dropout的方式为：

```
# 定义模型
class SVHN_Model1(nn.Module):
	def __init__(self):
		super(SVHN_Model1, self).__init__()
		# CNN提取特征模块
		self.cnn = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2)),
			nn.ReLU(),
			nn.Dropout(0.25),  # 1/4随机失活
			nn.MaxPool2d(2),
			nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2)),
			nn.ReLU(),
			nn.Dropout(0.25),  # 1/4随机失活
 			nn.MaxPool2d(2),
 		)
 		#全连接层
		self.fc1 = nn.Linear(32*3*7, 11)
		self.fc2 = nn.Linear(32*3*7, 11)
		self.fc3 = nn.Linear(32*3*7, 11)
		self.fc4 = nn.Linear(32*3*7, 11)
		self.fc5 = nn.Linear(32*3*7, 11)
		self.fc6 = nn.Linear(32*3*7, 11)

	def forward(self, img): 
		feat = self.cnn(img)
		feat = feat.view(feat.shape[0], -1)
		c1 = self.fc1(feat)
		c2 = self.fc2(feat)
		c3 = self.fc3(feat)
		c4 = self.fc4(feat)
		c5 = self.fc5(feat)
		c6 = self.fc6(feat)
		return c1, c2, c3, c4, c5, c6

```


### 2.2 TTA

测试集数据扩增（Test Time Augmention，TTA）也是常用的集成学习技巧，数据扩增不仅可以用在训练时候，而且可以在预测时候进行数据扩充.
> train的时候我们经常加入data augmentation， 比如旋转，对比度调整，gamma变换等等，其实本质上是为了增加泛化性。在test的时候，同样可以加入augmented images，相当于一个ensemble，模型分数也会有所提高。

```
import torch

class Test_time_agumentation(object):

    def __init__(self, is_rotation=True):
        self.is_rotation = is_rotation

    def __rotation(self, img):
        """
        clockwise rotation 90 180 270
        """
        img90 = img.rot90(-1, [2, 3]) # 1 逆时针； -1 顺时针
        img180 = img.rot90(-1, [2, 3]).rot90(-1, [2, 3])
        img270 = img.rot90(1, [2, 3])
        return [img90, img180, img270]

    def __inverse_rotation(self, img90, img180, img270):
        """
        anticlockwise rotation 90 180 270
        """
        img90 = img90.rot90(1, [2, 3]) # 1 逆时针； -1 顺时针
        img180 = img180.rot90(1, [2, 3]).rot90(1, [2, 3])
        img270 = img270.rot90(-1, [2, 3])
        return img90, img180, img270

    def __flip(self, img):
        """
        Flip vertically and horizontally
        """
        return [img.flip(2), img.flip(3)]

    def __inverse_flip(self, img_v, img_h):
        """
        Flip vertically and horizontally
        """
        return img_v.flip(2), img_h.flip(3)

    def tensor_rotation(self, img):
        """
        img size: [H, W]
        rotation degree: [90 180 270]
        :return a rotated list
        """
        # assert img.shape == (1024, 1024)
        return self.__rotation(img)

    def tensor_inverse_rotation(self, img_list):
        """
        img size: [H, W]
        rotation degree: [90 180 270]
        :return a rotated list
        """
        # assert img.shape == (1024, 1024)
        return self.__inverse_rotation(img_list[0], img_list[1], img_list[2])

    def tensor_flip(self, img):
        """
        img size: [H, W]
        :return a flipped list
        """
        # assert img.shape == (1024, 1024)
        return self.__flip(img)

    def tensor_inverse_flip(self, img_list):
        """
        img size: [H, W]
        :return a flipped list
        """
        # assert img.shape == (1024, 1024)
        return self.__inverse_flip(img_list[0], img_list[1])


if __name__ == "__main__":
    a = torch.tensor([[0, 1],[2, 3]]).unsqueeze(0).unsqueeze(0)
    print(a)
    tta = Test_time_agumentation()
    # a = tta.tensor_rotation(a)
    a = tta.tensor_flip(a)
    print(a)
    a = tta.tensor_inverse_flip(a)
    print(a)

```
 
 在DataWhale学习任务中给出一个简单的使用方法：
 
 ```
 for idx, name in enumerate(tqdm_notebook(glob.glob('./test_mask/*.png')[:])):
    image = cv2.imread(name)
    image = trfm(image)
    with torch.no_grad():
        image = image.to(DEVICE)[None]
        score1 = model(image).cpu().numpy()
        
        score2 = model(torch.flip(image, [0, 3]))
        score2 = torch.flip(score2, [3, 0]).cpu().numpy()

        score3 = model(torch.flip(image, [0, 2]))
        score3 = torch.flip(score3, [2, 0]).cpu().numpy()
        
        score = (score1 + score2 + score3) / 3.0
        score_sigmoid = score[0].argmax(0) + 1 
 
 ```

### 2.3 Snapshot

在论文Snapshot Ensembles中，作者提出使用cyclical learning rate进行训练模型，并保存精度比较好的一些checkopint，最后将多个checkpoint进行模型集成。

![](https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/Snapshot.png)

由于在cyclical learning rate中学习率的变化有周期性变大和减少的行为，因此CNN模型很有可能在跳出局部最优进入另一个局部最优。在Snapshot论文中作者通过使用表明，此种方法可以在一定程度上提高模型精度，但需要更长的训练时间。

![](https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/%E5%AF%B9%E6%AF%94.png)

```
def proposed_lr(initial_lr, iteration, epoch_per_cycle):
    # proposed learning late function
    return initial_lr * (cos(pi * iteration / epoch_per_cycle) + 1) / 2
    

 snapshots = []
    _lr_list, _loss_list = [], []
    count = 0
    epochs_per_cycle = epochs // cycles
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)

    for i in range(cycles):

        for j in range(epochs_per_cycle):
            _epoch_loss = 0

            lr = proposed_lr(initial_lr, j, epochs_per_cycle)
            optimizer.state_dict()["param_groups"][0]["lr"] = lr

            for batch_idx, (data, target) in enumerate(train_loader):
                if cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                _epoch_loss += loss.data[0]/len(train_loader)
                loss.backward()
                optimizer.step()

            _lr_list.append(lr)
            _loss_list.append(_epoch_loss)
            count += 1

            if vis is not None and j % 10 == 0:
                vis.line(np.array(_lr_list), np.arange(count), win="lr",
                         opts=dict(title="learning rate",
                                   xlabel="epochs",
                                   ylabel="learning rate (s.e.)"))
                vis.line(np.array(_loss_list), np.arange(count),  win="loss",
                         opts=dict(title="loss",
                                   xlabel="epochs",
                                   ylabel="training loss (s.e.)"))

        snapshots.append(model.state_dict())
    return snapshots

```





# 参考文献

https://blog.csdn.net/m0_37833142/article/details/106502602

https://github.com/datawhalechina/team-learning-cv/blob/master/AerialImageSegmentation/Task6%EF%BC%9A%E6%A8%A1%E5%9E%8B%E9%9B%86%E6%88%90.md

https://blog.csdn.net/qq_39575835/article/details/103933933
