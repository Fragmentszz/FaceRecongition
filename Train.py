import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import pandas as pd
from publicmodule import *

if __name__ == '__main__':
    start = time.time()
    train_set = np.array(pd.read_excel("./imgs/Train/traits.xlsx"))
    test_set = np.array(pd.read_excel("./imgs/Test/traits.xlsx"))
    train_lable = np.array(pd.read_excel("./imgs/Train/labels.xlsx"))
    test_lable = np.array(pd.read_excel("./imgs/Test/labels.xlsx"))
    print(train_lable)
    num_label = np.unique(train_lable) # 输出神经元个数276

    # 数据预处理
    dim = (train_set.shape)[1]
    train_set,dmean,dstd = data_tf(train_set)
    test_set = data_tf_yb(test_set,dmean,dstd)
    np.save("dmean",dmean)
    np.save("dstd",dstd)
    cato = np.max(train_lable)
    print(dim,cato)

    train_set_ = GetLoader(train_set, train_lable)
    test_set_ = GetLoader(test_set, test_lable)
    train_data = DataLoader(train_set_, batch_size=32, shuffle=True)  # 训练数据
    test_data = DataLoader(test_set_, batch_size=32, shuffle=False)  # 测试数据


    # print(train_data.num_workers)
    # 定义一个类，继承自 torch.nn.Module，torch.nn.Module是callable的类
    # 在整个类里面重新定义一个标准的BP全连接神经网络，网络一共是四层，
    # 层数定义：62, 200， 350， 500， 625
    # 其中输入层62个节点，输出层是625个节点，分别代表625个人，其他的层都是隐藏层。
    # 我们使用了Relu的激活函数，而不是sigmoid激活函数
    # 整个子类需要重写forward函数，

    output = len(num_label)

    # 创建和实例化一个整个模型类的对象
    model = BPNNModel()
    model.layer1 = nn.Sequential(nn.Linear(dim, 200), nn.ReLU())
    model.layer2 = nn.Sequential(nn.Linear(200, 350), nn.ReLU())
    model.layer3 = nn.Sequential(nn.Linear(350, 500), nn.ReLU())
    model.layer4 = nn.Sequential(nn.Linear(500, cato))
    # 打印出整个模型
    print(model)

    # 定义 loss 函数，这里用的是交叉熵损失函数(Cross Entropy)，这种损失函数之前博文也讲过的。
    criterion = nn.CrossEntropyLoss()
    # 我们优先使用随机梯度下降，lr是学习率: 0.1
    optimizer = torch.optim.SGD(model.parameters(), 1e-1)

    # 为了实时观测效果，我们每一次迭代完数据后都会，用模型在测试数据上跑一次，看看此时迭代中模型的效果。
    # 用数组保存每一轮迭代中，训练的损失值和精确度，也是为了通过画图展示出来。
    train_losses = []
    train_acces = []
    # 用数组保存每一轮迭代中，在测试数据上测试的损失值和精确度，也是为了通过画图展示出来。
    eval_losses = []
    eval_acces = []
    model = model.double()
    # 创建四个Sequential对象，Sequential是一个时序容器，将里面的小的模型按照序列建立网络

    for e in range(100):
        # 4.1==========================训练模式==========================
        train_loss = 0
        train_acc = 0
        model.train()   # 将模型改为训练模式
        # 每次迭代都是处理一个小批量的数据，batch_size是64
        for im, label in train_data:
            im = Variable(im)
            label = Variable(label)
            # 计算前向传播，并且得到损失函数的值
            out = model(im)
            # print(label)
            loss = criterion(out, label.long())

            # 反向传播，记得要把上一次的梯度清0，反向传播，并且step更新相应的参数。
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录误差
            train_loss += loss.item()

            # 计算分类的准确率
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / im.shape[0]
            train_acc += acc

        train_losses.append(train_loss / len(train_data))
        train_acces.append(train_acc / len(train_data))

        # 在测试集上检验效果
        eval_loss = 0
        eval_acc = 0
        model.eval()  # 将模型改为预测模式

        # 每次迭代都是处理一个小批量的数据，batch_size是128
        for im, label in test_data:
            im = Variable(im)  # torch中训练需要将其封装即Variable，此处封装100特征值
            label = Variable(label)  # 此处为标签
            out = model(im)  # 经网络输出的结果
            loss = criterion(out, label.long())  # 得到误差
            # 记录误差
            eval_loss += loss.item()
            # 记录准确率
            _, pred = out.max(1)  # 得到出现最大值的位置，也就是预测得到的数即0—624
            print(pred)
            num_correct = (pred == label).sum().item()  # 判断是否预测正确
            acc = num_correct / im.shape[0]  # 计算准确率
            eval_acc += acc
        eval_acc = eval_acc * 100*1.0
        train_acc = train_acc * 100*1.0
        eval_losses.append(eval_loss / len(test_data))
        eval_acces.append(eval_acc / len(test_data))
        print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
              .format(e, train_loss / len(train_data), train_acc / len(train_data),
                      eval_loss / len(test_data), eval_acc / len(test_data)))

    end = time.time()
    print('程序运行时间是：{}'.format(end - start))

    print(eval_acces)
    # plt.title('train loss')
    # plt.plot(np.arange(len(train_losses)), train_losses)
    plt.figure(1)
    plt.plot(np.arange(len(train_acces)), train_acces)
    plt.title('train acc')
    plt.show()
    # plt.plot(np.arange(len(eval_losses)), eval_losses)
    # plt.title('test loss')
    plt.figure(2)
    plt.plot(np.arange(len(eval_acces)), eval_acces)
    plt.title('test acc')
    plt.show()

    torch.save(model.state_dict(), "./bp_model")
