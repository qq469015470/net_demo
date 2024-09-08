import torch
import torchvision
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from torch.utils.data import DataLoader
from torchvision import transforms
import PIL
import sys
import os

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #定义卷积层
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),#1个channel如只有灰度的图片就是1个channcel,rgb就是3个channel,10个输出卷积,卷积核大小为5
            torch.nn.ReLU(),#卷积后执行激活函数
            torch.nn.MaxPool2d(kernel_size=2),#2x2最大池化层,在w*h的张量上滑动,步长默认为kernel_size*kernel_size,在2x2中取最大的值,所以变换后的大小为w/2*h/2
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),#卷积10个channel输入,20个channel输出,大小为5*5
            torch.nn.ReLU(),#激活函数
            torch.nn.MaxPool2d(kernel_size=2),#2x2最大池化层
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),#全连接层, 相当于用 1x320矩阵与320x50的矩阵相乘最后得出1x50的矩阵(矩阵相乘取两边)
            torch.nn.Linear(50, 10),#全连接层 1x50与50x10矩阵相乘 得出1x10矩阵
        )

    def forward(self, x):#输入一个(batch_size,1,28,28)的张量, 即1个灰度值,28*28的图片
        batch_size = x.size(0)#批次大小
        x = self.conv1(x)#与第一个卷积层运算,得出(batch_size,10,12,12)张量
        x = self.conv2(x)#与第二个卷积层运算,得出(batch_size,20,4,4)张量
        x = x.view(batch_size, -1)#展开张量20*4*4=320,得出(batch_size, 320)张量
        x = self.fc(x)#进入全连接层得出(batch_size,1,10)张量
        #print('batch_size:{}, forward:{}'.format(batch_size, x.size()))
        return x

path = './data'#训练数据保存及读取路径

batch_size = 64#批数,同时训练64批
learning_rate = 0.01#梯度下降的学习率
momentum = 0.5#梯度下降惯性
EPOCH = 10#全部训练集的次数, 全部训练集循环训练10次

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean = [0.5], std = [0.5])])

train_dataset = torchvision.datasets.MNIST(path, train=True, transform=transform, download=True);#拉取MNIST的训练集
test_dataset = torchvision.datasets.MNIST(path, train=False, transform=transform, download=True);#拉佢MNIST的测试集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#按批次划分数据
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)#按皮批次划分数据

model = Net()
if os.path.exists('checkpoint.pth'):
	model.load_state_dict(torch.load('checkpoint.pth')['model']);#读取权重

criterion = torch.nn.CrossEntropyLoss()#定义损失函数,如何去算出梯度
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)#优化器,如何去更新梯度

def train(epoch):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data#inputs为样本的张量, target为样本的正确结果
        optimizer.zero_grad()#清空上次更新的梯度值

        outputs = model(inputs)#进入网络
        loss = criterion(outputs, target)#进入损失函数,判断损失率
        
        loss.backward()#计算梯度
        optimizer.step()#更新梯度

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()
        
        if batch_idx % 300 == 299:
            print('[%d %5d]: loss: %.3f, acc: %.2f %%'
                    % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))

            running_loss = 0.0
            running_total = 0
            running_correct = 0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('[%d / %d]: Accuracy on test set: %.1f %% ' % (epoch+1, EPOCH, 100 * acc))
    return acc
        

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('输入"tran"参数训练,输入"test 文件名测试"')
        exit(0)	

    if sys.argv[1] == 'train':
        acc_list_test = []
        for epoch in range(EPOCH):
            train(epoch)

            #显示每次训练的预测成功率折线图
            #acc_test = test()
            #acc_list_test.append(acc_test)

            #plt.plot(acc_list_test)
            #plt.xlabel('Epoch')
            #plt.ylabel('Accuracy On TestSet')
            #plt.show()

        torch.save({'model': model.state_dict()}, 'checkpoint.pth');#保存训练好的权重
    elif sys.argv[1] == 'test':
        if len(sys.argv) != 3:
            print('请输入文件名')
            exit(0)

        img1 = mpimg.imread(sys.argv[2])#读取图片,类型为数组'ndarray',3x28x28的张量 3为rgb3个channel

        #转换为1channel灰度图,1x28x28
        input_transform = transforms.Compose([
           transforms.Grayscale(1), #这一句就是转为单通道灰度图像
           transforms.ToTensor(),
           transforms.Normalize(mean = [0.5], std = [0.5]),
        ])
        gray_image = input_transform(PIL.Image.fromarray(img1, mode='RGB'))

        gray_image = gray_image.unsqueeze(0)#升维

        outputs = model(gray_image)#进入神经网络
        _, predicted = torch.max(outputs, dim=1)#取出模型预测的10个张量的最大值的索引,刚好索引相当于所要预测的数字
        
        #显示输入的图片,及预测的值
        fig = plt.figure()
        plt.subplot(1, 1, 1)
        plt.tight_layout()
        plt.imshow(img1, cmap='gray', interpolation='none')
        plt.title("Labels: {}".format(predicted[0]))
        plt.xticks([])
        plt.yticks([])
        plt.show()
    else:
        print('参数不正确')
        exit(0)
