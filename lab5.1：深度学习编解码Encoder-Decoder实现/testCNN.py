import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      # 数据库模块
import matplotlib.pyplot as plt

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 3           # 训练整批(所有)数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 50
LR = 0.001          # 学习率
DOWNLOAD_MNIST = False #True  # 如果你已经下载好了mnist数据就写上 False


# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root='./data/',    # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
                                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=True,          # 没下载就下载, 下载了就不用再下了
)

test_data = torchvision.datasets.MNIST(root='./data/', train=False)

# plot one example




# 批训练 50 samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

##是对数据维度进行扩充,train_data加载时会自动扩维 为了节约时间, 我们测试时只测试前2000个
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000] #shape (2000,)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.encoder = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=16,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (16, 28, 28)
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32 * 7 * 7, 784),  # fully connected layer,
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(784,10)    #output 10 classes
        )
    def forward(self, x):
        x = self.encoder(x)                    #(batch_size, 32 , 7 , 7)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.decoder(x)
        return output     #(batch_size,10)

cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

# # training and testing
# #training
# for epoch in range(EPOCH):
#     for step, (b_x, b_y) in enumerate(train_loader):   # 分配 batch data, normalize x when iterate train_loader
#         output = cnn(b_x)               # cnn output
#         loss = loss_func(output, b_y)   # cross entropy loss
#         optimizer.zero_grad()           # clear gradients for this training step
#         loss.backward()                 # backpropagation, compute gradients
#         optimizer.step()                # apply gradients
#
#         #testing
#         if step%50 ==0:
#             test_output = cnn(test_x)
#             pred_y = torch.max(test_output, 1)[1].data.numpy() #shape(2000,10) 只返回最大值的每个索引
#             accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
#             print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
#
# torch.save(cnn,'cnn_minist.pkl')
# print('finish training')


#resume test
print('load cnn model')
cnn1 = torch.load('cnn_minist.pkl')

test_output = cnn1(test_x[70:90])
pred_y = torch.max(test_output, 1)[1].data.numpy()
accuracy = float((pred_y == test_y[70:90].data.numpy()).astype(int).sum()) / float(test_y[70:90].size(0))
print(pred_y, 'prediction number')
print(test_y[70:90].numpy(), 'real number')
print('accuracy',accuracy)

#total 2000 test set
total_test_output = cnn1(test_x)
total_pred_y = torch.max(total_test_output, 1)[1].data.numpy()
total_accuracy = float((total_pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
print('total accuracy',total_accuracy)

