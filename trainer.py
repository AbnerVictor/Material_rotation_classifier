import torch
import torch.utils.data as torch_data
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import cv2
import matplotlib.pyplot as plt
import numpy as np
from my_data import *
from my_clf_model import *

train_dataset = './dataset/train_set_2'
test_dataset = './dataset/test_set_2'


def imshow(img, title):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('GPU available: ' + str(torch.cuda.is_available()))

    # 归一化参数
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=0.5, std=0.5),
                                    transforms.Resize(size=(200, 200))])
    my_batch_size = 5

    train_set = rotation_dataset(train_dataset, transform=transform)
    train_loader = torch_data.DataLoader(train_set, batch_size=my_batch_size, shuffle=True, num_workers=0)

    test_set = rotation_dataset(test_dataset, transform=transform)
    test_loader = torch_data.DataLoader(test_set, batch_size=my_batch_size, shuffle=True, num_workers=0)

    classes = ('0', '90', '180', '270')

    dataiter = iter(train_loader)
    images, labels = dataiter.__next__()

    # # show images
    imshow(torchvision.utils.make_grid(images), ' '.join('%5s' % classes[labels[j]] for j in range(my_batch_size)))

    # 初始化网络
    net = My_Classify_Net()
    print(net)
    net = net.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 初始化权重
    net.apply(weight_reset)
    net.train()  # 启用dropout

    for epoch in range(10):
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader, 0):

            inputs, labels = data  # get inputs and labels from data list

            optimizer.zero_grad()  # zero the parameter gradients

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)  # forward + backward + optimize
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算在训练集上的acc
            # net.eval() #关闭dropout
            outputs = net(inputs)
            predictions = torch.max(func.softmax(outputs.to('cpu'), dim=1), 1)[1]
            # print(torch.max(func.softmax(outputs.to('cpu')), dim=1)[1]+1)
            pred_y = predictions.data.numpy().squeeze()
            target_y = labels.to('cpu').data.numpy()
            accuracy = sum(pred_y == target_y) / len(target_y)
            running_acc += accuracy
            # net.train()

            if i % 48 == 47:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 48))
                print('[%d, %5d] acc on batch: %.3f' % (epoch + 1, i + 1, running_acc / 48))
                running_loss = 0.0
                running_acc = 0.0

    print('Training Finished')

    rand_num = np.random.randint(1, 23)
    images, labels = list(test_loader)[rand_num]

    net.eval()

    images = images.to(device)
    outputs = net(images)
    predictions = torch.max(func.softmax(outputs.to('cpu'), dim=1), 1)[1]

    # show images
    imshow(torchvision.utils.make_grid(images.to('cpu')), '')
    # print labels
    print('labels:', ' '.join('%5s' % classes[labels[j]] for j in range(my_batch_size)))
    print('predis', ' '.join('%5s' % classes[predictions[j]] for j in range(my_batch_size)))

    test_acc = 0

    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = net(inputs)
        predictions = torch.max(func.softmax(outputs.to('cpu'), dim=1), 1)[1]
        pred_y = predictions.data.numpy().squeeze()
        target_y = labels.to('cpu').data.numpy()
        accuracy = sum(pred_y == target_y) / len(target_y)
        test_acc += accuracy

    print(f'average accuracy on test set {len(test_set)} images: {test_acc/(len(test_set)/5)}')