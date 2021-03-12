import torch
import torch.utils.data as torch_data
import torchvision
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import enum

train_dataset = './dataset/train_set_1'
test_dataset = './dataset/test_set_1'

angle_label = {
    '0': 0,
    '90': 1,
    '180': 2,
    '270': 3
}


def load_data(dataset, cata):
    label = angle_label[cata]
    data_list = []
    label_list = []

    for img_path in os.listdir(os.path.join(dataset, cata)):
        img = cv2.imread(os.path.join(os.path.join(dataset, cata), img_path), flags=cv2.IMREAD_GRAYSCALE)
        data_list.append(img)
        label_list.append(label)

    return data_list, label_list


def data_loader(dataset):
    # 加载训练数据
    data_0, label_0 = load_data(dataset, '0')
    data_90, label_90 = load_data(dataset, '90')
    data_180, label_180 = load_data(dataset, '180')
    data_270, label_270 = load_data(dataset, '270')

    data_0.extend(data_90)
    data_0.extend(data_180)
    data_0.extend(data_270)

    label_0.extend(label_90)
    label_0.extend(label_180)
    label_0.extend(label_270)

    print(f'from {dataset} loaded image shape: {data_0[0].shape}, loaded data count: {len(data_0)}')
    plt.imshow(data_0[0], cmap='gray')
    plt.show()

    return data_0, label_0


class rotation_dataset(torch.utils.data.Dataset):
    data = []
    label = []

    def __init__(self, path, transform=None, target_transform=None):
        super(rotation_dataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.data, self.label = data_loader(path)

    def __getitem__(self, index):
        data = self.transform(self.data[index])
        return data, self.label[index]

    def __len__(self):
        return len(self.data)


def imshow(img, title):
    img = img / 2 + 0.5     # unnormalize
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
                                    transforms.Resize(size=(100, 100))])
    my_batch_size = 5

    train_set = rotation_dataset(train_dataset, transform=transform)
    train_loader = torch_data.DataLoader(train_set, batch_size=my_batch_size, shuffle=True, num_workers=0)

    test_set = rotation_dataset(test_dataset, transform=transform)
    test_loader = torch_data.DataLoader(test_set, batch_size=my_batch_size, shuffle=False, num_workers=0)

    classes = ('0', '90', '180', '270')

    dataiter = iter(train_loader)
    images, labels = dataiter.__next__()

    # # show images
    imshow(torchvision.utils.make_grid(images), ' '.join('%5s' % classes[labels[j]] for j in range(my_batch_size)))
    
