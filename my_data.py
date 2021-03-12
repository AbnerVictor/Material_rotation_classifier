import torch
import torch.utils.data as torch_data
import cv2
import os
import matplotlib.pyplot as plt

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

