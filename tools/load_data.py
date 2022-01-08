import torchvision.transforms as transforms
import torchvision as tv
import torch
from tools import myconfig


def load_cifar10_for_vgg():
    return load_dataset(dataset_name="cifar10", batch_size=64)


def load_cifar100_for_vgg():
    return load_dataset(dataset_name="cifar100", batch_size=64)


# resnet与vgg所使用的batch_size不同，故定义不同的函数。
def load_cifar10_for_resnet():
    return load_dataset(dataset_name="cifar10", batch_size=128)


def load_cifar100_for_resnet():
    return load_dataset(dataset_name="cifar100", batch_size=128)


def load_imagenet(batch_size=256):
    return load_dataset(dataset_name="imagenet", batch_size=batch_size)


def load_dataset(dataset_name, batch_size):
    if dataset_name == "cifar10":
        # mean = [0.485, 0.456, 0.406]  # 这里用的期望和标准差是来自方毓楚的config
        # std = [0.229, 0.224, 0.225]
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]  # 这里用的期望和标准差是来自SFP的代码
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        transform1 = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 以一定的概率翻转图像,默认0.5
            transforms.RandomCrop(32, 4),  # 随机裁剪图像为不同大小(0.08~1.0)和宽高比(3/4~4/3)
            transforms.ToTensor(),  # 转化为Tensor类型
            transforms.Normalize(mean, std),
            # 标准化，数据来自https://github.com/Ksuryateja/pytorch-cifar10/blob/master/cifar10.py
        ])
        transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        trainset = tv.datasets.CIFAR10(root=myconfig.Dataset['CIFAR10'],
                                       train=True,
                                       transform=transform1,
                                       download=True)
        testset = tv.datasets.CIFAR10(root=myconfig.Dataset['CIFAR10'],
                                      train=False,
                                      transform=transform2,
                                      download=True)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=8)  # 定义训练批处理数据
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=8)  # 定义测试批处理数据
        return trainloader, testloader
    elif dataset_name == "cifar100":
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]  # 这里用的期望和标准差是来自SFP的代码
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        transform1 = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 以一定的概率翻转图像,默认0.5
            transforms.RandomCrop(32, 4),  # 随机裁剪图像为不同大小(0.08~1.0)和宽高比(3/4~4/3)
            transforms.ToTensor(),  # 转化为Tensor类型
            transforms.Normalize(mean, std),
        ])
        transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        trainset = tv.datasets.CIFAR100(root=myconfig.Dataset['CIFAR100'],
                                        train=True,
                                        transform=transform1,
                                        download=True)
        testset = tv.datasets.CIFAR100(root=myconfig.Dataset['CIFAR100'],
                                       train=False,
                                       transform=transform2,
                                       download=True)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=8)  # 定义训练批处理数据
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=8)  # 定义测试批处理数据
        return trainloader, testloader
    elif dataset_name == "imagenet":
        # Data loading code (refers to SFP's code)
        traindir = myconfig.Dataset['ImageNet'] + "train"
        valdir = myconfig.Dataset['ImageNet'] + "validation"
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = tv.datasets.ImageFolder(traindir,
                                                transforms.Compose([transforms.RandomResizedCrop(224),
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.ToTensor(),
                                                                    normalize, ]))

        trainloader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=8,
                                                  pin_memory=True,
                                                  sampler=None)

        testloader = torch.utils.data.DataLoader(tv.datasets.ImageFolder(valdir,
                                                                         transforms.Compose([transforms.Resize(256),
                                                                                             transforms.CenterCrop(224),
                                                                                             transforms.ToTensor(),
                                                                                             normalize, ])),
                                                 batch_size=batch_size, shuffle=False,
                                                 num_workers=0, pin_memory=True)
        return trainloader, testloader
    else:
        return


# if __name__ == "__main__":
#     load_cifar100_for_vgg()
