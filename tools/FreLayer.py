import torch
import torch.nn as nn
import numpy as np
import copy

'''
层的定义与构建
'''


class FreLayer(nn.Module):
    # 频率域网络的定义
    def __init__(self, weight_matrix, kernel_size, in_channels, padding, rank=0, stride=(1, 1)):
        super(FreLayer, self).__init__()
        self.weight_matrix = torch.nn.Parameter(weight_matrix)  # 使得可对参数求梯度
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.padding = padding
        self.stride = stride
        if rank == 0:
            self.rank = min(weight_matrix.shape[0], weight_matrix.shape[1])
        else:
            self.rank = rank

    def forward(self, x):
        out_h = int((x.size()[2] + 2 * self.padding[0] - self.kernel_size[0]) //
                    self.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * self.padding[1] - self.kernel_size[1]) //
                    self.stride[1] + 1)
        Inp_unf = torch.nn.functional.unfold(input=x, kernel_size=self.kernel_size, padding=self.padding,
                                             stride=self.stride)  # type=torch.Tensor，展开为矩阵
        _, A = get_DCTbase(transformation_size=self.kernel_size, c_in=self.in_channels)  # 获取基函数
        X = A.matmul(Inp_unf)  # X 将要被转置
        out_unf = matmul_frequency(X, self.weight_matrix)  # 矩阵乘法
        O = output_fold(out_unf, (out_h, out_w))  # 恢复为高维张量
        return O

    def extra_repr(self):  # 这是torch.nn.Module父类的方法
        """
        为了打印自定义的额外信息,需要重写该方法.使用__str__(self)是无效的.
        """
        return str(self.weight_matrix.shape[0]) + \
               ",out=" + str(self.weight_matrix.shape[1]) + \
               ",kernel=" + str(self.kernel_size) + \
               ",in=" + str(self.in_channels) + \
               ",padding=" + str(self.padding) + \
               ",stride=" + str(self.stride)


class MatrixLayer(nn.Module):
    # 频率域网络的定义
    def __init__(self, weight_matrix, kernel_size, in_channels, padding, rank=0, stride=(1, 1)):
        super(MatrixLayer, self).__init__()
        self.weight_matrix = torch.nn.Parameter(weight_matrix)  # 使得可对参数求梯度
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.padding = padding
        self.stride = stride
        if rank == 0:
            self.rank = min(weight_matrix.shape[0], weight_matrix.shape[1])
        else:
            self.rank = rank

    def forward(self, x):
        out_h = int((x.size()[2] + 2 * self.padding[0] - self.kernel_size[0]) //
                    self.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * self.padding[1] - self.kernel_size[1]) //
                    self.stride[1] + 1)
        Inp_unf = torch.nn.functional.unfold(input=x, kernel_size=self.kernel_size, padding=self.padding,
                                             stride=self.stride)  # type=torch.Tensor，展开为矩阵
        Inp_unf = Inp_unf.transpose(1, 2)  # Inp_unf.shape = (batch_size, out_h * out_w, c_in * d * d)
        Out_unf = Inp_unf.matmul(self.weight_matrix)  # 矩阵乘法
        Out_unf = Out_unf.transpose(1, 2)  # Out_unf.shape = (batch_size, c_out, out_h * out_w )
        Out = torch.nn.functional.fold(input=Out_unf, output_size=(out_h, out_w), kernel_size=(1, 1))
        return Out

    def extra_repr(self):
        return str(self.weight_matrix.shape[0]) + \
               ",out=" + str(self.weight_matrix.shape[1]) + \
               ",kernel=" + str(self.kernel_size) + \
               ",in=" + str(self.in_channels) + \
               ",padding=" + str(self.padding) + \
               ",stride=" + str(self.stride)


# =============================================================================
# 工具函数
base_dict = {}


def get_DCTbase(transformation_size, c_in):
    """
    获取DCT变换的基
    :param transformation_size:
    :param c_in:
    :return:
    """
    # if base_dict.__contains__(c_in):
    #     return None, base_dict[c_in]
    x, y = transformation_size  # type=tuple
    if x != y:
        print("cannot using DCT!")
        return None
    else:
        d = x
        C = torch.zeros((d, d))  # 矩阵C是用于做T=BFB^T
        for i in range(d):
            for j in range(d):
                C[i, j] = np.sqrt(2 / d) * np.cos((i * np.pi * (2 * j + 1)) / (2 * d))
        C[0, :] = C[0, :] / np.sqrt(2)
        Base = np.kron(C, C)  # C与自身做kronecker积,Base.shape=(d*d,d*d)

        E = np.identity(c_in)  # 单位矩阵，shape=(c_in, c_in)
        A = np.kron(E, Base)  # 空间矩阵可经过A直接变换得到频率域矩阵,A.shape=(d*d*c_in,c_in)
        Base, A = torch.from_numpy(Base), torch.from_numpy(A)
        Base = Base.float()
        A = A.float()
        if torch.cuda.is_available() and A.is_cuda == False and Base.is_cuda == False:
            A = A.cuda()
            Base = Base.cuda()
        else:
            print("Error: No cuda is availalbe.")
        # base_dict[c_in] = A
        return Base, A


def matmul_frequency(X, Y):
    """
    频率域中的矩阵乘法
    :param X: shape=(batch_size, out_h * out_w, c_in*d*d), type=numpy.ndarray
    :param Y: shape=(c_in*d*d, c_out), type=numpy.ndarray
    :return: shape=(batch_size, c_out, out_h * out_w )
    """
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X)
    if not isinstance(Y, torch.Tensor):
        Y = torch.from_numpy(Y)
    X = X.transpose(1, 2)
    Out_unf = X.matmul(Y)  # shape=(batch_size, out_h * out_w, c_out)
    Out_unf.transpose(1, 2)  # shape=(batch_size, c_out, out_h * out_w )
    return Out_unf


def output_fold(Out_unf, output_size):
    """
    将输出矩阵O经过fold后得到高维tensor（空间），以便后续传播
    :param Out_unf: shape=(batch_size, out_h * out_w, c_out)，不符合官方定义，但符合论文的定义
    :param output_size: tuple, (out_h, out_w)
    :return:
    """
    if isinstance(output_size, tuple):
        Out_unf = Out_unf.transpose(1, 2)
        Out = torch.nn.functional.fold(input=Out_unf, output_size=output_size, kernel_size=(1, 1))
        # 指定输出大小即可，kernel_size指定为(1,1)，其余使用默认值（int自动转为tuple）。
        return Out
    else:
        print("Error: got wrong type for parameters. Tuple is needed.")
        return None


# ================================================================================================
# 构建层


def mod_to_frequency(mod):
    """
    将Conv2d转换为频率域矩阵:根据卷积层mod（torch.nn.Module）的权重，得到频率域矩阵
    :return:Y，频率域矩阵
    """
    spatial_weight = mod.weight.data  # type=torch.Tensor
    c_out, c_in, h, w = spatial_weight.shape
    Base, A = get_DCTbase(transformation_size=(h, w), c_in=c_in)
    W = spatial_weight.view(spatial_weight.size(0), -1).t()  # W.shape = (c_in * d * d, c_out)

    Y = A.matmul(W)  # Y.shape=W.shape
    return Y


def make_layer_vgg(spatial_net):
    """
    将Conv2d转换为FreLayer(废弃)
    :param new_net:
    :return:
    """
    index = 0
    layers = []
    new_net = copy.deepcopy(spatial_net)
    for mod in new_net.features:
        if isinstance(mod, nn.Conv2d):
            index += 1
            print("transfer layer ", index, " into Frequency Layer (DCT domain) ===> ")
            weight_matrix = mod_to_frequency(mod).cuda()
            layer = FreLayer(weight_matrix=weight_matrix, kernel_size=mod.kernel_size,
                             in_channels=mod.in_channels, padding=mod.padding, stride=mod.stride)
            layers += [layer]
            del layer
        else:  # 其他层直接拷贝
            layer = copy.deepcopy(mod)  # 优化，使用deepcopy
            layers += [layer]
    new_net.features = torch.nn.Sequential(*layers)
    return new_net


# ==========================================================================
# For ResNet withour downsample (旁路无Conv2d的ResNet，即仅含有BasicBlock)
from utils import get_module


def make_layer_resnet(resnet_spatial):
    """
    适用于vgg和resnet等所有网络结构（通用）
    :param resnet_spatial:
    :return:
    """
    new_net = copy.deepcopy(resnet_spatial)
    for name, mod in new_net.named_modules():  # 该方法可以递归遍历所有的子结构
        if isinstance(mod, nn.Conv2d):
            if mod.kernel_size[0] > 1 and mod.kernel_size[1] == mod.kernel_size[0]:
                print("transfer Conv2d to Frequency Layer (DCT domain) ===> ", name)
                # 将每一个conv2d改成FreLayer
                # print("transfer layer ", name, "(index =", index, ")into DCT trainable.")
                weight_matrix = mod_to_frequency(mod).cuda()
                frelayer = FreLayer(weight_matrix=weight_matrix, kernel_size=mod.kernel_size,
                                    in_channels=mod.in_channels, padding=mod.padding, stride=mod.stride)
                # 替换mod为frelayer
                _modules = get_module(model=new_net, name=name)  # 获取OrderedDict()
                _modules[name.split('.')[-1]] = frelayer  # 如此可以改变new_net中的值
            else:
                print("transfer Conv2d to MatrixLayer ===> ", name)
                spatial_weight = mod.weight.data  # type=torch.Tensor
                weight_matrix = spatial_weight.view(spatial_weight.size(0), -1).t()  # W.shape = (c_in * d * d, c_out)
                matrixLayer = MatrixLayer(weight_matrix=weight_matrix, kernel_size=mod.kernel_size,
                                          in_channels=mod.in_channels, padding=mod.padding, stride=mod.stride)
                # 替换mod为matrixLayer
                _modules = get_module(model=new_net, name=name)  # 获取OrderedDict()
                _modules[name.split('.')[-1]] = matrixLayer  # 如此可以改变new_net中的值

    return new_net


# if __name__ == "__main__":
#     # 测试这里的函数
#     print("===> Test freq_operation/FreLayer.py <===")
#     import load_net_imagenet
#
#     net = load_net_imagenet.load_resnet50()
#     net = make_layer_resnet(net)
#     print(net)
#     from decomposition import make_layer
