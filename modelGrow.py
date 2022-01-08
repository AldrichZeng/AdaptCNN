import torch
from getSeed import getSeed
from tools import FreLayer
import copy
from tools import load_data

ranks = {}
old_ranks = {}


def modelGrow():
    model_name = "resnet56"
    dataset_name = "cifar10"
    seed_net, roadmap = getSeed(model_name, dataset_name)
    uv_net = None
    global ranks, old_ranks
    for epoch in range(6):  # 假设考虑6个衍生模型
        if epoch == 0:
            uv_net = seed_net
            info = roadmap["model" + epoch]
            ranks = info["rank"]
            old_ranks = ranks
        else:
            # 获取信息
            info = roadmap["model" + epoch]
            ranks = info["rank"]
            # 为所有层增加rank
            new_net = changeModel(uv_net, ranks)
            # 增量式地重训练
            retrain_growpart(uv_net, new_net, old_ranks)
            old_ranks = ranks


def changeModel(uv_net, ranks):
    new_net = copy.deepcopy(uv_net)
    layer = 0
    for name, mod in new_net.named_modules():
        layer += 1
        if name in ['UVLayer']:
            # 获取原来的矩阵信息
            U_origin = mod.U.data.cpu().numpy()
            V_origin = mod.V.data.cpu().numpy()
            rank_origin = U_origin.shape[1]
            m = U_origin.shape[0]
            n = V_origin.shape[1]
            # 增加一些行/列
            U_new = torch.rand(m, ranks[layer]).data.cpu().numpy()
            V_new = torch.rand(ranks[layer], n).data.cpu().numpy()
            U_new[:, 0:rank_origin] = U_origin[:, 0:rank_origin]
            V_new[0:rank_origin, :] = V_origin[0:rank_origin, 0]
            # 修改mod信息（结构变了）
            mod.U = torch.from_numpy(U_new)
            mod.V = torch.from_numpy(V_new)
    return new_net


def retrain_growpart(old_net, uv_net, old_ranks):
    """
    模型增长
    :param uv_net:
    :param ranks: 用于规定每层哪些部分不需要retrain
    :return:
    """
    args = {
        'learning_rate': 0.01,
        'momuntum': 0.0009,
        'weight_decay': 0.001
    }
    retrain(uv_net, args)

    # 注意，我们不是去更改register_full_backward_hook(hook)中的hook的grad_input，因为这只是针对input的梯度。
    # 我们要保证的是，U、V矩阵像是被mask了一样，部分参数不训练。这使得我们必须在retrain后，用mask把U、V的一部分还原成原来net的样子。

    for name, mod in uv_net.named_modules():
        U_old = None
        V_old = None
        old_rank = old_ranks[name]
        for old_name, old_mod in old_net.named_modules():
            if old_name == name:
                U_old = old_mod.U.data.cpu().numy()
                V_old = old_mod.V.data.cpu().numpy()
                break
        U = mod.U.data.cpu().numpy()
        V = mod.V.data.cpu().numpy()
        if U_old is None or V_old is None:
            exit(0)
        if U_old.shape[1] != old_rank or V_old.shape[0] != old_rank:
            exit(0)
        U[:, 0:old_rank] = U_old[:, 0:old_rank]
        V[0:old_rank, :] = V_old[0:old_rank, :]
        # 恢复
        mod.U = torch.from_numpy(U)
        mod.V = torch.from_numpy(V)


def retrain(uv_net, args):
    """
    重训练
    :param uv_net:
    :param args:
    :return:
    """
    learning_rate = args['learning_rate']
    momentum = args['momuntum']
    weight_decay = args['weight_decay']
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(uv_net.parameters(),  # vgg和resnet在某些设定上略有不同
                                lr=learning_rate,  # 学习率
                                momentum=momentum, weight_decay=weight_decay)

    if args.dataset == "cifar10":
        trainLoader, testLoader = load_data.load_cifar10_for_vgg()
    elif args.dataset == "cifar100":
        trainLoader, testLoader = load_data.load_cifar100_for_vgg()

    for batch_step, data in enumerate(trainLoader):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        # 前向传播
        output = uv_net(input)
        # 反向传播
        optimizer.zero_grad()
        loss = criterion(output, labels)
        loss.backward()
        # 更新
        optimizer.step()
