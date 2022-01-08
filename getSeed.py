# 我们要获得一个seed，即种子网络
import torch
import pickle  # 用于载入文件中的数据

root = "/home/zengyao/model/"


def getSeed(model_name, dataset_name):
    """
    :param model_name:
    :param dataset_name:
    :return:
    """
    # step1:get the seed model
    seed_net = torch.load(root + "_" + model_name + "_" + dataset_name)
    # step2: get the road map
    file = open(root + "roadmap/" + model_name + "_" + dataset_name)
    roadmap = pickle.load(file)
    file.close()

    '''
    roadmap形如下（类似Jason格式）：
    {
        model1:{
            rank: [rank_of_layer1, rank_of_layer2, ...] # type=List<int>
            acc: # type=float
            size: 内存(MB) # type=float
            FLOPs: # type = float
            runtime: 运行时内存(MB) # type=float
        }
        model2:{
            ...
        }
    }
    '''
    return seed_net, roadmap


