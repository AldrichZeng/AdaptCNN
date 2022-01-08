# This file is to specify the path in host.

Dataset = {}
Dataset[
    'CIFAR10'] = "/home/xxx..../CIFAR10/ (Please delete xxx.... and specify the path)"  # todo:please specify the path
Dataset[
    'CIFAR100'] = "/home/xxx..../CIFAR100/ (Please delete xxx.... and specify the path)"  # todo:please specify the path
Dataset[
    'ImageNet'] = "/home/xxx..../imagenet/ (Please delete xxx.... and specify the path)"  # todo:please specify the path

Path = {}
Path['root'] = "/home/xxx..../ (Please delete xxx.... and specify the path)"  # todo:please specify the path
Path['tensorboard'] = Path['root'] + "tensorboard/"  # todo:please make this directory in your host.
Path['dataset'] = Path['root'] + "data/"  # todo:please make this directory in your host.
Path['trainedModels'] = Path['root'] + "models_save/"  # todo:please make this directory in your host.
Path['model_baseline'] = Path['root'] + "Baseline/"  # todo:please make this directory in your host.
Path['log'] = Path['root'] + "log/"  # todo:please make this directory in your host.
