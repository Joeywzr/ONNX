import urllib.request
import os, sys
import argparse
import torch
from pytorch_onnx import PytorchConnector


def parse_args():
    parser = argparse.ArgumentParser(description='ONNX Toolset')
    parser.add_argument('--type', '-t', type=str, required=True,
                        help='input \'caffe2\' or \'tensorflow\' or \'pytorch\'')
    parser.add_argument('--net', '-n', required=True, type=str,
                        choices=[
                            'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                            'densenet121', 'densenet161', 'densenet169', 'densenet201',
                            'inceptionv3', 'inceptionv4', 'inceptionresnetv2',
                            'se_resnet50', 'bninception', 'mobilenetv1', 'mobilenetv2'
                        ],
                        help='input model name(default: resnet50)')
    parser.add_argument('--download', '-d', action='store_true',
                        default=False, help='download the model(default: False)')
    parser.add_argument('--d_path', '-dp', type=str, default='', help='download path(default:current path)')
    parser.add_argument('--convert2onnx', '-c2o', action='store_true', default=False, help='convert to onnx(default:False)')
    parser.add_argument('--output', '-o', type=str, default='', help='output path')
    args = parser.parse_args()
    return args


def callbackfunc(blocknum, blocksize, totalsize):
    percent = 100.0 * blocknum * blocksize / totalsize
    if percent > 100:
        percent = 100
    sys.stdout.write('\r>> Downloading %s %.1f%%' %
                     (os.getcwd(), percent))
    sys.stdout.flush()


if __name__ == "__main__":
    pretrained_settings = {
        "vgg16": {
            "url": "https://download.pytorch.org/models/vgg16-397923af.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000
        },
        "vgg19": {
            "url": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000
        },
        "resnet18": {
            "url": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000
        },
        "resnet34": {
            "url": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000
        },
        "resnet50": {
            "url": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000
        },
        "resnet101": {
            "url": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000
        },
        "resnet152": {
            "url": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000
        },
        "densenet121": {
            "url": "https://download.pytorch.org/models/densenet121-a639ec97.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000
        },
        "densenet161": {
            "url": "https://download.pytorch.org/models/densenet161-8d451a50.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000
        },
        "densenet169": {
            "url": "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000
        },
        "densenet201": {
            "url": "https://download.pytorch.org/models/densenet201-c1103571.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000
        },
        "inceptionv3": {
            "url": "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth",
            "input_space": "RGB",
            "input_size": [3, 299, 299],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000
        },
        "inceptionv4": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth",
            "input_space": "RGB",
            "input_size": [3, 299, 299],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000
        },
        "inceptionresnetv2": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth",
            "input_space": "RGB",
            "input_size": [3, 299, 299],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000
        },
        "se_resnet50": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000
        },
        "bninception": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth",
            # "url": "http://yjxiong.me/others/bn_inception-9f5701afb96c8044.pth",
            "input_space": "BGR",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000
        },
        "mobilenetv2": {
            "url": "pretrained/mobilenetv2-0c6065bc.pth",
            "input_space": "BGR",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000
        }
    }
    args = parse_args()

    # print the info of model
    print(pretrained_settings[args.net])

    # if 'download' is True
    if args.download:
        if args.d_path:
            local = args.d_path
        else:
            local = "{}/{}.pth".format(sys.path[0], args.net)
        urllib.request.urlretrieve(pretrained_settings[args.net]['url'], local, callbackfunc)

    # if 'convert2onnx' is True
    if args.convert2onnx:
        connector = PytorchConnector(args)
        connector.translate()