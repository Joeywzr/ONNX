import torch
import sys
import torch.utils.model_zoo as model_zoo
import torchvision
from models.senet import se_resnet50
from models.inceptionresnetv2 import inceptionresnetv2
from torch.autograd import Variable
from torchvision.models.inception import Inception3
from models.inceptionv4 import inceptionv4
from models.bninception import bninception
from torchvision.models.densenet import densenet121, densenet169, densenet201, densenet161
from torchvision.models.vgg import vgg16, vgg19
from torchvision.models.resnet import resnet101, resnet152, resnet18, resnet34, resnet50

__NET_OK__ = ['vgg16', 'vgg19',
              'inceptionv3', 'inceptionv4', 'resnet18', 'resnet34', 'resnet50',
              'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169',
              'densenet201', 'inceptionresnetv2', 'mobilenetv1', 'mobilenetv2',
              'se_resnet50', 'bninception']


class PytorchConnector:
    def __init__(self, args):
        self.net = args.net
        self.output = args.output
        if args.output:
            self.output = args.output
        else:
            self.output = "{}/{}/{}.onnx".format(sys.path[0], self.net, self.net)

    def translate(self):
        if self.net == 'inceptionv3' or self.net == 'inceptionv4' or self.net == 'inceptionresnetv2':
            x = Variable(torch.randn(1, 3, 299, 299))
            if self.net == 'inceptionv3':
                model = Inception3()
            else:
                model = globals().get(self.net)()
            # TODO: mobilenetv1/2

        else:
            x = Variable(torch.randn(1, 3, 224, 224))
            model = globals().get(self.net)()


        torch.onnx.export(model, x, self.output, verbose=True)