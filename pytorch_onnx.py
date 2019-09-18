import torch
import sys
import torch.utils.model_zoo as model_zoo
import torchvision
from models.senet import se_resnet50
from models.inceptionresnetv2 import inceptionresnetv2
from models.mobilenetv2 import mobilenetv2
from models.mobilenetv1 import mobilenetv1
from torch.autograd import Variable
from models.inceptionv3 import inception_v3
from models.inceptionv4 import inceptionv4, InceptionV4
from models.bninception import bninception, BNInception
from torchvision.models.densenet import densenet121, densenet169, densenet201, densenet161
from torchvision.models.vgg import vgg16, vgg19
from torchvision.models.resnet import resnet101, resnet152, resnet18, resnet34, resnet50


__NET_OK__ = ['vgg16', 'vgg19',
              'inceptionv3', 'inceptionv4', 'resnet18', 'resnet34', 'resnet50',
              'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169',
              'densenet201', 'inceptionresnetv2', 'mobilenetv2',
              'se_resnet50', 'bninception']


class PytorchConnector:
    def __init__(self, args):
        self.net = args.net
        self.output = args.output
        if args.output:
            self.output = args.output
        else:
            self.output = "{}/{}.onnx".format(sys.path[0], self.net)

    def translate(self):
        if self.net == 'inceptionv3' or self.net == 'inceptionv4' or self.net == 'inceptionresnetv2':
            x = torch.randn(1, 3, 299, 299)
            if self.net == 'inceptionv3':
                model = inception_v3(pretrained=True)
                # model = Inception3()
                # m = torch.load('/home/memo/.torch/models/inception_v3_google-1a9a5a14.pth')
                # model.load_state_dict(m)
                # model.train(False)
                # print(model)
            else:
                model = globals().get(self.net)(pretrained='imagenet')

        elif self.net == 'bninception' or self.net == 'se_resnet50':
            x = Variable(torch.randn(1, 3, 224, 224))
            model = globals().get(self.net)(pretrained='imagenet')
        elif self.net == 'mobilenetv2' :
            x = Variable(torch.randn(1, 3, 224, 224))
            model = mobilenetv2()
            model.load_state_dict(torch.load('pretrained/mobilenetv2-0c6065bc.pth', map_location=torch.device('cpu')))
        elif self.net == 'mobilenetv1':
            print('not available yet')
            # x = Variable(torch.randn(1, 3, 224, 224))
            # model = mobilenetv1()
            # model.load_state_dict(torch.load('pretrained/mobilenetv2-0c6065bc.pth'))
        else:
            x = Variable(torch.randn(1, 3, 224, 224))
            model = globals().get(self.net)(pretrained=True)


        torch.onnx.export(model, x, self.output, verbose=True)