from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from .oaenet import OAENet

print('Pytorch Vision: ', torch.__version__)
print('Torchvision Version: ', torchvision.__version__)


class ResModel(object):
    '''return the resnet model'''

    def __init__(self, model_name, num_classes):
        self.model, self.input_size = self.initialize_model(model_name, num_classes, feature_extract=False,
                                                            use_pretrained=False)
        self.name = model_name

    def set_parameter_requires_grad(self, feature_extracting):
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False

    def initialize_model(self, model_name, num_classes, feature_extract=False, use_pretrained=True):
        model_ft = None
        input_size = 0

        if model_name == "ResNet34":
            """ Resnet34
            """
            model_ft = models.resnet34(pretrained=use_pretrained)
            self.set_parameter_requires_grad(feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            self.set_parameter_requires_grad(feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            self.set_parameter_requires_grad(feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            self.set_parameter_requires_grad(feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 299

        elif model_name == "oaenet":
            model_ft = OAENet(num_classes=num_classes)
        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

    def load_model(self):
        return self.model

    def load_input_size(self):
        return self.input_size
