import torch.nn as nn
from .unet_like import Unet_like
from .mobilenet import MobileNetV2


class OAENet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        out_channel = 32
        main_inv_res_setting = [[1, 16, 1, 1], [6, 24, 2, 2], ]

        self.maintenance_branch = MobileNetV2(inverted_residual_setting=main_inv_res_setting, last_channel=out_channel)
        self.attention_branch = Unet_like(out_channels=out_channel)
        self.backbone = MobileNetV2(input_img=False)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, inputs):
        input, mask = inputs
        x_m = self.maintenance_branch(input)
        if self.training:
            x_a, seg_out = self.attention_branch(mask)
            y_f = x_m * (x_a + 1)
            x = self.backbone(y_f)
            x_c = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
            x = self.classifier(x_c)
            return (x, x_c, seg_out)
        else:
            x_a = self.attention_branch(mask)
            y_f = x_m * (x_a + 1)
            x = self.backbone(y_f)
            x_c = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
            x = self.classifier(x_c)
            return x


class OAENetModel(object):
    '''return the resnet model'''

    def __init__(self, model_name, num_classes):
        self.model, self.input_size = self.initialize_model(model_name, num_classes)
        self.name = model_name

    def initialize_model(self, model_name, num_classes):
        if model_name == "OAENet":
            model = OAENet(num_classes=num_classes)
            input_size = 224
        else:
            print("Invalid model name, exiting...")
            exit()
        return model, input_size
