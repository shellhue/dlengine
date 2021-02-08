
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torchvision import models
import os
from vit_pytorch import ViT


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def load_weights(model, weights):
    if isinstance(weights, str) and os.path.isfile(weights):
        weights = torch.load(weights, map_location="cpu")
        if "model" in weights:
            weights = weights["model"]
        model.load_state_dict(weights)
    return model


def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True, weights=None, device="cpu"):
    model_ft = None
    input_size = 224

    if model_name == "resnet18":
        model_ft = models.resnet18(pretrained=use_pretrained, progress=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained, progress=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "resnest50":
        model_ft = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "mobilenetv2":
        model_ft = models.mobilenet_v2(pretrained=use_pretrained, progress=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "vit":
        v = ViT(
            image_size=input_size,
            patch_size=32,
            num_classes=num_classes,
            dim=int(1024 * 1.0),
            depth=6,
            heads=16,
            mlp_dim=int(2048 * 1),
            dropout=0.1,
            emb_dropout=0.1
        )
        v = v.to(device)
        return v, input_size
    else:
        print("Invalid model name, exiting...")
        exit()

    model_ft = load_weights(model_ft, weights)
    model_ft = model_ft.to(device)
    return model_ft, input_size


class Classifier(nn.Module):
    def __init__(self,
                 model_name="resnet50",
                 num_classes=1000,
                 device="cuda",
                 feature_extract=False,
                 use_pretrained=False,
                 loss_weights=None):
        super().__init__()
        model, input_size = initialize_model(model_name, num_classes, feature_extract,
                                             use_pretrained=use_pretrained, device=device)
        self.backbone = model
        self._input_size = input_size
        self._num_classes = num_classes
        self._device = device
        self.ce_loss = nn.CrossEntropyLoss(weight=loss_weights)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        if self.training:
            # training
            imgs, labels = inputs
            imgs = imgs.to(self._device)
            labels = labels.to(self._device)
            outputs = self.backbone(imgs)
            ce_loss = self.ce_loss(outputs, labels)
            return {
                "cross_entropy_loss": ce_loss
            }
        elif isinstance(inputs, tuple) or isinstance(inputs, list):
            # testing
            imgs, _ = inputs
            imgs = imgs.to(self._device)
            outputs = self.backbone(imgs)
            return self.softmax(outputs)
        elif isinstance(inputs, torch.Tensor):
            # inference
            imgs = inputs
            imgs = imgs.to(self._device)
            outputs = self.backbone(imgs)
            return outputs
        else:
            imgs, _ = inputs
            print(type(inputs), len(inputs))
            print(imgs.shape)
            assert False
