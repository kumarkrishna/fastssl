import torch.nn as nn
from torchvision.models.resnet import resnet50
import numpy as np
import torch
from copy import deepcopy
from fastssl.models.resnets import ResNet18, ResNet50

class BackBone(nn.Module):
    def __init__(self,
                 name='resnet50feat',
                 dataset='cifar10',
                 projector_dim=128,
                 hidden_dim=128):
        super(BackBone, self).__init__()
        self.name = name
        self.build_backbone(dataset=dataset, 
                            projector_dim=projector_dim, 
                            hidden_dim=hidden_dim)

    def build_backbone(self, dataset='cifar10', projector_dim=128, hidden_dim=512):
        """
        Build backbone model.
        """
        if 'resnet50' in self.name:
            base_width = 64
            if len(self.name.split('_width'))>1:
                base_width = int(self.name.split('_width')[-1])
            self._resnet50mod(base_width, dataset)
            # self.feat_dim = 2048
            self.feat_dim = 32*base_width
        elif 'resnet18' in self.name:
            base_width = 64
            if len(self.name.split('_width'))>1:
                base_width = int(self.name.split('_width')[-1])
            self._resnet18mod(base_width, dataset)
            self.feat_dim = 8*base_width
        else:
            num_layers = int(self.name.split('_')[-1]) if len(self.name.split('_'))>1 else 2
            self.name = self.name.split('_')[0] if len(self.name.split('_'))>1 else self.name
            if 'dualstream' in self.name:
                self._shallowConvDualmod(dataset, layers=num_layers)
            else:
                self._shallowConvmod(dataset,layers=num_layers)
            self.feat_dim = 2048
        if 'proj' in self.name:
            self.build_projector(projector_dim=projector_dim, hidden_dim=hidden_dim)
        if 'pred' in self.name:
            self.build_predictor(projector_dim=projector_dim)

    def _resnet50mod(self, base_width, dataset):
        backbone = []
        # for name, module in resnet50().named_children():
        for name, module in ResNet50(width=base_width).named_children():
            if name == 'conv1':
                module = nn.Conv2d(
                    # 3, 64, kernel_size=3, stride=1,
                    3, base_width, kernel_size=3, stride=1,
                    padding=1, bias=False
                )
            # check validity for adding layer to module
            if self.is_valid_layer(module, dataset):
                backbone.append(module)
        
        self.feats = nn.Sequential(*backbone)

    def _resnet18mod(self, base_width, dataset):
        backbone = []
        for name, module in ResNet18(width=base_width).named_children():
            if name == 'conv1':
                module = nn.Conv2d(
                    3, base_width, kernel_size=3, stride=1,
                    padding=1, bias=False
                )
            # check validity for adding layer to module
            if self.is_valid_layer(module, dataset):
                backbone.append(module)
        self.feats = nn.Sequential(*backbone)

    def _shallowConvmod(self,dataset,layers=2):
        assert layers%2==0, "Set number of layers for shallow Conv to be even, currently {}".format(layers)
        backbone = []
        in_ch = 3
        out_ch = 16
        for lidx in range(layers):
            in_ch = min(512,in_ch)
            out_ch = min(512,out_ch)
            if lidx < 4:  # give the first 4 layers MaxPool
                module = [nn.Conv2d(in_ch,out_ch,kernel_size=3, stride=1,padding=1,bias=True),
                          nn.BatchNorm2d(out_ch),
                          nn.ReLU(),
                          nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
            else:
                module = [nn.Conv2d(in_ch,out_ch,kernel_size=3, stride=1,padding=1,bias=True),
                          nn.BatchNorm2d(out_ch),
                          nn.ReLU()]
            backbone.extend(module)
            in_ch = out_ch
            out_ch = out_ch*4
        out_size = int(np.sqrt(2048/(out_ch/2)))
        backbone.append(nn.AdaptiveAvgPool2d(output_size=(out_size, out_size)))
        self.feats = nn.Sequential(*backbone)

    def _shallowConvDualmod(self,dataset,layers=4):
        assert layers%2==0, "Set number of layers for shallow Conv to be even, currently {}".format(layers)
        shallow= []
        stream1 = []
        deep = []
        in_ch = 3
        out_ch = 16
        for lidx in range(layers):
            in_ch = min(512,in_ch)
            out_ch = min(512,out_ch)
            if lidx == 0:
                shallow = [nn.Conv2d(in_ch,out_ch,kernel_size=3, stride=1,padding=1,bias=True),
                           nn.BatchNorm2d(out_ch),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
            elif lidx != 0 and lidx != layers-1:
                if lidx < 4: # give the first 4 layers MaxPool
                    module1 = [nn.Conv2d(in_ch,out_ch,kernel_size=3, stride=1,padding=1,bias=True),
                               nn.BatchNorm2d(out_ch),
                               nn.ReLU(),
                               nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
                else:
                    module1 = [nn.Conv2d(in_ch,out_ch,kernel_size=3, stride=1,padding=1,bias=True),
                               nn.BatchNorm2d(out_ch),
                               nn.ReLU()]
                stream1.extend(module1)
            elif lidx == layers-1:
                in_ch = in_ch*2
                deep = [nn.Conv2d(in_ch,out_ch,kernel_size=3, stride=1,padding=1,bias=True),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU()]
            in_ch = out_ch
            out_ch = out_ch*4
        out_size = int(np.sqrt(2048/(out_ch/2)))
        deep.append(nn.AdaptiveAvgPool2d(output_size=(out_size, out_size)))
        # self.feats = nn.Sequential(*backbone)
        self.shallow_out = nn.Sequential(*shallow)
        self.stream1_out = nn.Sequential(*stream1)
        self.stream2_out = deepcopy(self.stream1_out)
        self.deep_out = nn.Sequential(*deep)


    def build_projector(self, projector_dim, hidden_dim):
        projector = [
            nn.Linear(self.feat_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projector_dim, bias=True),
        ]
        self.proj = nn.Sequential(*projector)

    def build_predictor(self, projector_dim, use_mlp=True):
        predictor = [
            nn.Linear(projector_dim, projector_dim, bias=False)
        ]
        if use_mlp:
            predictor += [
                nn.ReLU(inplace=True),
                nn.Linear(projector_dim, projector_dim, bias=True),
            ]
        self.pred = nn.Sequential(*predictor)

    def is_valid_layer(self, module, dataset):
        """
        Check if a layer is valid for the dataset.
        """
        if 'cifar' in dataset:
            return self._check_valid_layer_cifar10(module)
        elif 'stl' in dataset:
            return self._check_valid_layer_stl10(module)
        elif 'imagenet' in dataset:
            return self._check_valid_layer_imagenet(module)
        else:
            raise NotImplementedError

    def _check_valid_layer_cifar10(self, module):
        if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
            return True
        return False

    def _check_valid_layer_stl10(self, module):
        if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
            return True
        return False

    def _check_valid_layer_imagenet(self, module):
        if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d): # TODO: ask Arna why MaxPool2d or Linear are not valid layers for ImageNet?
            return True
        return False

    def forward(self, x):
        if 'dualstream' in self.name:
            shallow_out = self.shallow_out(x)
            stream1_out = self.stream1_out(shallow_out)
            stream2_out = self.stream2_out(shallow_out)
            deep_out = self.deep_out(torch.cat((stream1_out, stream2_out), dim=1))
            return deep_out
        else:
            return self.feats(x)
        
    def forward_backbone(self, x):
        return self.feats(x)
    
    def forward_projector(self, x):
        return self.proj(x)


class streamNet(nn.Module):
    def __init__(self):
        super(streamNet, self).__init__()

        shallow_module = [nn.Conv2d(3, 16,kernel_size=3, stride=1,padding=1,bias=True),
                          nn.BatchNorm2d(16),
                          nn.ReLU(),
                          nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
        self.shallow_module = nn.Sequential(*shallow_module)

        backbone = [nn.Conv2d(16, 64,kernel_size=3, stride=1,padding=1,bias=True),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),

                    nn.Conv2d(64, 256,kernel_size=3, stride=1,padding=1,bias=True),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),

                    nn.Conv2d(256, 512,kernel_size=3, stride=1,padding=1,bias=True),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
                    ]

        self.backbone_modules = nn.Sequential(*backbone)

        skip_module = [nn.Conv2d(16, 512,kernel_size=6, stride=9,padding=1,bias=True),
                       nn.BatchNorm2d(512),
                       nn.ReLU()]
        self.skip_module = nn.Sequential(*skip_module)

        out_size = int(np.sqrt(2048/512))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(out_size, out_size))


    def forward(self, x):
        shallow_out = self.shallow_module(x)
        backbone_out = self.backbone_modules(shallow_out)
        skip_out = self.skip_module(shallow_out)
        output = self.avgpool(skip_out + backbone_out)
        return output


class dualstreamNet(nn.Module):
    def __init__(self):
        super(dualstreamNet, self).__init__()

        shallow_module = [nn.Conv2d(3, 16,kernel_size=3, stride=1,padding=1,bias=True),
                          nn.BatchNorm2d(16),
                          nn.ReLU(),
                          nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
        self.shallow_module = nn.Sequential(*shallow_module)

        stream1 = [nn.Conv2d(16, 64,kernel_size=3, stride=1,padding=1,bias=True),
                   nn.BatchNorm2d(64),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
                   nn.Conv2d(64, 256,kernel_size=3, stride=1,padding=1,bias=True),
                   nn.BatchNorm2d(256),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
        self.stream1 = nn.Sequential(*stream1)
        self.stream2 = deepcopy(self.stream1)

        deep_module = [nn.Conv2d(256, 512,kernel_size=3, stride=1,padding=1,bias=True),
                       nn.BatchNorm2d(512),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
        self.deep_module = nn.Sequential(*deep_module)
        self.deep_module2 = deepcopy(self.deep_module)
        self.deep_module3 = deepcopy(self.deep_module)
        self.deep_module4 = deepcopy(self.deep_module)

        skip_module = [nn.Conv2d(16, 512,kernel_size=6, stride=9,padding=1,bias=True),
                       nn.BatchNorm2d(512),
                       nn.ReLU()]
        self.skip_module = nn.Sequential(*skip_module)
        self.skip_module2 = deepcopy(self.skip_module)

        out_size = int(np.sqrt(2048/512))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(out_size, out_size))


    def forward(self, x):

        shallow_out = self.shallow_module(x)
        stream1_out = self.stream1(shallow_out)
        stream2_out = self.stream2(shallow_out)

        out_1 = self.deep_module(stream1_out)
        out_2 = self.deep_module2(stream1_out)

        out_3 = self.deep_module3(stream2_out)
        out_4 = self.deep_module4(stream2_out)

        sum_1 = out_1 + out_3 + self.skip_module(shallow_out)
        sum_2 = out_2 + out_4 + self.skip_module2(shallow_out)

        output = self.avgpool(sum_1 + sum_2)
        return output