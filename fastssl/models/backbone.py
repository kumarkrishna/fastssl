import torch.nn as nn
from torchvision.models.resnet import resnet50
import numpy as np
import torch

class BackBone(nn.Module):
    def __init__(self,
                 name='resnet50feat',
                 dataset='cifar10', 
                 projector_dim=128,
                 hidden_dim=128):
        super(BackBone, self).__init__()
        self.name = name
        self.build_backbone(dataset=dataset, projector_dim=projector_dim, hidden_dim=hidden_dim)

    def build_backbone(self, dataset='cifar10', projector_dim=128, hidden_dim=512):
        """
        Build backbone model.
        """
        if 'resnet' in self.name:
            self._resnet50mod(dataset)
        else:
            num_layers = int(self.name.split('_')[-1]) if len(self.name.split('_'))>1 else 2
            self.name = self.name.split('_')[0] if len(self.name.split('_'))>1 else self.name
            if 'dualstream' in self.name:
                self._shallowConvDualmod(dataset, layers=num_layers)
            else:
                self._shallowConvmod(dataset,layers=num_layers)
        if 'proj' in self.name:
            self.build_projector(projector_dim=projector_dim, hidden_dim=hidden_dim)
        if 'pred' in self.name:
            self.build_predictor(projector_dim=projector_dim)

    def _resnet50mod(self, dataset):
        backbone = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1,
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
            module = [nn.Conv2d(in_ch,out_ch,kernel_size=3, stride=1,padding=1,bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU()]
            backbone.extend(module)
            in_ch = out_ch
            out_ch = out_ch*2
        out_size = int(np.sqrt(2048/(out_ch/2)))
        backbone.append(nn.AdaptiveAvgPool2d(output_size=(out_size, out_size)))
        self.feats = nn.Sequential(*backbone)

    def _shallowConvDualmod(self,dataset,layers=4):
        assert layers%2==0, "Set number of layers for shallow Conv to be even, currently {}".format(layers)
        shallow= []
        stream1 = []
        stream2 = []
        deep = []
        in_ch = 3
        out_ch = 16
        for lidx in range(layers):
            in_ch = min(512,in_ch)
            out_ch = min(512,out_ch)
            if lidx == 0:
                shallow = [nn.Conv2d(in_ch,out_ch,kernel_size=3, stride=1,padding=1,bias=False),
                          nn.BatchNorm2d(out_ch),
                          nn.ReLU()]
            if lidx != 0 and lidx != layers-1:
                module1 = [nn.Conv2d(in_ch,out_ch,kernel_size=3, stride=1,padding=1,bias=False),
                          nn.BatchNorm2d(out_ch),
                          nn.ReLU()]
                stream1.extend(module1)
                module2 = module1.copy()
                stream2.extend(module2)
            if lidx == layers-1:
                in_ch = in_ch*2
                deep = [nn.Conv2d(in_ch,out_ch,kernel_size=3, stride=1,padding=1,bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU()]
            in_ch = out_ch
            out_ch = out_ch*2
        out_size = int(np.sqrt(2048/(out_ch/2)))
        deep.append(nn.AdaptiveAvgPool2d(output_size=(out_size, out_size)))
        # self.feats = nn.Sequential(*backbone)
        self.shallow_out = nn.Sequential(*shallow)
        self.stream1_out = nn.Sequential(*stream1)
        self.stream2_out = nn.Sequential(*stream2)
        self.deep_out = nn.Sequential(*deep)

    
    def build_projector(self, projector_dim, hidden_dim):
        projector = [
            nn.Linear(2048, hidden_dim, bias=False),
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
        if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
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
