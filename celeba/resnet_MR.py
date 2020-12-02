import torch
import torch.nn as nn
import torch.nn.functional as F
from MR_layer import MR

class ConvLayer(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, norm_layer=None, num_tasks=4, sigma=0.4):
        super(ConvLayer, self).__init__()
        modules = [nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                   stride=stride, padding=padding, bias=False)]
        if norm_layer is not None:
            modules.append(norm_layer(num_features=out_planes, track_running_stats=False))
        modules.append(MR(out_planes, num_tasks, sigma))
        self.conv_block = nn.Sequential(*modules)
        

    def forward(self, x):
        return self.conv_block(x)
    
    def get_weight(self):
        return self.conv_block[0].weight
        
    def get_routing_block(self):
        return self.conv_block[-1]
    
    def get_routing_masks(self):
        mapping = self.conv_block[-1].unit_mapping.detach().cpu().numpy()
        tested = self.conv_block[-1].tested_tasks.detach().cpu().numpy()
        return mapping, tested 
        
        
        
def get_blocks(model):
    if isinstance(model, ConvLayer):
        blocks = [model]
    else:
        blocks = []
        for child in model.children():
            blocks += get_blocks(child)
    return blocks


 

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=64, norm_layer=None, num_tasks=4, sigma=0.4):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if base_width != 64:
            raise ValueError('BasicBlock only supports base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = ConvLayer(inplanes, planes, kernel_size=3, padding=1, stride=stride, norm_layer=norm_layer, num_tasks=num_tasks, sigma=sigma)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, padding=1, norm_layer=norm_layer, num_tasks=num_tasks, sigma=sigma)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.len = 3 if downsample is not None else 2
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    
    
class ResNet(nn.Module):
    def __init__(self, block, layers, task_groups,
                 width_per_group=64, norm_layer=None, sigma=0.4):
        super(ResNet, self).__init__()
        self.active_task = 0
        self.sigma = sigma
        self.num_tasks = len(task_groups)
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.base_width = width_per_group
        self.conv1 = ConvLayer(3, self.inplanes, kernel_size=7, stride=2, padding=3, norm_layer=norm_layer, num_tasks=self.num_tasks, sigma=sigma)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.ModuleList()
        for task in range(len(task_groups)):
            self.fc.append(nn.Linear(512, len(task_groups[task])))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = ConvLayer(self.inplanes, planes, kernel_size=1, stride=stride, num_tasks=self.num_tasks, sigma=self.sigma)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.base_width, norm_layer, num_tasks=self.num_tasks, 
                            sigma=self.sigma))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                base_width=self.base_width, norm_layer=norm_layer,
                                num_tasks=self.num_tasks, sigma=self.sigma))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc[self.active_task](x)

        return x

    
    def change_task(self, task):
        def aux(m):
            if hasattr(m, 'active_task'):
                m.set_active_task(task)
        self.apply(aux)

    def set_active_task(self, active_task):
        self.active_task = active_task
        
    def get_blocks(self):
        return get_blocks(self)
           
    def get_weights(self):
        return [block.get_weight() for block in self.get_blocks()]
    
    def get_routing_masks(self):
        return [block.get_routing_masks() for block in self.get_blocks()]
    
    def get_block(self, depth):
        return self.get_blocks()[depth]
    
    def get_weight(self, depth):
        return self.get_weights()[depth]
    
    def get_routing_mask(self, depth):
        return self.get_routing_masks()[depth]
    
    def update_routing_mask(self, depth, new_mask, device, reset=False):
        conv_block = self.get_block(depth)
        TR_layer = conv_block.get_routing_block()
        TR_layer.assign_mapping(new_mask, device, reset=reset)

    
    


def resnet18(task_groups, **kwargs):
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], task_groups, **kwargs)
