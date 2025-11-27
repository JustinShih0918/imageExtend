# utils/losses.py
import torch
import torch.nn as nn
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    """
    Computes Perceptual Loss using a pretrained VGG19 network.
    Input images should be in range [-1, 1].
    """
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        # Load VGG19 pretrained on ImageNet
        # We slice VGG into blocks to extract features at different depths
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        blocks = []
        blocks.append(vgg[:4].eval())
        blocks.append(vgg[4:9].eval())
        blocks.append(vgg[9:18].eval())
        blocks.append(vgg[18:27].eval()) # Added deeper layers for better structure
        blocks.append(vgg[27:36].eval())
        
        # Freeze parameters (we don't train VGG)
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
                
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        
        # ImageNet normalization parameters
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        # Input/Target are expected in [-1, 1], convert to [0, 1] for VGG
        input = (input + 1) / 2
        target = (target + 1) / 2
        
        # Normalize with ImageNet mean/std
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        # Resize to 224x224 if needed (standard VGG input size)
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        loss = 0.0
        x = input
        y = target
        
        # Compute L1 loss between features at each block
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
            
        return loss