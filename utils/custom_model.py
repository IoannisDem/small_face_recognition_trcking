import torch
import torch.nn as nn
import torchvision
from torchvision import models

class distill_model(nn.Module):
    def __init__(self, output_embed=512):
        super(distill_model, self).__init__()
        self.model = models.mobilenet_v3_small(pretrained=True)
        
        new_classifier = nn.Linear(576, output_embed, bias=True)
        
        self.model.classifier = new_classifier
        
    def forward(self, x):
        return self.model(x)
