import torchvision
from torch import nn

class FeatureResNet(nn.Module):
    """ResNet model for feature extraction."""
    def __init__(self):
        super(FeatureResNet, self).__init__()
        self.resnet = torchvision.models.resnet50(weights="IMAGENET1K_V1")

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)

        return x