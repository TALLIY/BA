import torch.nn as nn


class DenseNetwork(nn.Module):
    def __init__(self, layer_size):
        super(DenseNetwork, self).__init__()
        self.layer1 = nn.Linear(layer_size, layer_size, bias=True)
        self.layer2 = nn.Linear(layer_size, layer_size, bias=True)
        self.layer3 = nn.Linear(layer_size, layer_size, bias=True)
        self.layer4 = nn.Linear(layer_size, layer_size, bias=True)
        self.layer5 = nn.Linear(layer_size, layer_size, bias=True)
        self.layer6 = nn.Linear(layer_size, layer_size, bias=True)
        self.activation = nn.SiLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.activation(out)
        out = self.layer2(out)
        out = self.activation(out)
        out = self.layer3(out)
        out = self.activation(out)
        out = self.layer4(out)
        out = self.activation(out)
        out = self.layer5(out)
        out = self.activation(out)
        out = self.layer6(out)
        out = self.activation(out)

        return out
