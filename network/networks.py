import torch.nn as nn
from triangular_layers import lower_traingular_layer, upper_traingular_layer


class dense_network(nn.Module):
    def __init__(self, layer_size):
        super(dense_network, self).__init__()
        self.layer1 = nn.Linear(layer_size, layer_size, bias=False)
        self.layer2 = nn.Linear(layer_size, layer_size, bias=False)
        self.layer3 = nn.Linear(layer_size, layer_size, bias=False)
        self.layer4 = nn.Linear(layer_size, layer_size, bias=False)
        self.leakyRelu = nn.LeakyReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.leakyRelu(out)
        out = self.layer2(out)
        out = self.leakyRelu(out)
        out = self.layer3(out)
        out = self.leakyRelu(out)
        out = self.layer4(out)
        return out


class sparse_network(nn.Module):
    def __init__(self, layer_size):
        super(sparse_network, self).__init__()
        self.upper_traingular1 = upper_traingular_layer(layer_size, layer_size)
        self.lower_traingular1 = lower_traingular_layer(layer_size, layer_size)
        self.upper_traingular2 = upper_traingular_layer(layer_size, layer_size)
        self.lower_traingular2 = lower_traingular_layer(layer_size, layer_size)
        self.upper_traingular3 = upper_traingular_layer(layer_size, layer_size)
        self.lower_traingular3 = lower_traingular_layer(layer_size, layer_size)
        self.upper_traingular4 = upper_traingular_layer(layer_size, layer_size)
        self.lower_traingular4 = lower_traingular_layer(layer_size, layer_size)
        self.leakyRelu = nn.LeakyReLU()

    def forward(self, x):
        out = self.lower_traingular1(x)
        out = self.upper_traingular1(out)
        out = self.leakyRelu(out)
        out = self.lower_traingular2(out)
        out = self.upper_traingular2(out)
        out = self.leakyRelu(out)
        out = self.lower_traingular3(out)
        out = self.upper_traingular3(out)
        out = self.leakyRelu(out)
        out = self.upper_traingular4(out)
        out = self.lower_traingular4(out)

        return out
