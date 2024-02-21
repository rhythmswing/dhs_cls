
import torch

class LinearClassificationModule(torch.nn.Module):
    """
    Simple linear layer for classification.
    """

    def __init__(self, input_size, output_size):
        super(LinearClassificationModule, self).__init__()
        self.net = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.net(x)

class TwoLayerClassificationModule(torch.nn.Module):
    """
    Simple two-layer classification.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerClassificationModule, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


class TwoLayerJointClassificationModule(torch.nn.Module):
    """
    Two-layer classification module with incorporation of additional features.
    """
    
    def __init__(self, input_size, hidden_size, output_size, additional_feature_size):
        super(TwoLayerJointClassificationModule, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size))

        self.additional_net = torch.nn.Linear(additional_feature_size, hidden_size)

        self.output_net = torch.nn.Linear(hidden_size*2, output_size)


    def forward(self, x, y):
        o1 = self.net(x) 
        o2 = self.additional_net(y)
        joint = torch.nn.functional.silu(torch.cat((o1, o2), dim=1))
        o = self.output_net(joint)
        return o