import torch
import torch.nn as nn


class SampleNet(nn.Module):
    """
    TNet module for adversarial networks with fixed activation layers and predefined parameters.
    """

    def __init__(self, feature_dim=64, t_batchsize=64, t_var=1):
        super(SampleNet, self).__init__()
        self.feature_dim = feature_dim  # Feature dimension
        self.t_sigma_num = t_batchsize // 16  # Number of sigmas for t_net
        self._input_adv_t_net_dim = feature_dim  # Input noise dimension
        self._input_t_dim = feature_dim  # t_net input dimension
        self._input_t_batchsize = t_batchsize  # Batch size
        self._input_t_var = t_var  # Variance of input noise

        # Fixed activation layers
        self.activation_1 = nn.LeakyReLU(negative_slope=0.2)
        self.activation_2 = nn.Tanh()

        # Create a simple 3-layer fully connected network using fixed activation layers
        self.t_layers_list = nn.ModuleList()
        ch_in = self.feature_dim
        num_layer = 3
        for i in range(num_layer):
            self.t_layers_list.append(nn.Linear(ch_in, ch_in))
            self.t_layers_list.append(nn.BatchNorm1d(ch_in))
            # Use activation_1 for the first two layers, and activation_2 for the last layer
            self.t_layers_list.append(
                self.activation_1 if i < (num_layer - 1) else self.activation_2
            )

    def forward(self, device):
        # Generate white noise
        if self.t_sigma_num > 0:
            # Initialize the white noise input
            self._t_net_input = torch.randn(
                self.t_sigma_num, self._input_adv_t_net_dim
            ) * (self._input_t_var**0.5)
            self._t_net_input = self._t_net_input.to(device).detach()

            # Forward pass
            a = self._t_net_input
            for layer in self.t_layers_list:
                a = layer(a)

            a = a.repeat(int(self._input_t_batchsize / self.t_sigma_num), 1)

            # Generate the final t value
            # self._t = torch.randn(self._input_t_batchsize, self._input_t_dim) * ((self._input_t_var / self._input_t_dim) ** 0.5)
            # self._t = self._t.to(device).detach()
            self._t = a
        else:
            # When t_sigma_num = 0, generate standard Gaussian noise as t
            self._t = torch.randn(self._input_t_batchsize, self._input_t_dim) * (
                (self._input_t_var / self._input_t_dim) ** 0.5
            )
            self._t = self._t.to(device).detach()
        return self._t