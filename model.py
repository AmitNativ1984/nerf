import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
        Sine-Cosine positional encoding for input points
    """

    def __init__(self, 
                 n_freq: int,
                 include_input: bool = True,
                 log_space: bool = False) -> None:
        super().__init__()
        self.n_freq = n_freq
        self.log_space = log_space

        # Define the frequencies:
        if self.log_space:
            freq_bands = 2. ** torch.linspace(0., n_freq - 1, n_freq)
        else:
            freq_bands = torch.linspace(2 ** 0, 2 ** (n_freq - 1), n_freq)

        if include_input:
            self.embed_fncs = [lambda x: x]
        else:
            self.embed_fncs = []
        
        for freq in freq_bands:
            self.embed_fncs.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fncs.append(lambda x, freq=freq: torch.cos(x * freq))

        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            apply positional encoding to input tensor
        """

        return torch.cat([f(x) for f in self.embed_fncs], dim=-1)
    

class TinyNerfModel(nn.Module):
    """
        Define a tiny NeRF model with 3 fully connected layers
    """

    def __init__(self, filter_size=128, num_encoding_functions=6, include_input=True):
        super().__init__()

        if include_input:
            self.layer1 = nn.Linear(3 + 3 * 2 *num_encoding_functions, filter_size)
        else:
            self.layer1 = nn.Linear(3 * 2 * num_encoding_functions, filter_size)

        self.layer2 = nn.Linear(filter_size, filter_size)
        self.layer3 = nn.Linear(filter_size, filter_size)
        self.layer4 = nn.Linear(filter_size, 4)     # output shape: (rgb, density)

        self.relu = nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.layer4(x)
    

class FullNerfModel(nn.Module):
    """
        Define a medium NeRF model with 8 fully connected layers
    """

    def __init__(self, filter_size=256, num_encoding_functions=6, include_input=True):
        super().__init__()

        if include_input:
            self.layer1 = nn.Linear(3 + 3 * 2 *num_encoding_functions, filter_size)
        else:
            self.layer1 = nn.Linear(3 * 2 * num_encoding_functions, filter_size)

        self.layer2 = nn.Linear(filter_size, filter_size)
        self.layer3 = nn.Linear(filter_size, filter_size)
        self.layer4 = nn.Linear(filter_size, filter_size)
        # in layer5 we concatenate with the input
        self.layer5 = nn.Linear(filter_size + 3 + 3 * 2 *num_encoding_functions, filter_size)
        self.layer6 = nn.Linear(filter_size, filter_size)
        self.layer7 = nn.Linear(filter_size, filter_size)
        self.layer8 = nn.Linear(filter_size, filter_size)

        self.fc_density = nn.Linear(filter_size, 1)
        
        self.layer9 = nn.Linear(filter_size, 128)     
        self.fc_rgb = nn.Linear(128, 3)     
        
        self.relu = nn.functional.relu       

    def forward(self, input):
        x = self.relu(self.layer1(input))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = torch.cat([x, input], dim=-1)
        x = self.relu(self.layer5(x))
        x = self.relu(self.layer6(x))
        x = self.relu(self.layer7(x))
        x = self.layer8(x)

        density = self.fc_density(x)

        x = self.relu(self.layer9(x))
        rgb = self.fc_rgb(x)

        return torch.cat([rgb, density], dim=-1)
