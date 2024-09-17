# https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py
import torch
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn, channel_last = True):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.channel_last = channel_last
    def forward(self, x):
        if self.channel_last:
            return self.fn(self.norm(x)) + x
        else:
            return self.fn(rearrange(self.norm(rearrange(x, 'b c n -> b n c')), 'b n c -> b c n')) + x
    
class SoftmaxLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SoftmaxLinear, self).__init__(in_features, out_features, bias)
        self.softmax = nn.Softmax(dim=1)  # Softmax across the input dimension

    def forward(self, input):
        # Apply softmax to the weights (transpose to apply softmax across columns, then transpose back)
        normalized_weight = self.softmax(self.weight) 

        # Perform the linear operation using matrix multiplication and bias addition
        output = input.matmul(normalized_weight.t())
        if self.bias is not None:
            output += self.bias
        return output 
    
def FeedForward(dim, expansion_factor = 4, dropout = 0., first = nn.Linear, second = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        first(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        second(inner_dim, dim),
        nn.Dropout(dropout)
    )

class DoubleFeedForward(nn.Module):
    def __init__(self, dim1, dim2, expansion_factor = 4, expansion_factor_token = 0.5,  dropout =  0.):
        super().__init__()
      
        inner_dim1 = int(dim1 * expansion_factor)
        inner_dim2 = int(dim2 *  expansion_factor_token)
        self.first1 = nn.Linear(dim1, inner_dim1)
        self.first2 = nn.Linear(dim2, inner_dim2)
        self.second1 = nn.Linear(inner_dim1, dim1)
        self.second2 = nn.Linear(inner_dim2, dim2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_2 = rearrange(x, 'b n c -> b c n')

        x = self.dropout(self.first1(x))
        x = self.dropout(self.second1(x))
        x_2 = self.dropout(self.first2(x_2))
        x_2 = self.dropout(self.second2(x_2))

        x_2 = rearrange(x_2, 'b c n -> b n c')
        
        return x + x_2

def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            Rearrange('b n c -> b c n'),
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout), channel_last=False),
            Rearrange('b c n -> b n c'),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )

class Patchify(nn.Module):
    def __init__(self, image_size, channels, patch_size, dim):
        super().__init__()
        res1_patches = (image_size // patch_size) ** 2
        dim1 = (patch_size ** 2) * channels
        self.res1 = nn.Unfold(kernel_size=patch_size, stride = patch_size) 
        
        self.res1dilate1 = nn.Unfold(kernel_size=patch_size, stride = patch_size, padding=2, dilation=2) 

        self.proj = nn.Linear(dim1 + dim1, dim)

    def forward(self, x):
        x1 = self.res1(x)
        x2 = self.res1dilate1(x)

        x = torch.cat([x1, x2], dim=1)
        x = rearrange(x, 'b c n -> b n c')
        x = self.proj(x)
        return x 

def PatchMLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 2, expansion_factor_token = 0.25, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)

    return nn.Sequential(
        Patchify(image_size, channels, patch_size, dim),
        *[nn.Sequential(
            PreNormResidual(dim, DoubleFeedForward(dim, num_patches, expansion_factor, expansion_factor_token, dropout)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout)),
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )

