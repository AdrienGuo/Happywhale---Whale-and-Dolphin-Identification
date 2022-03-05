# Reference: https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from torchsummary import summary

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


data_path = "images/"
train_df = pd.read_csv(data_path + "train.csv")
train_df['path'] = data_path + "train_images/" + train_df['image']
print(train_df['path'][0])

img = Image.open(train_df['path'][0])
print(img)
fig = plt.figure()
plt.imshow(img)

# resize to imagenet size
transform = Compose([Resize((224, 224)), ToTensor()])
x = transform(img)
x = x.unsqueeze(0)      # add Batch dim
print("x shape: ", x.shape)

# divide the input picture to patches
patch_size = 16
pathes = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
print(pathes.shape)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int=3, patch_size: int=16, emb_size: int=768, img_size: int = 224) :
        self.patch_size = patch_size
        super().__init__()
        self.proj = nn.Sequential(
            # break-down the image in p1 x p2 patches and flat them
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            # nn.Linear(patch_size * patch_size * in_channels, emb_size)

            # in the original implementation, the author used a Con2d instead of the way above
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 +1, emb_size))
    
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.proj(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)     # repeat 就是用來增加維度的

        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)

        # add position embedding
        x += self.positions
        return x

print("patches_embedded: ", PatchEmbedding()(x).shape)

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int=768, num_heads: int=8, dropout: float=0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.queries = nn.Linear(emb_size, emb_size)
        self.keys = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)         # 應該是最後 concat 起來，然後要做大小的變換時用的
        
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), 'b n (h d) -> b h n d', h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        # bhqk: batch, num_heads, query_len, key_len
        # get rid of the "d" dimension ??
        # I guess einsum will do "matrix multiplication".

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)             # ~ 是甚麼鬼
        
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)

        # sum up over the third axis
        # out = torch.einsum('bhal, bhlv -> bhav', att, values)       # 我覺得原文寫得不好
        out = torch.einsum('bhav, bhvd -> bhad', att, values)         # 我自己改寫的
        # a: query_len = key_len = value_len
        # v: value_len, d: value_dim 因為 v_d 不一定等於 q_d, k_d
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)

        return out

patches_embedded = PatchEmbedding()(x)
print("multihead_attn: ", MultiHeadAttention()(patches_embedded).shape)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, **kwargs):             # keyword arguments
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwadBlock(nn.Sequential):           # 這裡用 nn.Sequential 很特別，似乎因為這樣就不用寫 forwad 了
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
    
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwadBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p),
            )
            ))

print("TransformerEncoderBlock: ", TransformerEncoderBlock()(patches_embedded).shape)

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

class ViT(nn.Sequential):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 768,
                 img_size: int = 224,
                 depth: int = 12,
                 n_classes: int = 1000,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

print(summary(ViT(), (3, 224, 224), device='cuda'))