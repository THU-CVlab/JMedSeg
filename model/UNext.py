import jittor as jt
from jittor import nn

## drop_path & DropPath
#  Both borrowed from the famous Timm library (a part of https://github.com/huggingface/pytorch-image-models)
#  A slight modification has been made to run the sampling pass upon Jittor.

def drop_path(x: jt.Var, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True) -> jt.Var:
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    
    # original: 
    #
    #   random_tensor = x.new_empty(shape).bernoulli_()
    # 

    random_tensor = jt.bernoulli(x.new_full(shape, keep_prob))
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor /= keep_prob
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=0., scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def execute(self, x:jt.Var) -> jt.Var:
        return drop_path(x, self.drop_prob, self.is_training(), self.scale_by_keep)


## UNext on Jittor.
#  Referred to the original code (https://github.com/jeya-maria-jose/UNeXt-pytorch)
#  

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
    
    def execute(self, x: jt.Var, H, W) -> jt.Var:
        B, _, C = x.shape
        x = x.transpose((0, 2, 1)).view((B, C, H, W))
        x = self.dwconv(x)
        x = x.flatten(2).transpose((0, 2, 1))
        return x

class ShiftMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
        self.shift_size = shift_size
        self.pad = shift_size // 2

    def execute(self, x: jt.Var, H, W) -> jt.Var:
        B, N, C = x.shape
        
        xn = x.transpose((0, 2, 1)).view(B, C, H, W).contiguous()
        xn = nn.pad(xn, (self.pad, self.pad, self.pad, self.pad))
        xs = jt.chunk(xn, self.shift_size, 1)
        x_shift = [jt.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = jt.concat(x_shift, 1)
        x_s = x_cat[:, :, self.pad : self.pad + H, self.pad: self.pad + W]
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose((0, 2, 1))
        
        x = self.fc1(x_shift_r)
        
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        
        xn = x.transpose((0, 2, 1)).view(B, C, H, W).contiguous()
        xn = nn.pad(xn, (self.pad, self.pad, self.pad, self.pad))
        xs = jt.chunk(xn, self.shift_size, 1)
        x_shift = [jt.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = jt.concat(x_shift, 1)
        x_s = x_cat[:, :, self.pad : self.pad + H, self.pad: self.pad + W]
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose((0, 2, 1))
        
        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x
        

class ShiftedBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ShiftMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def execute(self, x: jt.Var, H, W) -> jt.Var:
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        
        self.H, self.W = img_size // patch_size, img_size // patch_size
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)
    
    def execute(self, x: jt.Var) -> tuple[jt.Var, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose((0, 2, 1))
        x = self.norm(x)
        
        return x, H, W
        

class UNext(nn.Module):
    def __init__(self, num_classes, input_channels=3, img_size=224, patch_size=3, patch_stride=2,
                 embed_dims=[128, 160, 256], mlp_ratios=[1., 1., 1., 1.],  drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, depths=[1, 1, 1],  **kwargs):
        super().__init__()
        
        self.encoder1 = nn.Conv2d(input_channels, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)
        
        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)
        
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])
        
        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)
        
        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, sum(depths))]
        
        self.block1 = ShiftedBlock(embed_dims[1], mlp_ratio=mlp_ratios[0], drop=drop_rate, drop_path=dpr[1],
                                   norm_layer=norm_layer)
        self.block2 = ShiftedBlock(embed_dims[2], mlp_ratio=mlp_ratios[1], drop=drop_rate, drop_path=dpr[2],
                                   norm_layer=norm_layer)
        
        self.dblock1 = ShiftedBlock(embed_dims[1], mlp_ratio=mlp_ratios[2], drop=drop_rate, drop_path=dpr[0],
                                    norm_layer=norm_layer)
        self.dblock2 = ShiftedBlock(embed_dims[0], mlp_ratio=mlp_ratios[3], drop=drop_rate, drop_path=dpr[0],
                                    norm_layer=norm_layer)
        
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=patch_size, stride=patch_stride,
                                              in_chans=embed_dims[0], embed_dim=embed_dims[1])
        
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=patch_size, stride=patch_stride,
                                              in_chans=embed_dims[1], embed_dim=embed_dims[2])
        
        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        
        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=1)
        
    def execute(self, x: jt.Var) -> jt.Var:
        B = x.shape[0]
        
        out = nn.relu(nn.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        
        out = nn.relu(nn.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        
        out = nn.relu(nn.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out
        
        out, H, W = self.patch_embed3(out)
        out = self.block1(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).transpose((0, 3, 1, 2)).contiguous()
        t4 = out
        
        out, H ,W = self.patch_embed4(out)
        out = self.block2(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).transpose((0, 3, 1, 2)).contiguous()
        
        out = nn.relu(nn.interpolate(self.dbn1(self.decoder1(out)), scale_factor=2))
        
        out = out + t4
        _, _, H, W = out.shape
        out = out.flatten(2).transpose((0, 2, 1))
        out = self.dblock1(out, H, W)
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).transpose((0, 3, 1, 2)).contiguous()
        out = nn.relu(nn.interpolate(self.dbn2(self.decoder2(out)), scale_factor=2))
        out = out + t3
        _, _, H, W = out.shape
        out = out.flatten(2).transpose((0, 2, 1))
        out = self.dblock2(out, H, W)
        
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).transpose((0, 3, 1, 2)).contiguous()
        
        out = nn.relu(nn.interpolate(self.dbn3(self.decoder3(out)), scale_factor=2))
        out = out + t2
        out = nn.relu(nn.interpolate(self.dbn4(self.decoder4(out)), scale_factor=2))
        out = out + t1
        out = nn.relu(nn.interpolate(self.decoder5(out), scale_factor=2))
        
        out = self.final(out)
        return out
        
        
