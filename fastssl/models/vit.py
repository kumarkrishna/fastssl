import torch.nn as nn

import timm
from timm.models import register_model
from timm.models.vision_transformer import _create_vision_transformer

vit_configs = {
    "tiny": { "num_heads": 3},
    "small": { "num_heads": 4},
    "": { "num_heads": 6},
}

def create_model_constructor(embed_dim, num_heads, config_name, no_bn):
    """ Register custom models with timm
    """
    if len(config_name) > 0:
        config_name += "_"
    bn_str = "_nobn" if no_bn else ""
    model_name = f"vit_{config_name}patch16_224_{embed_dim}{bn_str}"
    norm_layer = nn.Identity if no_bn else nn.LayerNorm
    def constructor_fn(pretrained=False, **kwargs):
        """ ViT (ViT-S/16)
        """
        model_kwargs = dict(
            patch_size=16,
            embed_dim=embed_dim,
            depth=12,
            num_heads=num_heads,
            mlp_ratio=4.,
            norm_layer=norm_layer,
            global_pool='token',
        )
        model = _create_vision_transformer(model_name, pretrained=pretrained, **dict(model_kwargs, **kwargs))
        return model
    
    constructor_fn.__name__ = model_name
    return constructor_fn


for config in vit_configs:
    for no_bn in [True, False]:
        for dim in range(1,65):
            num_heads = vit_configs[config]["num_heads"]
            embed_dim = dim * num_heads
            register_model(create_model_constructor(embed_dim, num_heads, config, no_bn))


def ViT(width, no_bn=False, num_classes=0, in_chans=3, **kwargs):
    """ Vision Transformer with input size 224x224 and patch size 16
        6 MLP heads and learnable class token
    """
    config = ""
    num_heads = vit_configs[config]["num_heads"]
    embed_dim = width * num_heads
    bn_str = "_nobn" if no_bn else ""
    model = timm.create_model(
        f"vit_{config}patch16_224_{embed_dim}{bn_str}",
        num_classes=num_classes,
        in_chans=in_chans,
        **kwargs,
    )
    return model


def ViTSmall(width, no_bn=False, num_classes=0, in_chans=3, **kwargs):
    """ Vision Transformer with input size 224x224 and patch size 16
        4 MLP heads and learnable class token
    """
    config = "small"
    num_heads = vit_configs[config]["num_heads"]
    embed_dim = width * num_heads
    bn_str = "_nobn" if no_bn else ""
    model = timm.create_model(
        f"vit_{config}_patch16_224_{embed_dim}{bn_str}",
        num_classes=num_classes,
        in_chans=in_chans,
        **kwargs,
    )
    return model


def ViTTiny(width, no_bn=False, num_classes=0, in_chans=3, **kwargs):
    """ Vision Transformer with input size 224x224 and patch size 16
        3 MLP heads and learnable class token
    """
    config = "tiny"
    num_heads = vit_configs[config]["num_heads"]
    embed_dim = width * num_heads
    bn_str = "_nobn" if no_bn else ""
    model = timm.create_model(
        f"vit_{config}_patch16_224_{embed_dim}{bn_str}",
        num_classes=num_classes,
        in_chans=in_chans,
        **kwargs,
    )
    return model

