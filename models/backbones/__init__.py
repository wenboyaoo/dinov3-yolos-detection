from .dinov3 import build_dinov3
from .vanilla_vit import build_vanilla_vit

BACKBONE_BUILDERS = {
    'dinov3': build_dinov3,
    'vanilla_vit': build_vanilla_vit
}

def build_backbone(backbone='dinov3', **kwargs):
    assert backbone in BACKBONE_BUILDERS.keys()
    return BACKBONE_BUILDERS[backbone](**kwargs)