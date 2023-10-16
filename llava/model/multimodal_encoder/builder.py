import os
from .clip_encoder import CLIPVisionTower
from .dinov2_encoder import Dinov2VisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if is_absolute_path_exists or "dinov2" in vision_tower:
        return Dinov2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
