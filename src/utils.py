import os
import importlib
from typing import Any

import torch
import numpy as np
import omegaconf
from torchvision import transforms
from omegaconf import DictConfig


def set_seed(seed=42):
    """Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def get_transform(
    transforms_config: omegaconf.DictConfig, is_train: bool
) -> transforms.Compose:
    transforms_config = omegaconf.OmegaConf.to_object(
        transforms_config["train" if is_train else "test"]
    )
    transforms_list = []
    for t_name, t_params in transforms_config.items():
        transforms_list.append(getattr(transforms, t_name)(**t_params))
    transform = transforms.Compose(transforms_list)
    return transform


def get_criterion(criterion_cfg: DictConfig):
    criterion = []
    for crit in criterion_cfg:
        if crit.args:
            loss = load_obj(crit.name)(**crit.args)
        else:
            loss = load_obj(crit.name)()
        weight = crit.weight
        criterion.append((loss, weight))
    return criterion
