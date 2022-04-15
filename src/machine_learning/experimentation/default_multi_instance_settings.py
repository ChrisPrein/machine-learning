from dataclasses import dataclass
from typing import Any, Dict

from .default_single_instance_settings import DefaultSignleInstanceSettings

@dataclass
class DefaultMultiInstanceSettings:
    instances: Dict[str, DefaultSignleInstanceSettings]