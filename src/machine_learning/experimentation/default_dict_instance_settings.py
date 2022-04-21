from dataclasses import dataclass
from typing import Any, Dict
from .default_instance_settings import DefaultInstanceSettings


@dataclass
class DefaultDictInstanceSettings:
    instances: Dict[str, DefaultInstanceSettings]