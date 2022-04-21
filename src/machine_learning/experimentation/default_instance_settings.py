from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DefaultInstanceSettings:
    name: str
    params: Dict[str, Any]