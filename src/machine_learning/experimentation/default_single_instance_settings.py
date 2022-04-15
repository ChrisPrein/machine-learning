from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DefaultSignleInstanceSettings:
    name: str
    params: Dict[str, Any]