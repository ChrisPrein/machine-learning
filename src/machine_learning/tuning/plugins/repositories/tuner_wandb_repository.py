from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict
from .tuner_repository import TunerRepository
from wandb.wandb_run import Run
from ray.tune import Tuner

__all__ = ['TunerWandBRepository']

class TunerWandBRepository(TunerRepository):
    def __init__(self, run: Run):
        super().__init__()

        if run is None:
            raise TypeError('run')

        self.run: Run = run
        self.cache: Dict[str, Tuner] = {}
        self.files_dir: Path = Path(self.run.settings.files_dir)

    async def get(self, name: str) -> Tuner:
        if name in self.cache:
            return self.cache[name]

        try:
            experiment_dir: Path = self.files_dir / name

            if not experiment_dir.is_dir() or len(list(experiment_dir.rglob('*'))) <= 2:
                return None

            tuner: Tuner = Tuner.restore(str(experiment_dir))

            self.cache[name] = tuner

            return tuner
        except:
            return None