from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict
from .training_checkpoint_repository import TrainingCheckpointRepository, TrainingCheckpoint
from wandb.wandb_run import Run

__all__ = ['TrainingCheckpointWandBRepository']

class TrainingCheckpointWandBRepository(TrainingCheckpointRepository):
    def __init__(self, run: Run):
        super().__init__()

        if run is None:
            raise TypeError('run')

        self.run: Run = run
        self.cache: Dict[str, TrainingCheckpoint] = {}

    def get_file_name(self, name) -> str:
        return f'{name}.json'

    def get_file_path(self, name) -> Path:
        return Path(self.run.dir) / self.get_file_name(name)

    async def get(self, name: str) -> TrainingCheckpoint:
        if name in self.cache:
            return self.cache[name]

        try:
            checkpoint_file = self.run.restore(self.get_file_name(name))

            content_dict: Dict[str, Any] = json.load(checkpoint_file)

            training_checkpoint: TrainingCheckpoint = TrainingCheckpoint(**content_dict)

            self.cache[name] = training_checkpoint

            return training_checkpoint
        except:
            return None

    async def save(self, checkpoint: TrainingCheckpoint, name: str):
        self.cache[name] = checkpoint

        content_dict: Dict[str, Any] = asdict(checkpoint)

        file_path: Path = self.get_file_path(name)
        file_path.touch()

        file = file_path.open('w')

        json.dump(content_dict, file)

        file.close()

        self.run.save(str(file_path))