from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict
from .model_metadata_repository import ModelMetadataRepository, ModelMetadata
from wandb.wandb_run import Run

__all__ = ['ModelMetadataWandBRepository']

class ModelMetadataWandBRepository(ModelMetadataRepository):
    def __init__(self, run: Run):
        super().__init__()

        if run is None:
            raise TypeError('run')

        self.run: Run = run
        self.cache: Dict[str, ModelMetadata] = {}
        self.files_dir: Path = Path(self.run.settings.files_dir)

    def get_file_name(self, name) -> str:
        return f'{name}.json'

    def get_file_path(self, name) -> Path:
        return Path(self.run.dir) / self.get_file_name(name)

    async def get(self, name: str) -> ModelMetadata:
        if name in self.cache:
            return self.cache[name]

        try:
            file_path: Path = self.files_dir / self.get_file_name(name)

            content_dict: Dict[str, Any] = json.load(file_path.open())

            model_metadata: ModelMetadata = ModelMetadata(**content_dict)

            self.cache[name] = model_metadata

            return model_metadata
        except:
            return None

    async def save(self, metadata: ModelMetadata, name: str):
        self.cache[name] = metadata

        content_dict: Dict[str, Any] = asdict(metadata)

        file_path: Path = self.get_file_path(name)
        file_path.touch()

        file = file_path.open('w')

        json.dump(content_dict, file)

        file.close()

        self.run.save(str(file_path))