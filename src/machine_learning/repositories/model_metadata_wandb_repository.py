from typing import Dict
from .model_metadata_repository import ModelMetadataRepository, ModelMetadata
from wandb.wandb_run import Run

class ModelMetadataWandBRepository(ModelMetadataRepository):
    def __init__(self, run: Run):
        super().__init__()

        if run is None:
            raise TypeError('run')

        self.run: Run = run
        self.cache: Dict[str, ModelMetadata] = {}

    async def get(self, name: str) -> ModelMetadata: ...

    async def save(self, metadata: ModelMetadata, name: str): ...

    