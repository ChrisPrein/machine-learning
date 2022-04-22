from typing import Any, Callable, Dict, Generic, List, Type, Union

from .abstractions.instance_factory import InstanceFactory, TInstance
from .default_instance_settings import DefaultInstanceSettings
from .default_instance_factory import DefaultInstanceFactory

class DefaultNamedSingleInstanceFactory(Generic[TInstance], InstanceFactory[Dict[str, DefaultInstanceSettings], Dict[str, TInstance]]):
    def __init__(self, available_types: Dict[str, Union[Type[TInstance], Callable[[Dict[str, DefaultInstanceSettings]], TInstance]]]):
        if available_types is None:
            raise ValueError("available_types")

        self.__single_instance_factory: InstanceFactory[DefaultInstanceSettings, TInstance] = DefaultInstanceFactory[TInstance](available_types)

    def create(self, settings: Dict[str, DefaultInstanceSettings]) -> Dict[str, TInstance]:
        if settings is None:
            raise ValueError("settings")

        instances: Dict[str, TInstance] = {}

        for alias, instance_item in settings.items():
            instances[alias] = self.__single_instance_factory.create(instance_item)

        return instances

        