from typing import Callable, Dict, Generic, List, Type, Union

from .abstractions.instance_factory import InstanceFactory, TInstance
from .default_instance_settings import DefaultInstanceSettings
from .default_instance_factory import DefaultInstanceFactory
from .default_dict_instance_settings import DefaultDictInstanceSettings

class DefaultNamedSingleInstanceFactory(Generic[TInstance], InstanceFactory[DefaultDictInstanceSettings, Dict[str, TInstance]]):
    def __init__(self, available_types: Dict[str, Union[Type[TInstance], Callable[[DefaultInstanceSettings], TInstance]]]):
        if available_types is None:
            raise ValueError("available_types")

        self.__single_instance_factory: InstanceFactory[DefaultDictInstanceSettings, TInstance] = DefaultInstanceFactory[TInstance](available_types)

    def create(self, settings: DefaultDictInstanceSettings) -> Dict[str, TInstance]:
        if settings is None:
            raise ValueError("settings")

        if settings.instances is None:
            raise ValueError("settings.instances")

        instances: Dict[str, TInstance] = {}

        for alias, instance_item in settings.instances.items():
            instances[alias] = self.__single_instance_factory.create(instance_item)

        return instances

        