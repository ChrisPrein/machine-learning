from typing import Callable, Dict, Generic, Type, Union

from .abstractions.single_instance_factory import SingleInstanceFactory, TInstance
from .abstractions.multi_instance_factory import MultiInstanceFactory, TInstance
from .default_single_instance_settings import DefaultSignleInstanceSettings
from .default_multi_instance_settings import DefaultMultiInstanceSettings
from .default_single_instance_factory import DefaultSingleInstanceFactory

class DefaultMultiInstanceFactory(Generic[TInstance], MultiInstanceFactory[DefaultMultiInstanceSettings, TInstance]):
    def __init__(self, available_types: Dict[str, Union[Type[TInstance], Callable[[DefaultSignleInstanceSettings], TInstance]]]):
        if available_types is None:
            raise ValueError("available_types")

        self.__single_instance_factory: SingleInstanceFactory[DefaultSignleInstanceSettings, TInstance] = DefaultSingleInstanceFactory[TInstance](available_types)

    def create(self, settings: DefaultMultiInstanceSettings) -> Dict[str, TInstance]:
        if settings is None:
            raise ValueError("settings")

        if settings.instances is None:
            raise ValueError("settings.instances")

        instances: Dict[str, TInstance] = {}

        for alias, instance_settings in settings.instances.items():
            instances[alias] = self.__single_instance_factory.create(instance_settings)

        return instances

        