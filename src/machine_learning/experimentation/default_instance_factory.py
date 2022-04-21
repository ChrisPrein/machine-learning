from typing import Callable, Dict, Generic, Type, Union

from .abstractions.instance_factory import InstanceFactory, TInstance
from .default_instance_settings import DefaultInstanceSettings

class DefaultInstanceFactory(Generic[TInstance], InstanceFactory[DefaultInstanceSettings, TInstance]):
    def __init__(self, available_types: Dict[str, Union[Type[TInstance], Callable[[DefaultInstanceSettings], TInstance]]]):
        if available_types is None:
            raise ValueError("available_types")

        self.__available_types: Dict[str, Union[Type[TInstance], Callable[[DefaultInstanceSettings], TInstance]]] = available_types

    def create(self, settings: DefaultInstanceSettings) -> TInstance:
        if settings is None:
            raise ValueError("settings")

        if settings.name is None:
            raise ValueError("settings.name")

        if settings.params is None:
            raise ValueError("settings.params")

        if not settings.name in self.__available_types:
            raise KeyError(f'No matching type with name {settings.name} found.')

        type: Union[Type[TInstance], Callable[[DefaultInstanceSettings], TInstance]] = self.__available_types[settings.name]

        return type(**settings.params)