from abc import ABC
from typing import Any, Optional, Type, TypeVar, Union


class BaseConfig(ABC):
    '''Base abstract class for configurations.'''

    def get_group(
        self,
        group: str
    ) -> dict[str, Any]:
        '''
        Gets a group of parameters starting wit the name `group_`, where
        `group` is the given string.
        '''
        return {
            name.removeprefix(f'{group}_'): val
            for name, val in self.__dict__.items()
            if name.startswith(f'{group}_')
        }


ConfigType = TypeVar('ConfigType', bound=BaseConfig)


def init_config(
    config: Optional[Union[ConfigType, dict]],
    cls: Type[ConfigType]
) -> ConfigType:
    '''
    Initializes the configuration, by
        - returning it if valid
        - converting from a dict to a dataclass
        - instantiating the default configuration.
    '''
    if config is None:
        return cls()

    if isinstance(config, cls):
        return config

    if isinstance(config, dict):
        if not hasattr(cls, '__dataclass_fields__'):
            raise ValueError('Configiration class must be a dataclass.')
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: config[k] for k in keys if k in config})

    raise ValueError('Invalid configuration type; expected None, dict or '
                     f'a dataclass, got {cls} instead.')
