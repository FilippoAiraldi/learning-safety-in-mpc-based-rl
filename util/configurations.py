from typing import Optional, Type, TypeVar, Union


ConfigType = TypeVar('ConfigType')


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
