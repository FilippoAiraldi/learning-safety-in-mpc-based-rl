import inspect
import numpy as np


class MPCDebug:
    def __init__(self) -> None:
        self._x_info: list[tuple[range, str]] = []
        self._g_info: list[tuple[range, str]] = []

    def x_describe(self, index: int) -> None:
        return self._describe(self._x_info, index)

    def g_describe(self, index: int) -> None:
        return self._describe(self._g_info, index)

    def _describe(self, info: list[tuple[range, str]], index: int) -> None:
        for range_, description in info:
            if index in range_:
                print(description)
                return
        raise ValueError(f'Index {index} not found.')

    def _register(self, group: str, name: str, dims: tuple[int, ...]) -> None:
        assert group in {'x', 'g'}, 'Invalid type specified.'

        # get info and flattened size and then build the description
        info = inspect.getframeinfo(inspect.stack()[2][0])
        type_ = 'Decision variable' if group == 'x' else 'Constraint'
        shape = 'x'.join(str(d) for d in dims)
        context = '; '.join(info.code_context).strip()
        description = \
            f'{type_} \'{name}\' of shape {shape} defined at\n' \
            f'  filename: {info.filename}\n' \
            f'  function: {info.function}:{info.lineno}\n' \
            f'  context:  {context}\n' \

        # append to list
        info = self._x_info if group == 'x' else self._g_info
        last = 0 if len(info) == 0 else info[-1][0].stop
        N = np.prod(dims)
        info.append((range(last, last + N), description))
