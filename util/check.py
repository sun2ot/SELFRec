from numba import jit

def chain(*validators):
    def wrapped(lst, name=''):
        for validator in validators:
            validator(lst, name)
    return wrapped


@jit(nopython=True)
def has_None(lst: list, name: str='') -> None:
    if any(item is None for item in lst):
        raise ValueError(f"{name} contains None values.")


@jit(nopython=True)
def is_str(lst: list, name: str='') -> None:
    if not all(isinstance(item, str) for item in lst):
        raise TypeError(f"{name} contains non-string elements.")