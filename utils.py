# https://stackoverflow.com/questions/7811247/how-to-fill-specific-positional-arguments-with-partial-in-python
from functools import partial
from typing import Any


class bind[T](partial[T]):
    """
    An improved version of partial which accepts Ellipsis (...) as a placeholder
    """

    def __call__(self, *args: Any, **keywords: Any):
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        processed_args = (next(iargs) if arg is ... else arg for arg in self.args)
        return self.func(*processed_args, *iargs, **keywords)
