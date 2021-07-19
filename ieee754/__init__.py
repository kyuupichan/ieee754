from .ieee754 import *

version = 'ieee754 0.01'
version_short = version.split()[-1]

__all__ = sum((
    ieee754.__all__,
), ())
