"""Load environment implementations only when their classes are requested.

Gym registrations already point to individual modules. Importing this package
or its utilities must also work without the optional MuJoCo installation.
"""

from importlib import import_module


_MODULE_CLASSES = {
    'ant': ('AntTest', 'AntWall', 'AntWallTest', 'AntWallBroken',
            'AntWallBrokenTest', 'AntCircle', 'AntCircleTest'),
    'half_cheetah': ('HalfCheetahTest', 'HalfCheetahWithObstacle', 'HalfCheetahEqual',
                     'HalfCheetahBackward', 'HalfCheetahWithPos', 'HalfCheetahWithPosTest'),
    'point': ('PointCircle', 'PointCircleTest', 'PointHalfCircle', 'PointHalfCircleTest',
              'PointObstacle', 'PointObstacleTest', 'PointObstacle2', 'PointObstacle2Test',
              'PointDS', 'PointDSTest', 'PointEllip', 'PointEllipTest'),
    'walker': ('Walker2dTest', 'WalkerWithPos', 'WalkerWithPosTest'),
}
_CLASS_MODULES = {name: module for module, names in _MODULE_CLASSES.items() for name in names}
__all__ = list(_CLASS_MODULES)


def __getattr__(name):
    if name not in _CLASS_MODULES:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module = import_module(f'{__name__}.{_CLASS_MODULES[name]}')
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
