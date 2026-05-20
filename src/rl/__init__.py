"""强化学习截面选股模块"""

__all__ = [
    'RLConfig',
    'DataAdapter',
    'FactorPanelBuilder',
    'RLInference',
    'RLStrategy',
    'EnvFactory',
    'RLTrainer',
]


def __getattr__(name):
    _lazy_map = {
        'RLConfig': '.config',
        'DataAdapter': '.data_adapter',
        'FactorPanelBuilder': '.factor_panel',
        'RLInference': '.inference',
        'RLStrategy': '.rl_strategy',
        'EnvFactory': '.env_factory',
        'RLTrainer': '.trainer',
    }
    if name in _lazy_map:
        import importlib
        module = importlib.import_module(_lazy_map[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
