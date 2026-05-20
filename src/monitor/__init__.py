"""策略实时监控模块"""

__all__ = [
    "LiveEngine",
    "MonitorConfig",
    "MonitorState",
    "DataPoller",
    "ConsoleDashboard",
    "AlertManager",
    "AlertEvent",
    "AlertCallback",
    "LoggingCallback",
]


def __getattr__(name):
    _lazy_map = {
        "LiveEngine": ".live_engine",
        "MonitorConfig": ".config",
        "MonitorState": ".state",
        "DataPoller": ".poller",
        "ConsoleDashboard": ".dashboard",
        "AlertManager": ".alerts",
        "AlertEvent": ".alerts",
        "AlertCallback": ".alerts",
        "LoggingCallback": ".alerts",
    }
    if name in _lazy_map:
        import importlib
        module = importlib.import_module(_lazy_map[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {name!r} has no attribute {name!r}")
