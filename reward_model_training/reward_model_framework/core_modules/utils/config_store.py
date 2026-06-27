"""Lightweight singleton holder for the active Hydra config.

Utilities use this to retrieve path overrides (e.g., static configs or reward
model locations) without requiring the full config to be passed around.
"""


class ConfigStore:
    _instance = None
    cfg = None

    @staticmethod
    def instance(cfg=None):
        if ConfigStore._instance is None and cfg is not None:
            store = ConfigStore()
            store.cfg = cfg
            ConfigStore._instance = store
        return ConfigStore._instance

