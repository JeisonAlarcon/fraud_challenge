"""Compatibilidad con pickles guardados cuando el paquete se llamaba ``fraud_challenge``.

Los ``.pkl`` previos (modelos, ``preprocessor.pkl``, ``calibrator.pkl``) referencian rutas como
``fraud_challenge.features.registry``. Tras aplanar ``src/`` esas rutas ya no existen; este helper
registra alias en ``sys.modules`` para que sigan cargando sin reentrenar. Se puede borrar cuando
todos los artefactos se regeneren con las rutas nuevas.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import types

_PACKAGES = ("data", "features", "models", "training", "evaluation", "serving")
_LEGACY = "fraud_challenge"


def install_legacy_pickle_aliases() -> None:
    legacy_root = sys.modules.get(_LEGACY)
    if legacy_root is None:
        legacy_root = types.ModuleType(_LEGACY)
        legacy_root.__path__ = []  # type: ignore[attr-defined]
        sys.modules[_LEGACY] = legacy_root
    for pkg_name in _PACKAGES:
        pkg = importlib.import_module(pkg_name)
        sys.modules[f"{_LEGACY}.{pkg_name}"] = pkg
        setattr(legacy_root, pkg_name, pkg)
        for info in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg_name}."):
            try:
                module = importlib.import_module(info.name)
            except ImportError:
                continue
            sys.modules[f"{_LEGACY}.{info.name}"] = module
    config = importlib.import_module("config")
    legacy_root.__version__ = config.__version__  # type: ignore[attr-defined]
    legacy_root.FEATURE_PIPELINE_CODE_VERSION = config.FEATURE_PIPELINE_CODE_VERSION  # type: ignore[attr-defined]
