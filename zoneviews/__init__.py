"""zoneviews: encoder-facing views (h-rep / v-rep / graph / rasterizations)
over the zones of ``prob_jobshop_ext``'s ``SymbolicJobShopEnv``.

This is a library + tests only: no RL, no neural networks, no training.

Import-path note
-----------------
The repository root (``RL-for-zones/``) contains a literal directory named
``pyudbm`` (the pyudbm *project* directory, itself containing the real
installed package at ``pyudbm/pyudbm/``). If the repository root ends up on
``sys.path`` *before* pyudbm's own editable-install path entry — which
happens whenever code runs with the repository root as the current working
directory, since Python puts ``''`` (CWD) at ``sys.path[0]`` — then
``import pyudbm`` resolves to the bare, ``__init__``-less repository-root
``pyudbm/`` directory as an empty PEP 420 namespace package, shadowing the
real one (confirmed empirically: ``pyudbm.Context`` becomes unavailable).

Since this package's tests and scripts are meant to be runnable with the
repository root as the working directory (the natural place to run
``pytest`` from), this module fixes ``sys.path`` on first import: it makes
sure the pyudbm project directory is inserted ahead of everything else, so
the *first* thing that resolves ``pyudbm`` is the real package. Once
``pyudbm`` is cached in ``sys.modules`` this is permanent for the process —
no other module in this repository needs to repeat this fix.
"""
from __future__ import annotations

import os
import sys

_ZONEVIEWS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_ZONEVIEWS_DIR)
_PYUDBM_PROJECT_DIR = os.path.join(_REPO_ROOT, "pyudbm")

if "pyudbm" not in sys.modules:
    if os.path.isdir(_PYUDBM_PROJECT_DIR) and _PYUDBM_PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PYUDBM_PROJECT_DIR)

import pyudbm  # noqa: E402  (must run after the sys.path fix above)

if not hasattr(pyudbm, "Context"):
    raise ImportError(
        "pyudbm imported as an empty namespace package (missing Context) — "
        "the repository-root pyudbm/ directory is shadowing the real "
        "package. This should not happen after zoneviews' sys.path fix; "
        f"pyudbm.__file__={getattr(pyudbm, '__file__', None)!r}, "
        f"sys.path[:3]={sys.path[:3]!r}."
    )

if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

del pyudbm
