# SPDX-License-Identifier: Apache-2.0
"""Ground-truth hints for API errors, from the libraries actually installed in this image.

The model repairs from memory; when its memory of an API is wrong (observed: RandomState.t instead
of RandomState.standard_t, repeated through two repairs and two rewrites), more prompting does not
help. Introspecting the real objects does: for AttributeError / ImportError / ModuleNotFoundError /
unexpected-keyword TypeError, list the closest REAL names and signatures.
"""

from __future__ import annotations

import difflib
import importlib
import inspect
import re


def _resolve(dotted: str):
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except Exception:  # noqa: BLE001
            continue
        try:
            for p in parts[i:]:
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            return None
    return None


_ALIASES = {
    "DataFrame": "pandas.DataFrame", "Series": "pandas.Series", "Timestamp": "pandas.Timestamp",
    "DatetimeIndex": "pandas.DatetimeIndex", "Index": "pandas.Index", "ndarray": "numpy.ndarray",
    "Generator": "numpy.random.Generator", "RandomState": "numpy.random.RandomState",
    "DataFrameGroupBy": "pandas.core.groupby.DataFrameGroupBy", "SeriesGroupBy": "pandas.core.groupby.SeriesGroupBy",
    "Rolling": "pandas.core.window.rolling.Rolling", "DatetimeProperties": "pandas.core.indexes.accessors.DatetimeProperties",
}


def _close(name: str, options: list[str], n: int = 6) -> list[str]:
    pub = [o for o in options if not o.startswith("_")]
    low = name.lower()
    # token match first: 't' -> 'standard_t', 'cdf' -> 'logcdf'/'cdf_x', 'iteritems' parts...
    hits = [o for o in pub if low in o.lower().split("_") and o.lower() != low]
    hits += [h for h in difflib.get_close_matches(name, pub, n=n, cutoff=0.5) if h not in hits]
    hits += [o for o in pub if name.lower() in o.lower() and o not in hits][: n - len(hits)]
    return hits[:n]


def _sig(obj, name: str) -> str:
    try:
        return f"{name}{inspect.signature(getattr(obj, name))}"[:200]
    except Exception:  # noqa: BLE001
        return name


def hints(stderr: str) -> str:
    out: list[str] = []
    for m in re.finditer(r"AttributeError: '([\w.]+)' object has no attribute '(\w+)'", stderr):
        tname, attr = m.group(1), m.group(2)
        obj = _resolve(tname) or _resolve(_ALIASES.get(tname.split(".")[-1], ""))
        if obj is not None:
            close = _close(attr, dir(obj))
            out.append(f"'{tname}' has no attribute '{attr}'. Real attributes with similar names: "
                       + (", ".join(_sig(obj, c) for c in close) or "(none)"))
    for m in re.finditer(r"AttributeError: module '([\w.]+)' has no attribute '(\w+)'", stderr):
        mod = _resolve(m.group(1))
        if mod is not None:
            close = _close(m.group(2), dir(mod))
            out.append(f"module '{m.group(1)}' has no attribute '{m.group(2)}'. Similar real names: "
                       + (", ".join(close) or "(none)"))
    for m in re.finditer(r"ImportError: cannot import name '(\w+)' from '([\w.]+)'", stderr):
        mod = _resolve(m.group(2))
        if mod is not None:
            close = _close(m.group(1), dir(mod))
            out.append(f"'{m.group(2)}' does not export '{m.group(1)}'. Similar real names: " + (", ".join(close) or "(none)"))
    for m in re.finditer(r"ModuleNotFoundError: No module named '([\w.]+)'", stderr):
        import pkgutil

        mods = sorted({mi.name for mi in pkgutil.iter_modules()})
        close = _close(m.group(1).split(".")[0], mods)
        out.append(f"module '{m.group(1)}' is not installed (no internet, cannot install). Installed modules with "
                   f"similar names: {', '.join(close) or '(none)'}. Use numpy/pandas/scipy/statsmodels instead.")
    for m in re.finditer(r"TypeError: ([\w.]+)\(\) got an unexpected keyword argument '(\w+)'", stderr):
        fn = _resolve(m.group(1))
        if fn is not None:
            try:
                out.append(f"real signature: {m.group(1)}{inspect.signature(fn)}"[:300])
            except Exception:  # noqa: BLE001
                pass
    return "\n".join(dict.fromkeys(out))
