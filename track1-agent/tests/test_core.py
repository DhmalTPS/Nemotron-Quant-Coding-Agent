"""Unit tests for the deterministic parts of the agent (run: python -m pytest tests -q)."""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import codeio  # noqa: E402
from agent.contract import parse_json_template  # noqa: E402
from agent.llm import HouseClient  # noqa: E402
from agent.validate import validate  # noqa: E402
from agent.contract import Contract, FileSpec  # noqa: E402


def test_patch_exact_and_fuzzy():
    code = "def f(x):\n    y = x + 1\n    return y\n"
    reply = "<<<<<<< SEARCH\n    y = x + 1\n=======\n    y = x + 2\n>>>>>>> REPLACE"
    new, n, failed = codeio.apply_patches(code, reply)
    assert n == 1 and not failed and "x + 2" in new
    # indentation differs in SEARCH: fuzzy match keeps the program's indentation
    reply2 = "<<<<<<< SEARCH\ny = x + 1\nreturn y\n=======\ny = x * 3\nreturn y\n>>>>>>> REPLACE"
    new2, n2, _ = codeio.apply_patches(code, reply2)
    assert n2 == 1 and "    y = x * 3" in new2


def test_patch_miss_reported():
    _, n, failed = codeio.apply_patches("a = 1\n", "<<<<<<< SEARCH\nb = 2\n=======\nb = 3\n>>>>>>> REPLACE")
    assert n == 0 and failed


def test_extract_code_largest_block():
    text = "intro\n```python\nx=1\n```\nmore\n```python\nimport os\nprint(os.getcwd())\n```"
    assert "import os" in codeio.extract_code(text)


def test_join_continuation_overlap():
    first = "```python\nimport numpy as np\nx = np.arange(10)\ny = x.su"
    cont = "y = x.sum()\nprint(y)\n```"
    joined = codeio.join_continuation(first, cont)
    assert "y = x.sum()" in joined and joined.count("import numpy") == 1
    assert codeio.syntax_error(joined) == ""


class _T:  # minimal Task stand-in for remap
    path_map = {"/app/data": "/input/environment/data", "/app/spy.csv": "/input/environment/data/spy.csv",
                "/input": "/input"}


def test_remap_paths():
    code = 'a = "/app/spy.csv"\nb = open("/app/data/x.json")\nc = "/app/output/results.json"\nd = "/app/database"\ne="/output"'
    out = codeio.remap_paths(code, _T(), "/app/output/.agent_stage")
    assert '"/input/environment/data/spy.csv"' in out
    assert '"/input/environment/data/x.json"' in out
    assert '"/app/output/.agent_stage/results.json"' in out
    assert '"/app/database"' in out  # not a mapped prefix
    assert '"/app/output/.agent_stage"' in out


def test_template_parser_types_and_dynamic():
    block = '''{
  "n_rows": 18,
  "flag": <bool>,
  "zero_rates": {"1": 0.0XXXXXX, "2": 0.0XXXXXX, ...},
  "stats": {"mean": "<float>", "count": "<int>"},
  "intermediates": {"x": {"value": <float>}}
}'''
    keys, dyn = parse_json_template(block)
    assert keys["flag"] == "bool"
    assert keys["stats.count"] == "int" and keys["stats.mean"] == "float"
    assert keys["intermediates.x.value"] == "float"
    assert "zero_rates" in dyn and not any(k.startswith("zero_rates.") for k in keys)


def test_validator_types(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"flag": 1, "count": 2.5, "extra": 1}))
    c = Contract(files={"results.json": FileSpec("results.json", strong=True, json_keys={"flag": "bool", "count": "int"})})
    rep = validate(tmp_path, c, None, [])
    joined = " ".join(rep.errors)
    assert "boolean" in joined and "integer" in joined
    assert any("extra" in w for w in rep.warnings)


def test_validator_missing_and_index(tmp_path):
    (tmp_path / "a.csv").write_text(",x,y\n0,1,2\n")
    c = Contract(files={"a.csv": FileSpec("a.csv", strong=True, columns=["x", "y", "z"], n_rows=2),
                        "b.json": FileSpec("b.json", strong=True)})
    rep = validate(tmp_path, c, None, [])
    joined = " ".join(rep.errors)
    assert "b.json" in joined and "unnamed index" in joined and "missing columns" in joined and "rows" in joined


def test_llm_parse_think_tags():
    r = HouseClient._parse(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "<think>abc</think>\nANSWER"}}]}))
    assert r.content == "ANSWER" and r.reasoning == "abc"
    r2 = HouseClient._parse(json.dumps({"choices": [{"finish_reason": "length", "message": {"content": "<think>never ends"}}]}))
    assert r2.content == "" and r2.truncated


def test_compare_dirs(tmp_path):
    from agent.compare import compare_dirs

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "r.json").write_text(json.dumps({"n": 7, "x": 1.0, "ok": True}))
    (b / "r.json").write_text(json.dumps({"n": 8, "x": 1.0 + 1e-12, "ok": True}))
    (a / "t.csv").write_text("k,v\na,1.0\nb,2.0\n")
    (b / "t.csv").write_text("k,v\na,1.0\nb,2.5\n")
    d = compare_dirs(a, b, ["r.json", "t.csv"])
    assert any("'n'" in x for x in d) and not any("'x'" in x for x in d)
    assert any("column 'v'" in x for x in d)
    assert compare_dirs(a, a, ["r.json", "t.csv"]) == []


def test_stable_sorts_transform():
    from agent.transform import stable_sorts

    src = (
        "df = df.sort_values('date')\n"
        "idx = np.argsort(x)\n"
        "o = x.argsort()\n"
        "df2 = df.sort_values(by=['a', 'b'],\n    ascending=False,\n)\n"
        "df3 = df.sort_values('a', kind='mergesort')\n"
        "s = df.sort_index()\n"
    )
    out, n = stable_sorts(src)
    assert n == 5
    assert "sort_values('date', kind=\"stable\")" in out
    assert 'np.argsort(x, kind="stable")' in out and 'x.argsort(kind="stable")' in out
    assert "kind='mergesort'" in out and out.count("kind=") == 6
    assert 'sort_index(kind="stable")' in out
    compile(out, "t", "exec")


def test_loop_detection_and_trim():
    from agent.llm import _looping, _trim_loop

    good = "a = 1\nb = 2\n" + "x = 0\n" * 3 + "print(a)\n"
    assert not _looping(good)
    looped = "edit\n>>>>>>> REPLACE\n" + ">>>>>>> REPLACE\n" * 40 + ">>>>"
    assert _looping(looped)
    t = _trim_loop(looped)
    assert t.count(">>>>>>> REPLACE") == 1 and t.startswith("edit")
    two = "head\n" + "foo\nbar\n" * 12 + "fo"
    assert _looping(two) and _trim_loop(two).count("foo") == 1


def test_validator_placeholder_zeros(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"a": 0.0, "b": 0.0, "c": 0.0, "d": 1.5}))
    rep = validate(tmp_path, Contract(files={"results.json": FileSpec("results.json", strong=True)}), None, [])
    assert any("placeholder" in e for e in rep.errors)
    (tmp_path / "results.json").write_text(json.dumps({"nan_count": 0, "b": 0.3, "c": 1.2, "d": 1.5}))
    rep = validate(tmp_path, Contract(files={"results.json": FileSpec("results.json", strong=True)}), None, [])
    assert not any("placeholder" in e for e in rep.errors)
