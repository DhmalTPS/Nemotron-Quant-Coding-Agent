# SPDX-License-Identifier: Apache-2.0
"""Deterministic extraction of the deliverable contract from instruction.md (and, when it is
mounted, the checker source). The contract drives local validation: a missing file, key,
column, wrong JSON type or wrong row count is caught before the grader sees it (gate g1).

Instruction JSON templates are pseudo-JSON ("<float>", 0.0XXXX, ...), so parsing is tolerant
and key-oriented rather than json.loads based.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .task import Task

OUT_EXT = ("json", "csv", "parquet", "pqt", "png", "html", "txt", "md", "xlsx", "tsv", "jsonl", "npy", "pkl", "svg", "pdf", "py")
FNAME_RE = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[A-Za-z0-9_][\w.-]*\.(?:%s))(?![\w])" % "|".join(OUT_EXT))
OUTPATH_RE = re.compile(r"(?:/app/output|/output|OUTPUT_DIR)/+([\w./-]+\.(?:%s))\b" % "|".join(OUT_EXT))
NEVER = {"reward.json", "pytest_report.json", "reward.txt", "card.toml", "manifest.json", "instruction.md"}


@dataclass
class FileSpec:
    name: str
    strong: bool = False
    json_keys: dict[str, str] = field(default_factory=dict)  # dotted path -> type hint
    dynamic_parents: set[str] = field(default_factory=set)  # dotted paths whose children are data-dependent
    columns: list[str] = field(default_factory=list)
    n_rows: int | None = None

    def describe(self) -> str:
        bits = [f"- {self.name}" + ("" if self.strong else "  (mentioned; verify whether required)")]
        if self.columns:
            bits.append(f"    columns (in order): {self.columns}")
        if self.n_rows is not None:
            bits.append(f"    expected rows: {self.n_rows}")
        if self.json_keys:
            ks = [f"{k}:{v}" if v else k for k, v in list(self.json_keys.items())[:80]]
            bits.append(f"    json keys: {ks}")
        if self.dynamic_parents:
            bits.append(f"    data-dependent keys under: {sorted(self.dynamic_parents)[:20]}")
        return "\n".join(bits)


@dataclass
class Contract:
    files: dict[str, FileSpec] = field(default_factory=dict)

    def get(self, name: str) -> FileSpec:
        if name not in self.files:
            self.files[name] = FileSpec(name=name)
        return self.files[name]

    def describe(self) -> str:
        if not self.files:
            return "(no deliverable filenames detected automatically; read the instruction carefully)"
        return "\n".join(f.describe() for f in self.files.values())

    @property
    def strong_files(self) -> list[str]:
        return [n for n, f in self.files.items() if f.strong]


def _type_hint(val: str) -> str:
    v = val.strip().strip(",").strip()
    vl = v.lower().strip('"').strip("'")
    if v.startswith("{"):
        return "object"
    if v.startswith("["):
        return "array"
    if re.search(r"<\s*int", vl) or vl in ("int", "integer"):
        return "int"
    if re.search(r"<\s*(float|number|double)", vl) or vl in ("float", "number"):
        return "float"
    if re.search(r"<\s*bool", vl) or vl in ("true", "false", "bool", "boolean", "true/false", "true|false"):
        return "bool"
    if re.search(r"<\s*(str|string|iso|date)", vl):
        return "str"
    if re.fullmatch(r"-?\d+", vl):
        return "int?"
    if re.fullmatch(r"-?\d*\.\d+(e-?\d+)?|-?\d+e-?\d+|-?0\.0?x+|-?0\.x+", vl.replace("X", "x")):
        return "float"
    if v.startswith('"'):
        return "str"
    return ""


def parse_json_template(block: str) -> tuple[dict[str, str], set[str]]:
    """Return {dotted.key: type_hint} and the set of parents with data-dependent children."""
    keys: dict[str, str] = {}
    dynamic: set[str] = set()
    stack: list[str | None] = []  # names of open containers (None = anonymous)
    pending: str | None = None
    dyn_re = re.compile(r"-?\d+(\.\d+)?|\d{4}-\d{2}(-\d{2})?|<[^>]*>|\.\.\.|[A-Z]{2,5}")
    i, n = 0, len(block)

    def path(extra: str | None = None) -> str:
        parts = [p for p in stack if p is not None and p != "\x00dyn"]
        if extra is not None:
            parts.append(extra)
        return ".".join(parts)

    while i < n:
        c = block[i]
        if c == '"':
            j = i + 1
            while j < n and block[j] != '"':
                j += 2 if block[j] == "\\" else 1
            s = block[i + 1 : j]
            k = j + 1
            while k < n and block[k] in " \t":
                k += 1
            if k < n and block[k] == ":":
                m = re.match(r":\s*([^\n]*)", block[k:])
                vtxt = m.group(1) if m else ""
                if not vtxt.lstrip().startswith(("{", "[")):
                    vtxt = re.split(r"[,}\]]", vtxt, maxsplit=1)[0]  # scalar: stop at the next separator
                if dyn_re.fullmatch(s):
                    if "\x00dyn" not in stack:
                        dynamic.add(path() or "<root>")
                    pending = None
                    if vtxt.lstrip().startswith(("{", "[")):
                        # children of a data-dependent key are not tracked: push a sentinel
                        k2 = k + 1
                        while k2 < n and block[k2] not in "{[":
                            k2 += 1
                        stack.append("\x00dyn")
                        i = k2 + 1
                        continue
                elif "\x00dyn" not in stack:
                    keys[path(s)] = _type_hint(vtxt)
                    pending = s
                i = k + 1
                continue
            i = j + 1
            continue
        if c in "{[":
            stack.append(pending)
            pending = None
        elif c in "}]":
            if stack:
                stack.pop()
            pending = None
        elif c == ",":
            pending = None
        i += 1
    return keys, dynamic


def _backticked(s: str) -> list[str]:
    return re.findall(r"`([^`\n]{1,80})`", s)


def extract(task: Task) -> Contract:
    text = task.instruction
    c = Contract()
    input_names = {f.path.name for f in task.files}
    lines = text.splitlines()

    def is_candidate(name: str) -> bool:
        base = name.split("/")[-1]
        if base in NEVER or base in input_names:
            return False
        if name.startswith(("data/", "environment/")):
            return False
        return True

    # 1) explicit output-dir paths -> strong
    for m in OUTPATH_RE.finditer(text):
        nm = m.group(1).lstrip("/")
        if is_candidate(nm):
            c.get(nm).strong = True

    # 2) filenames in headings / bullet leads / near the word output -> strong; others weak
    section_is_output = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            low = stripped.lower()
            section_is_output = any(w in low for w in ("output", "deliverable", "result", "file", "write", "save"))
            if any(w in low for w in ("input", "data descr", "provided", "given")) and not any(
                w in low for w in ("output", "deliverable")
            ):
                section_is_output = False
        for m in FNAME_RE.finditer(line):
            nm = m.group(1)
            if "/" in nm and not nm.startswith(("output/",)):
                nm_base = nm.split("/")[-1]
                if nm.startswith(("/app/", "app/", "/input", "input/")):
                    continue
                nm = nm_base
            if not is_candidate(nm):
                continue
            low = line.lower()
            strong = (
                stripped.startswith("#")
                or bool(re.match(r"^[-*\d.)\s]*\**`?" + re.escape(nm), stripped))
                or (section_is_output and re.search(r"\b(write|save|output|produce|create|generate)", low) is not None)
                or re.search(r"\b(write|save|produce|output)\w*\b[^.]{0,60}" + re.escape(nm), low) is not None
            )
            if nm.endswith(".py") and not strong:
                continue
            fs = c.get(nm)
            fs.strong = fs.strong or strong

    # 3) JSON templates and column lists, attributed to the closest preceding filename
    last_file: str | None = None
    in_code = False
    code_lang = ""
    buf: list[str] = []
    for idx, line in enumerate(lines):
        st = line.strip()
        if st.startswith("```"):
            if not in_code:
                in_code = True
                code_lang = st[3:].strip().lower()
                buf = []
            else:
                in_code = False
                block = "\n".join(buf)
                if last_file and last_file.endswith(".json") and (code_lang in ("json", "") and "{" in block):
                    keys, dyn = parse_json_template(block)
                    fs = c.get(last_file)
                    if keys or dyn:
                        fs.json_keys.update(keys)
                        fs.dynamic_parents |= dyn
                        fs.strong = True
                elif last_file and last_file.endswith((".csv", ".tsv")) and code_lang in ("csv", "", "text"):
                    first = block.strip().splitlines()[0] if block.strip() else ""
                    if first.count(",") >= 1 and not c.get(last_file).columns:
                        c.get(last_file).columns = [x.strip() for x in first.split(",")]
            continue
        if in_code:
            buf.append(line)
            continue
        for m in FNAME_RE.finditer(line):
            nm = m.group(1).split("/")[-1]
            if nm in c.files:
                last_file = nm
        if last_file and last_file.endswith((".csv", ".tsv", ".parquet", ".pqt", ".xlsx")):
            fs = c.get(last_file)
            m = re.search(r"\bcolumns?\b[^:]{0,40}:\s*(.+)", line, re.I)
            if m and not fs.columns:
                cols = _backticked(m.group(1))
                if len(cols) == 1 and "," in cols[0]:
                    cols = [x.strip() for x in cols[0].split(",")]
                if len(cols) >= 2:
                    fs.columns = cols
            # markdown table whose header starts with Column
            if st.startswith("|") and re.match(r"\|\s*(column|field|name)\s*\|", st, re.I) and not fs.columns:
                cols = []
                for l2 in lines[idx + 2 : idx + 80]:
                    if not l2.strip().startswith("|"):
                        break
                    cell = l2.strip().strip("|").split("|")[0].strip().strip("`").strip()
                    if cell:
                        cols.append(cell)
                if cols:
                    fs.columns = cols
            m = re.search(r"\b(\d{1,6})\s+rows\b", line, re.I)
            if m and fs.n_rows is None and re.search(r"(should|must|exactly|there (are|should|will)|contain|one row|rows \()", line, re.I):
                fs.n_rows = int(m.group(1))

    # 4) checker hints (only when checks/ is mounted): filenames joined to the output dir
    if task.checks_text:
        for m in re.finditer(r"""(?:OUTPUT_DIR|output_dir|OUT_DIR|out_dir|/app/output|/output)[^\n'"]{0,40}['"]([\w./-]+\.(?:%s))['"]""" % "|".join(OUT_EXT), task.checks_text):
            nm = m.group(1).split("/")[-1]
            if is_candidate(nm):
                c.get(nm).strong = True
        for m in re.finditer(r"""['"]/(?:app/)?output/([\w./-]+\.(?:%s))['"]""" % "|".join(OUT_EXT), task.checks_text):
            nm = m.group(1)
            if is_candidate(nm):
                c.get(nm).strong = True
    # weak .py / .md mentions are usually methodology, not deliverables
    for nm in list(c.files):
        fs = c.files[nm]
        if not fs.strong and nm.endswith((".md", ".txt")):
            del c.files[nm]
    return c
