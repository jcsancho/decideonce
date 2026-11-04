"""Static detector for Python agent code (M14 demo slice S1).

Finds LLM calls (any call whose dotted name matches `discovery.llm_calls`)
whose result is consumed as a closed choice: compared against two or more
string/enum literals (`==`, `in`, `match`/`case`, dict lookup with literal
keys), possibly after `.strip()`/`.lower()`/`json.loads(...)[key]`, or
annotated `Literal[...]`/an `Enum` subclass. stdlib `ast` only; nothing is
imported or executed. Intra-function analysis only.
"""
from __future__ import annotations

import ast

NORMALIZE_METHODS = {"strip", "lower"}
ENUM_BASES = {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"}
LITERAL_NAMES = {"Literal"}


def dotted_path(node) -> str | None:
    """Dotted name of a Name/Attribute chain, else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_path(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def is_llm_call(node: ast.Call, llm_calls: list[str]) -> bool:
    path = dotted_path(node.func)
    if not path:
        return False
    return any(path == entry or path.endswith("." + entry)
               for entry in llm_calls)


def _unwrap(value):
    while isinstance(value, ast.Await):
        value = value.value
    return value


def _llm_root(value, llm_calls: list[str]) -> bool:
    """True when `value` is an LLM call, possibly normalized at assignment
    (`x = llm(...).strip()`, `x = json.loads(llm(...))[key]`)."""
    value = _unwrap(value)
    if isinstance(value, ast.Call):
        if is_llm_call(value, llm_calls):
            return True
        func = value.func
        if isinstance(func, ast.Attribute) \
                and func.attr in NORMALIZE_METHODS:
            return _llm_root(func.value, llm_calls)
        path = dotted_path(func)
        if path and (path == "json.loads" or path.endswith(".json.loads")
                     or path == "loads"):
            return any(_llm_root(a, llm_calls) for a in value.args)
        return False
    if isinstance(value, ast.Subscript):
        return _llm_root(value.value, llm_calls)
    return False


def _llm_call_name(value) -> str | None:
    """Dotted name of the LLM call at the root of a normalized value."""
    value = _unwrap(value)
    if isinstance(value, ast.Call):
        if dotted_path(value.func) and not (
                isinstance(value.func, ast.Attribute)
                and value.func.attr in NORMALIZE_METHODS):
            return dotted_path(value.func)
        if isinstance(value.func, ast.Attribute) \
                and value.func.attr in NORMALIZE_METHODS:
            return _llm_call_name(value.func.value)
        for a in value.args:
            name = _llm_call_name(a)
            if name:
                return name
        return dotted_path(value.func)
    if isinstance(value, ast.Subscript):
        return _llm_call_name(value.value)
    return None


def module_enums(tree: ast.Module) -> dict[str, list[str]]:
    """Enum subclasses defined in this module -> member names in order."""
    out: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [dotted_path(b) for b in node.bases]
        if not any(b and b.split(".")[-1] in ENUM_BASES for b in bases):
            continue
        members: list[str] = []
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name) and t.id not in members:
                        members.append(t.id)
            elif isinstance(stmt, ast.AnnAssign) \
                    and isinstance(stmt.target, ast.Name):
                if stmt.target.id not in members:
                    members.append(stmt.target.id)
        if members:
            out[node.name] = members
    return out


def literal_values(ann) -> list[str] | None:
    """String values of a Literal[...] annotation, else None."""
    if not isinstance(ann, ast.Subscript):
        return None
    base = dotted_path(ann.value)
    if not base or base.split(".")[-1] not in LITERAL_NAMES:
        return None
    elts = ann.slice.elts if isinstance(ann.slice, ast.Tuple) \
        else [ann.slice]
    values = []
    for e in elts:
        if isinstance(e, ast.Constant) and isinstance(e.value, str):
            values.append(e.value)
        else:
            return None
    return values


def _target_names(target) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for e in target.elts:
            names.extend(_target_names(e))
        return names
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return []


def _dict_literal_keys(node) -> list[str] | None:
    """String keys of a dict literal with 1+ string keys, else None."""
    if not isinstance(node, ast.Dict):
        return None
    keys = []
    for k in node.keys:
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            keys.append(k.value)
        else:
            return None
    return keys or None


class _SiteVisitor(ast.NodeVisitor):
    """Collect LLM call sites and assignments inside one function."""

    def __init__(self, llm_calls: list[str],
                 module_dicts: dict[str, list[str]] | None = None):
        self.llm_calls = llm_calls
        self.sites: list[dict] = []  # {call, targets, ann_options}
        self.assigns: list[tuple] = []  # (targets, value, lineno)
        self.dict_names: dict[str, list[str]] = dict(module_dicts or {})
        self.matches: list = []
        self.compares: list = []
        self.subscripts: list = []
        self.get_calls: list = []

    def visit_Assign(self, node: ast.Assign):
        self._record_assign(node.targets, node.value, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.value is not None:
            self._record_assign([node.target], node.value, node.lineno,
                                ann=node.annotation)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr):
        self._record_assign([node.target], node.value, node.lineno)
        self.generic_visit(node)

    def _record_assign(self, targets, value, lineno, ann=None):
        value = _unwrap(value)
        names: list[str] = []
        for t in targets:
            names.extend(_target_names(t))
        self.assigns.append((names, value, lineno))
        keys = _dict_literal_keys(value)
        if keys:
            for n in names:
                self.dict_names[n] = keys
        if _llm_root(value, self.llm_calls):
            ann_options = literal_values(ann) if ann is not None else None
            self.sites.append({"call": value, "targets": names,
                               "ann_options": ann_options,
                               "ann_node": ann, "lineno": value.lineno,
                               "func": _llm_call_name(value)})

    def visit_Compare(self, node: ast.Compare):
        self.compares.append(node)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match):
        self.matches.append(node)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        self.subscripts.append(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "get" and node.args:
            self.get_calls.append(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        pass  # nested scope: analyzed separately, never merged upward

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        pass

    def visit_ClassDef(self, node):
        pass


def _tainted_expr(node, tainted: set[str]) -> bool:
    node = _unwrap(node)
    if isinstance(node, ast.Name):
        return node.id in tainted
    if isinstance(node, ast.Subscript):
        return _tainted_expr(node.value, tainted)
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) \
                and func.attr in NORMALIZE_METHODS:
            return _tainted_expr(func.value, tainted)
        path = dotted_path(func)
        if path and (path == "json.loads" or path.endswith(".json.loads")
                     or path == "loads"):
            return any(_tainted_expr(a, tainted) for a in node.args)
    return False


def _propagate(assigns: list[tuple], seeds: set[str]) -> set[str]:
    tainted = set(seeds)
    for _ in range(len(assigns) + 1):
        grown = False
        for names, value, _ in assigns:
            if not names or all(n in tainted for n in names):
                continue
            if _tainted_expr(value, tainted):
                tainted.update(names)
                grown = True
        if not grown:
            break
    return tainted


def _const_str(node) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _enum_literal(node) -> str | None:
    if isinstance(node, ast.Attribute):
        return dotted_path(node)
    return None


def _str_tuple(node) -> list[str] | None:
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return None
    values = []
    for e in node.elts:
        s = _const_str(e)
        if s is None:
            return None
        values.append(s)
    return values


def _collect_literals(vis: _SiteVisitor, tainted: set[str]) -> tuple[list[str], list[dict]]:
    """Comparison literals over tainted names, in source order, with evidence."""
    found: list[tuple[str, dict]] = []

    def add(value: str, pattern: str, lineno: int):
        found.append((value, {"pattern": pattern, "line": lineno,
                              "literal": value}))

    for node in vis.compares:
        operands = [node.left] + list(node.comparators)
        if all(isinstance(op, ast.Eq) for op in node.ops):
            if any(_tainted_expr(o, tainted) for o in operands):
                for o in operands:
                    s = _const_str(o)
                    if s is not None:
                        add(s, "if_eq", node.lineno)
                    else:
                        e = _enum_literal(o)
                        if e is not None:
                            add(e, "if_eq", node.lineno)
        elif len(node.ops) == 1 and isinstance(node.ops[0], ast.In):
            left, right = node.left, node.comparators[0]
            if not _tainted_expr(left, tainted):
                continue
            vals = _str_tuple(right)
            if vals is not None:
                for v in vals:
                    add(v, "in", node.lineno)
            elif isinstance(right, ast.Name) and right.id in vis.dict_names:
                for v in vis.dict_names[right.id]:
                    add(v, "in_dict", node.lineno)
    for node in vis.matches:
        if not _tainted_expr(node.subject, tainted):
            continue
        for case in node.cases:
            pat = case.pattern
            if isinstance(pat, ast.MatchValue):
                s = _const_str(pat.value)
                if s is not None:
                    add(s, "match", getattr(pat, "lineno", node.lineno))
                else:
                    e = _enum_literal(pat.value)
                    if e is not None:
                        add(e, "match", getattr(pat, "lineno", node.lineno))
    for node in vis.subscripts:
        if not _tainted_expr(node.slice, tainted):
            continue
        keys = _dict_literal_keys(node.value)
        if keys is None and isinstance(node.value, ast.Name):
            keys = vis.dict_names.get(node.value.id)
        if keys:
            for v in keys:
                add(v, "dict", node.lineno)
    for node in vis.get_calls:
        target = node.func.value  # type: ignore[attr-defined]
        keys = _dict_literal_keys(target)
        if keys is None and isinstance(target, ast.Name):
            keys = vis.dict_names.get(target.id)
        if keys and _tainted_expr(node.args[0], tainted):
            for v in keys:
                add(v, "dict_get", node.lineno)
    options: list[str] = []
    evidence: list[dict] = []
    for value, ev in found:
        evidence.append(ev)
        if value not in options:
            options.append(value)
    return options, evidence


def find_file_candidates(path: str, text: str, cfg: dict) -> list[dict]:
    """Static candidates in one Python file (parse errors -> no candidates)."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    llm_calls = cfg.get("discovery", {}).get("llm_calls", [])
    enums = module_enums(tree)
    module_dicts: dict[str, list[str]] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            pairs = [(stmt.targets, stmt.value)]
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            pairs = [([stmt.target], stmt.value)]
        else:
            continue
        for targets, value in pairs:
            keys = _dict_literal_keys(_unwrap(value))
            if keys:
                for t in targets:
                    module_dicts.update({n: keys
                                         for n in _target_names(t)})
    out: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        vis = _SiteVisitor(llm_calls, module_dicts)
        for stmt in node.body:
            vis.visit(stmt)
        for site in vis.sites:
            seeds = set(site["targets"])
            if not seeds:
                continue
            tainted = _propagate(vis.assigns, seeds)
            options, evidence = _collect_literals(vis, tainted)
            ann_options = list(site["ann_options"] or [])
            ann = site["ann_node"]
            if not ann_options and ann is not None:
                enum_name = dotted_path(ann)
                if enum_name and enum_name.split(".")[-1] in enums:
                    ann_options = list(enums[enum_name.split(".")[-1]])
                    evidence.append({"pattern": "enum", "line": site["lineno"],
                                     "literal": enum_name})
            if ann_options:
                evidence.append({"pattern": "annotation",
                                 "line": site["lineno"],
                                 "literal": ",".join(ann_options)})
                options = ann_options + [o for o in options
                                         if o not in ann_options]
            if len(options) >= 2:
                out.append({"step": node.name,
                            "location": f"{path}:{site['lineno']}",
                            "options": options,
                            "evidence": {"call": site["func"],
                                         "matches": evidence}})
    out.sort(key=lambda c: (c["step"], c["location"]))
    return out


def find_static_candidates(code: list[dict], cfg: dict) -> list[dict]:
    """Static candidates over [{path, text}] files."""
    out: list[dict] = []
    for entry in code or []:
        if not isinstance(entry, dict):
            continue
        out.extend(find_file_candidates(str(entry.get("path", "")),
                                        str(entry.get("text", "")), cfg))
    return out
