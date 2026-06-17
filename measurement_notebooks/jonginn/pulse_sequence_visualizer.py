"""
pulse_sequence_visualizer.py

Hybrid pulse-sequence visualizer for QICK/slab-style experiment code.

It has two independent parts:

1) Static scanner: reads Python source with ast, finds Program classes and their
   inheritance tree, and summarizes calls such as self.pulse(), self.custom_pulse(),
   self.sync_all(), self.measure(), self.play_parity_pulse(), etc.  This does not
   import the experiment package and therefore works even when dependencies are not
   installed.

2) Runtime tracer: creates a traced subclass of a Program class and intercepts QICK
   timing/pulse methods while the Program's initialize()/body() are executed.  This
   gives a timeline that can be plotted with matplotlib.  It is intended for a
   dry-run view of the program for one concrete cfg; branch choices are whatever
   your cfg selects.

Typical use in a notebook:

    from pulse_sequence_visualizer import discover_program_classes, trace_program_class
    from experiments.qsim.sideband_scramble import SidebandScrambleProgram

    idx = discover_program_classes("/path/to/repo/experiments")
    idx.print_tree("QsimBaseProgram")

    prog, trace = trace_program_class(SidebandScrambleProgram, soccfg=soccfg, cfg=cfg)
    fig, ax = trace.plot(show_sync=True, annotate=True)
    trace.to_csv("sideband_scramble_trace.csv")

For the pasted DarkBaseProgram family:

    from dark_code import DarkT1Program
    prog, trace = trace_program_class(DarkT1Program, soccfg=soccfg, cfg=cfg)
    trace.plot()

Limitations:
- Runtime tracing is cfg-dependent; it draws the branch that actually executes.
- Lengths are best-effort. If the real QICK conversion is not used, us2cycles()
  defaults to identity, so the x-axis is effectively microseconds when your code
  passes microseconds into us2cycles().
- For flat_top pulses, duration is approximated as flat length + envelope length.
- Conditional jumps are annotated as markers, not as alternate branch timelines.
"""
from __future__ import annotations

import ast
import csv
import importlib
import inspect
import json
import math as stdlib_math
import re
import sys
import textwrap
import traceback
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, Union
from copy import deepcopy


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _basename(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def _is_self_attr_call(node: ast.Call) -> Optional[str]:
    """Return method name for calls like self.foo(...), otherwise None."""
    f = node.func
    if not isinstance(f, ast.Attribute):
        return None
    root = f.value
    while isinstance(root, ast.Attribute):
        root = root.value
    if isinstance(root, ast.Name) and root.id == "self":
        return f.attr
    return None


def _is_super_attr_call(node: ast.Call) -> Optional[str]:
    """Return method name for calls like super().foo(...), otherwise None."""
    f = node.func
    if not isinstance(f, ast.Attribute):
        return None
    root = f.value
    if not isinstance(root, ast.Call):
        return None
    callee = root.func
    if isinstance(callee, ast.Name) and callee.id == "super":
        return f.attr
    return None


def _is_program_method_call(node: ast.Call) -> Optional[str]:
    """Return method name for calls on self or super()."""
    return _is_self_attr_call(node) or _is_super_attr_call(node)


def _short_value(x: Any, maxlen: int = 140) -> str:
    try:
        s = repr(x)
    except Exception:
        s = f"<{type(x).__name__}>"
    if len(s) > maxlen:
        s = s[: maxlen - 3] + "..."
    return s


def _try_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
        return float(x)
    except Exception:
        return default


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]


def _nested_get(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        try:
            if isinstance(cur, Mapping):
                cur = cur[part]
            else:
                cur = getattr(cur, part)
        except Exception:
            return default
    return cur


def import_object(path: str) -> Any:
    """Import 'package.module:Name' or 'package.module.Name'."""
    if ":" in path:
        mod_name, obj_name = path.split(":", 1)
    else:
        mod_name, obj_name = path.rsplit(".", 1)
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as exc:
        if exc.name != "qick":
            raise
        install_qick_stubs()
        mod = importlib.import_module(mod_name)
    return getattr(mod, obj_name)


def install_qick_stubs(*, force: bool = False) -> None:
    """
    Install a tiny in-process qick module for visualization-only imports.

    The experiment modules import qick at module-import time, but the static
    scanner and dry-run tracer do not need real RFSoC hardware bindings.  This
    stub is intentionally minimal: it lets Program classes import and lets the
    PulseTraceMixin intercept make_program()/pulse/timing methods.
    """
    if "qick" in sys.modules and not force:
        return

    qick_mod = ModuleType("qick")
    helpers_mod = ModuleType("qick.helpers")

    class QickConfig(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def cycles2us(self, cycles, *args, **kwargs):
            return cycles

        def us2cycles(self, us, *args, **kwargs):
            return us

    class _QickStubProgram:
        def __init__(self, soccfg=None, cfg=None, *args, **kwargs):
            self.soccfg = soccfg
            self.cfg = cfg
            if hasattr(self, "make_program"):
                self.make_program()

        def acquire(self, *args, **kwargs):
            raise RuntimeError("qick stub cannot acquire data; it is for pulse visualization only")

    def gauss(mu=0, si=1, length=1, maxv=32767):
        n = max(int(length), 0)
        try:
            import numpy as np  # type: ignore

            x = np.arange(n)
            sigma = float(si) if si else 1.0
            return maxv * np.exp(-0.5 * ((x - float(mu)) / sigma) ** 2)
        except Exception:
            return [0.0] * n

    for cls_name in (
        "AveragerProgram",
        "RAveragerProgram",
        "NDAveragerProgram",
        "QickProgram",
    ):
        setattr(qick_mod, cls_name, _QickStubProgram)
    qick_mod.QickConfig = QickConfig
    helpers_mod.gauss = gauss
    qick_mod.helpers = helpers_mod

    sys.modules["qick"] = qick_mod
    sys.modules["qick.helpers"] = helpers_mod


# -----------------------------------------------------------------------------
# Static class scanner
# -----------------------------------------------------------------------------

PULSE_PRIMITIVE_CALLS = {
    # Low-level QICK timing/pulse/measurement/register calls. Higher-level lab
    # wrappers are inferred from the source call graph instead of listed here.
    "pulse", "setup_and_pulse", "set_pulse_registers", "default_pulse_registers",
    "measure", "setup_and_measure", "trigger", "readout", "read",
    "sync_all", "wait_all", "synci", "waiti",
    "condj", "label", "loopnz", "end",
    "math", "mathi", "bitwi", "safe_regwi", "regwi", "memwi",
    "declare_gen", "declare_readout", "declare_readout_ch", "declare",
    "set_readout_registers", "default_readout_registers",
    "add_gauss", "add_DRAG", "add_triangle", "add_pulse", "add_envelope",
    "reset_phase", "set", "seti",
}

# Backward-compatible public name. These are seed calls; discover_program_classes()
# expands them through self-call wrappers found in the scanned source tree.
DEFAULT_PULSE_CALLS = set(PULSE_PRIMITIVE_CALLS)

PROGRAM_BASE_HINTS = {
    # QICK root Program classes. Local bases such as DarkBaseProgram are inferred
    # transitively when their source is included in the scan.
    "AveragerProgram", "RAveragerProgram", "NDAveragerProgram", "QickProgram",
}


@dataclass
class StaticCall:
    class_name: str
    method_name: str
    call_name: str
    filename: str
    lineno: int
    args: str = ""
    condition_stack: Tuple[str, ...] = ()

    def format(self) -> str:
        cond = ""
        if self.condition_stack:
            cond = "  under: " + " / ".join(self.condition_stack)
        return f"{self.filename}:{self.lineno}  {self.class_name}.{self.method_name} -> self.{self.call_name}({self.args}){cond}"


@dataclass
class ProgramClassInfo:
    name: str
    bases: Tuple[str, ...]
    filename: str
    lineno: int
    methods: Tuple[str, ...]
    calls: Tuple[StaticCall, ...] = ()

    @property
    def short_bases(self) -> Tuple[str, ...]:
        return tuple(_basename(b) for b in self.bases)


@dataclass
class _RawClassInfo:
    name: str
    bases: Tuple[str, ...]
    filename: str
    lineno: int
    methods: Tuple[str, ...]
    method_nodes: Dict[str, ast.FunctionDef]
    method_self_calls: Dict[str, Tuple[str, ...]]


class _SelfCallCollector(ast.NodeVisitor):
    """Collect method names for calls like self.foo(...) or super().foo(...)."""

    def __init__(self):
        self.calls: List[str] = []

    def visit_Call(self, node: ast.Call) -> Any:
        name = _is_program_method_call(node)
        if name:
            self.calls.append(name)
        self.generic_visit(node)


def _self_calls_in_method(node: ast.FunctionDef) -> Tuple[str, ...]:
    visitor = _SelfCallCollector()
    visitor.visit(node)
    return tuple(visitor.calls)


def _infer_program_class_names(
    raw_classes: Sequence[_RawClassInfo],
    base_hints: Iterable[str],
    *,
    include_all_program_suffix: bool,
) -> set:
    """
    Infer Program-like classes from the AST inheritance graph.

    A class is Program-like if it directly or transitively inherits one of the
    seed base names. The suffix fallback is kept for partial scans where the
    base class definition is outside the scanned roots.
    """
    hints = {_basename(str(name)) for name in base_hints}
    program_names = set()
    changed = True
    while changed:
        changed = False
        for cls in raw_classes:
            short_bases = {_basename(base) for base in cls.bases}
            is_program = (
                cls.name in hints
                or bool(short_bases & hints)
                or bool(short_bases & program_names)
            )
            if include_all_program_suffix and cls.name.endswith("Program"):
                is_program = True
            if is_program and cls.name not in program_names:
                program_names.add(cls.name)
                changed = True
    return program_names


def _infer_pulse_call_names(raw_classes: Sequence[_RawClassInfo], seed_calls: Iterable[str]) -> set:
    """
    Infer all source-level wrapper methods that eventually call pulse primitives.

    This is deliberately name-based across the scanned tree. It lets static scan
    follow local wrappers such as play_foo() -> prepare_foo() -> pulse() without
    manually maintaining a list for every Program family.
    """
    pulse_like = {_basename(str(name)) for name in seed_calls}
    method_calls_by_name: Dict[str, set] = defaultdict(set)
    for cls in raw_classes:
        for method_name, calls in cls.method_self_calls.items():
            method_calls_by_name[method_name].update(calls)

    changed = True
    while changed:
        changed = False
        for method_name, calls in method_calls_by_name.items():
            if method_name not in pulse_like and calls & pulse_like:
                pulse_like.add(method_name)
                changed = True
    return pulse_like


class _PulseCallVisitor(ast.NodeVisitor):
    def __init__(self, class_name: str, method_name: str, filename: str, target_calls: set):
        self.class_name = class_name
        self.method_name = method_name
        self.filename = filename
        self.target_calls = target_calls
        self.calls: List[StaticCall] = []
        self.condition_stack: List[str] = []

    def visit_If(self, node: ast.If) -> Any:
        self.condition_stack.append("if " + _safe_unparse(node.test))
        for stmt in node.body:
            self.visit(stmt)
        self.condition_stack.pop()
        if node.orelse:
            self.condition_stack.append("else of " + _safe_unparse(node.test))
            for stmt in node.orelse:
                self.visit(stmt)
            self.condition_stack.pop()

    def visit_For(self, node: ast.For) -> Any:
        self.condition_stack.append("for " + _safe_unparse(node.target) + " in " + _safe_unparse(node.iter))
        for stmt in node.body:
            self.visit(stmt)
        self.condition_stack.pop()
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_While(self, node: ast.While) -> Any:
        self.condition_stack.append("while " + _safe_unparse(node.test))
        for stmt in node.body:
            self.visit(stmt)
        self.condition_stack.pop()
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_Call(self, node: ast.Call) -> Any:
        name = _is_program_method_call(node)
        if name in self.target_calls:
            arg_texts = [_safe_unparse(a) for a in node.args]
            arg_texts += [f"{kw.arg}={_safe_unparse(kw.value)}" for kw in node.keywords if kw.arg is not None]
            args = ", ".join(arg_texts)
            if len(args) > 220:
                args = args[:217] + "..."
            self.calls.append(
                StaticCall(
                    class_name=self.class_name,
                    method_name=self.method_name,
                    call_name=name,
                    filename=self.filename,
                    lineno=node.lineno,
                    args=args,
                    condition_stack=tuple(self.condition_stack),
                )
            )
        self.generic_visit(node)


class ProgramIndex:
    """Static index of Program-like classes found in a source tree."""

    def __init__(
        self,
        classes: Sequence[ProgramClassInfo],
        *,
        pulse_call_names: Optional[Iterable[str]] = None,
        program_class_names: Optional[Iterable[str]] = None,
    ):
        self.classes = list(classes)
        self.by_name: Dict[str, List[ProgramClassInfo]] = defaultdict(list)
        for c in self.classes:
            self.by_name[c.name].append(c)
        self.pulse_call_names = set(pulse_call_names or [])
        self.program_class_names = set(program_class_names or [c.name for c in self.classes])

    def program_classes(self) -> List[ProgramClassInfo]:
        return list(self.classes)

    def direct_children(self, base_name: str) -> List[ProgramClassInfo]:
        b = _basename(base_name)
        out = []
        for c in self.classes:
            if any(_basename(x) == b for x in c.bases):
                out.append(c)
        return sorted(out, key=lambda c: (c.filename, c.lineno, c.name))

    def descendants(self, base_name: str, include_base: bool = False) -> List[ProgramClassInfo]:
        seen = set()
        out: List[ProgramClassInfo] = []
        q = deque([_basename(base_name)])
        if include_base:
            for c in self.by_name.get(_basename(base_name), []):
                out.append(c)
                seen.add(c.name)
        while q:
            b = q.popleft()
            for child in self.direct_children(b):
                if child.name not in seen:
                    seen.add(child.name)
                    out.append(child)
                    q.append(child.name)
        return out

    def calls_for(self, class_name: str, include_inherited_descendants: bool = False) -> List[StaticCall]:
        infos: List[ProgramClassInfo]
        if include_inherited_descendants:
            infos = self.descendants(class_name, include_base=True)
        else:
            infos = self.by_name.get(class_name, [])
        calls: List[StaticCall] = []
        for c in infos:
            calls.extend(c.calls)
        return sorted(calls, key=lambda x: (x.filename, x.lineno, x.class_name, x.method_name))

    def print_tree(self, base_name: str, max_depth: int = 20, file: Any = None) -> None:
        file = file or sys.stdout
        def rec(name: str, depth: int) -> None:
            if depth > max_depth:
                print("  " * depth + "...", file=file)
                return
            for child in self.direct_children(name):
                print("  " * depth + f"{child.name}  ({child.filename}:{child.lineno})", file=file)
                rec(child.name, depth + 1)
        print(f"{_basename(base_name)}", file=file)
        rec(_basename(base_name), 1)

    def to_rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "class": c.name,
                "bases": ", ".join(c.bases),
                "file": c.filename,
                "line": c.lineno,
                "methods": ", ".join(c.methods),
                "n_pulse_calls": len(c.calls),
            }
            for c in self.classes
        ]

    def write_csv(self, path: Union[str, Path]) -> None:
        rows = self.to_rows()
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["class"])
            writer.writeheader()
            writer.writerows(rows)


def discover_program_classes(
    roots: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    base_hints: Optional[Iterable[str]] = None,
    pulse_calls: Optional[Iterable[str]] = None,
    include_tests: bool = False,
    include_all_program_suffix: bool = True,
) -> ProgramIndex:
    """
    Statically scan Python files and return Program-like classes.

    Parameters
    ----------
    roots:
        A repo root, experiments directory, single .py file, or list of those.
    base_hints:
        Seed class base names that should count as Program bases. Classes that
        transitively inherit these bases are inferred automatically.
    pulse_calls:
        Seed low-level method names to summarize as pulse/timing/measurement
        calls. Wrapper methods are inferred automatically if they eventually
        call one of these seed methods through self.* calls.
    include_tests:
        Include files under test/tests directories.
    include_all_program_suffix:
        Include classes whose name ends in 'Program' even if the base is unknown.
    """
    if isinstance(roots, (str, Path)):
        roots = [roots]
    root_paths = [Path(r).expanduser().resolve() for r in roots]
    hints = set(base_hints or PROGRAM_BASE_HINTS)
    seed_calls = set(pulse_calls or DEFAULT_PULSE_CALLS)

    files: List[Path] = []
    for root in root_paths:
        if root.is_file() and root.suffix == ".py":
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.rglob("*.py")))

    raw_classes: List[_RawClassInfo] = []
    for p in files:
        rel = None
        for root in root_paths:
            try:
                rel = str(p.relative_to(root))
                break
            except Exception:
                pass
        filename = rel or str(p)
        parts = set(p.parts)
        if not include_tests and ("tests" in parts or "test" in parts):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(text, filename=str(p))
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = []
            for b in node.bases:
                bases.append(_safe_unparse(b))
            methods: List[str] = []
            method_nodes: Dict[str, ast.FunctionDef] = {}
            method_self_calls: Dict[str, Tuple[str, ...]] = {}
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods.append(item.name)
                    method_nodes[item.name] = item
                    method_self_calls[item.name] = _self_calls_in_method(item)
            raw_classes.append(
                _RawClassInfo(
                    name=node.name,
                    bases=tuple(bases),
                    filename=filename,
                    lineno=node.lineno,
                    methods=tuple(methods),
                    method_nodes=method_nodes,
                    method_self_calls=method_self_calls,
                )
            )

    program_names = _infer_program_class_names(
        raw_classes,
        hints,
        include_all_program_suffix=include_all_program_suffix,
    )
    target_calls = _infer_pulse_call_names(raw_classes, seed_calls)

    out: List[ProgramClassInfo] = []
    for raw in raw_classes:
        if raw.name not in program_names:
            continue
        calls: List[StaticCall] = []
        for method_name in raw.methods:
            node = raw.method_nodes[method_name]
            visitor = _PulseCallVisitor(raw.name, method_name, raw.filename, target_calls)
            visitor.visit(node)
            calls.extend(visitor.calls)
        out.append(
            ProgramClassInfo(
                name=raw.name,
                bases=raw.bases,
                filename=raw.filename,
                lineno=raw.lineno,
                methods=raw.methods,
                calls=tuple(calls),
            )
        )
    return ProgramIndex(
        sorted(out, key=lambda c: (c.filename, c.lineno, c.name)),
        pulse_call_names=target_calls,
        program_class_names=program_names,
    )


# -----------------------------------------------------------------------------
# Runtime tracer
# -----------------------------------------------------------------------------

@dataclass
class PulseEvent:
    t0: float
    duration: float
    ch: str
    kind: str
    label: str
    params: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    note: str = ""

    @property
    def t1(self) -> float:
        return self.t0 + self.duration


@dataclass
class TraceMarker:
    t: float
    kind: str
    label: str
    params: Dict[str, Any] = field(default_factory=dict)
    source: str = ""


@dataclass
class CircuitBlock:
    ch: str
    kind: str
    label: str
    params_text: str
    events: List[PulseEvent] = field(default_factory=list)
    count: int = 1
    step: int = 0
    sweep_text: str = ""
    pattern_count: int = 1
    pattern_size: int = 1
    pattern_text: str = ""
    pattern_notes: Tuple[str, ...] = ()

    @property
    def t0(self) -> float:
        return self.events[0].t0 if self.events else 0.0

    @property
    def t1(self) -> float:
        return self.events[-1].t1 if self.events else self.t0


class PulseTrace:
    """Container and plotter for traced pulse events."""

    def __init__(self, unit: str = "us", *, show_zero_gain: bool = False):
        self.unit = unit
        self.show_zero_gain = show_zero_gain
        self.events: List[PulseEvent] = []
        self.markers: List[TraceMarker] = []
        self.metadata: Dict[str, Any] = {}
        self.channel_next: Dict[str, float] = defaultdict(float)
        self.t_ref: float = 0.0
        self.current_section: str = ""

    def now(self) -> float:
        if self.channel_next:
            return max([self.t_ref] + list(self.channel_next.values()))
        return self.t_ref

    def add_event(self, *, ch: Any, kind: str, label: str, duration: Any = 0.0, t0: Optional[Any] = None,
                  params: Optional[Dict[str, Any]] = None, source: str = "", note: str = "") -> PulseEvent:
        chs = _as_list(ch)
        if not chs:
            chs = ["?"]
        last_event = None
        dur = max(_try_float(duration, 0.0) or 0.0, 0.0)
        for c in chs:
            cstr = str(c)
            if t0 is None or t0 == "auto":
                start = self.channel_next.get(cstr, self.t_ref)
            else:
                start = self.t_ref + (_try_float(t0, 0.0) or 0.0)
            if (params or {}).get("gain", None) == 0 and not self.show_zero_gain and kind == "pulse":
                self.channel_next[cstr] = max(self.channel_next.get(cstr, self.t_ref), start + dur)
                continue
            ev = PulseEvent(
                t0=start,
                duration=dur,
                ch=cstr,
                kind=kind,
                label=label,
                params=dict(params or {}),
                source=source or self.current_section,
                note=note,
            )
            self.events.append(ev)
            self.channel_next[cstr] = max(self.channel_next.get(cstr, self.t_ref), ev.t1)
            last_event = ev
        return last_event  # type: ignore[return-value]

    def add_marker(self, kind: str, label: str, *, t: Optional[Any] = None,
                   params: Optional[Dict[str, Any]] = None, source: str = "") -> TraceMarker:
        tt = self.now() if t is None else self.t_ref + (_try_float(t, 0.0) or 0.0)
        m = TraceMarker(t=tt, kind=kind, label=label, params=dict(params or {}), source=source or self.current_section)
        self.markers.append(m)
        return m

    def sync_all(self, delay: Any = 0.0, *, label: str = "sync_all") -> None:
        delay_f = max(_try_float(delay, 0.0) or 0.0, 0.0)
        max_end = self.now()
        if delay_f > 0:
            self.events.append(
                PulseEvent(
                    t0=max_end,
                    duration=delay_f,
                    ch="SYNC",
                    kind="sync",
                    label=label,
                    params={"delay": delay_f},
                    source=self.current_section,
                    note="barrier delay after all active channels",
                )
            )
        new_ref = max_end + delay_f
        self.t_ref = new_ref
        # After QICK sync_all the per-channel timestamps are reset relative to a common reference.
        for ch in list(self.channel_next.keys()):
            self.channel_next[ch] = new_ref

    def wait_all(self, delay: Any = 0.0, *, label: str = "wait_all") -> None:
        delay_f = max(_try_float(delay, 0.0) or 0.0, 0.0)
        max_end = self.now()
        self.events.append(
            PulseEvent(
                t0=max_end,
                duration=delay_f,
                ch="WAIT",
                kind="wait",
                label=label,
                params={"delay": delay_f},
                source=self.current_section,
                note="tProc wait; shown as a marker/delay for readability",
            )
        )
        self.t_ref = max_end + delay_f

    def channels(self, show_sync: bool = True) -> List[str]:
        chans = sorted({e.ch for e in self.events})
        if not show_sync:
            chans = [c for c in chans if c not in {"SYNC", "WAIT"} and not c.startswith("ADC")]
        return chans

    def to_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for i, e in enumerate(sorted(self.events, key=lambda x: (x.t0, x.ch, x.kind))):
            row = {
                "i": i,
                "t0": e.t0,
                "t1": e.t1,
                "duration": e.duration,
                "channel": e.ch,
                "kind": e.kind,
                "label": e.label,
                "source": e.source,
                "note": e.note,
                "params_json": json.dumps(e.params, default=str, ensure_ascii=False),
            }
            rows.append(row)
        return rows

    def to_csv(self, path: Union[str, Path]) -> None:
        rows = self.to_rows()
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows[0].keys()) if rows else ["i", "t0", "t1", "duration", "channel", "kind", "label"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def to_json(self, path: Union[str, Path]) -> None:
        payload = {
            "unit": self.unit,
            "events": [asdict(e) for e in self.events],
            "markers": [asdict(m) for m in self.markers],
        }
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def _format_circuit_value(self, value: Any) -> str:
        val = _try_float(value, None)
        if val is not None and stdlib_math.isfinite(val):
            if abs(val) >= 100 or (abs(val) > 0 and abs(val) < 0.001):
                return f"{val:.4g}"
            return f"{val:.5g}"
        return _short_value(value, maxlen=24)

    def _normalize_circuit_label(self, label: Any) -> str:
        raw = str(label or "").strip()
        if not raw:
            return "pulse"

        lower = raw.lower()
        qubit_match = re.match(r"^(pi|hpi|pi2|hpi2)_qubit_(ge|ef)(?:_.*)?$", lower)
        if qubit_match:
            pulse, transition = qubit_match.groups()
            return f"{pulse}_{transition}"

        storage_match = re.match(r"^(pi|hpi|pi2|hpi2)_m1s(\d+)(?:_arb)?$", lower)
        if storage_match:
            pulse, storage = storage_match.groups()
            return f"{pulse} M1-S{storage}"

        if lower in {"pi_f0g1", "pi_f0g1_arb"}:
            return "pi f0g1"
        if lower in {"pi_m1si_low", "pi_m1si_high"}:
            return "pi M1-Si"

        if raw.startswith("temp_gaussian"):
            return "pulse"
        return raw

    def _format_phase_value(self, value: Any, *, signed: bool = False) -> str:
        val = _try_float(value, None)
        if val is None or not stdlib_math.isfinite(val):
            return _short_value(value, maxlen=16)
        if abs(val) < 1e-9:
            val = 0.0
        text = f"{val:.1f}" if abs(val - round(val)) > 1e-6 else str(int(round(val)))
        if signed and val > 0:
            text = "+" + text
        return text

    def _relative_phase(self, phase: float, previous: float) -> float:
        delta = phase - previous
        wrapped = ((delta + 180.0) % 360.0) - 180.0
        if abs(wrapped + 180.0) < 1e-9 and delta > 0:
            return 180.0
        return wrapped

    def _block_phase_values(self, block: CircuitBlock) -> List[float]:
        values = []
        for event in block.events:
            phase = _try_float((event.params or {}).get("phase"), None)
            if phase is not None and stdlib_math.isfinite(phase):
                values.append(float(phase))
        return values

    def _block_phase_text(
        self,
        block: CircuitBlock,
        previous_phase_by_channel: MutableMapping[str, float],
        *,
        phase_mode: str = "relative",
        show_zero_phase: bool = False,
    ) -> str:
        if block.kind != "pulse":
            return ""
        phases = self._block_phase_values(block)
        if not phases:
            return ""

        rounded = {round(p, 6) for p in phases}
        if len(rounded) > 1:
            previous_phase_by_channel[block.ch] = phases[-1]
            return f"ph {self._format_phase_value(phases[0])}->{self._format_phase_value(phases[-1])}"

        phase = phases[0]
        previous = previous_phase_by_channel.get(block.ch)
        previous_phase_by_channel[block.ch] = phase

        pieces = []
        if show_zero_phase or abs(phase) > 1e-9:
            pieces.append(f"ph={self._format_phase_value(phase)}")
        if phase_mode in {"relative", "both"} and previous is not None:
            delta = self._relative_phase(phase, previous)
            if abs(delta) > 1e-9:
                pieces.append(f"dph={self._format_phase_value(delta, signed=True)}")
        return " ".join(pieces)

    def _block_detail_text(
        self,
        block: CircuitBlock,
        phase_text: str,
        *,
        detail: str = "phase",
    ) -> str:
        if block.kind in {"init", "loop", "update"}:
            return block.params_text
        mode = str(detail or "phase").lower()
        variation_text = block.params_text if str(block.params_text).startswith("varies:") else ""
        if mode in {"none", "off", "false"}:
            return ""
        if mode in {"full", "all", "params"}:
            return block.params_text
        if mode in {"phase", "ph"}:
            return variation_text or phase_text
        if mode in {"compact", "essential"}:
            params = block.events[0].params if block.events else {}
            pieces = []
            if variation_text:
                pieces.append(variation_text)
            elif phase_text:
                pieces.append(phase_text)
            if params.get("waveform") == "pulse_to_test" and not variation_text:
                sweep = self._block_sweep_text(block)
                if sweep:
                    pieces.append(sweep)
            return " ".join(pieces)
        return phase_text

    def _block_param_variation_text(self, block: CircuitBlock) -> str:
        if block.kind != "pulse" or len(block.events) <= 1:
            return ""
        pieces = []
        for key, short in (("freq", "f"), ("gain", "g"), ("length", "len"), ("phase", "ph")):
            values = [
                (event.params or {}).get(key)
                for event in block.events
                if key in (event.params or {}) and (event.params or {}).get(key) is not None
            ]
            if len(values) <= 1:
                continue
            normalized = set()
            for value in values:
                val = _try_float(value, None)
                if val is not None and stdlib_math.isfinite(val):
                    normalized.add(round(float(val), 9))
                else:
                    normalized.add(_short_value(value, 24))
            if len(normalized) > 1:
                pieces.append(self._sequence_variation_text(short, values, index_var="i"))
        if not pieces:
            return ""
        return "varies: " + ", ".join(pieces[:3])

    def _sequence_variation_text(self, label: str, values: Sequence[Any], *, index_var: str = "i") -> str:
        vals = [_try_float(v, None) for v in values]
        if all(v is not None and stdlib_math.isfinite(v) for v in vals):
            nums = [float(v) for v in vals if v is not None]
            if len(nums) >= 4:
                rounded = [round(v, 9) for v in nums]
                if len(set(rounded[0::2])) == 1 and len(set(rounded[1::2])) == 1 and rounded[0] != rounded[1]:
                    return f"{label} alternates {self._format_circuit_value(nums[0])}/{self._format_circuit_value(nums[1])}"
                deltas = [round(nums[i + 1] - nums[i], 9) for i in range(len(nums) - 1)]
                if deltas and len(set(deltas)) == 1 and abs(deltas[0]) > 1e-12:
                    return f"{label} {self._format_circuit_value(nums[0])}+{index_var}*{self._format_circuit_value(deltas[0])}"
            return f"{label} {self._format_circuit_value(nums[0])}->{self._format_circuit_value(nums[-1])}"
        return f"{label} {_short_value(values[0], 18)}->{_short_value(values[-1], 18)}"

    def _sequence_summary_text(self, name: str, values: Any, *, limit_to: Optional[int] = None) -> str:
        seq = self._iter_channel_values(values)
        if limit_to is not None and len(seq) > limit_to:
            seq = seq[:limit_to]
        if not seq:
            return f"{name}: <empty>"
        vals = [_try_float(v, None) for v in seq]
        if all(v is not None and stdlib_math.isfinite(v) for v in vals):
            nums = [float(v) for v in vals if v is not None]
            if len(nums) == 1:
                return f"{name}: {self._format_circuit_value(nums[0])}"
            deltas = [round(nums[i + 1] - nums[i], 9) for i in range(len(nums) - 1)]
            if deltas and len(set(deltas)) == 1:
                step = self._format_circuit_value(deltas[0])
                return (
                    f"{name}[i]: {self._format_circuit_value(nums[0])}"
                    f"+i*{step}, i=0..{len(nums) - 1}"
                )
            return f"{name}[i]: {self._format_circuit_value(nums[0])}->{self._format_circuit_value(nums[-1])}, n={len(nums)}"
        if len(seq) == 1:
            return f"{name}: {_short_value(seq[0], 32)}"
        return f"{name}[i]: {_short_value(seq[0], 24)} -> {_short_value(seq[-1], 24)}, n={len(seq)}"

    def _compact_sequence_text(self, values: Any, *, max_items: int = 8) -> str:
        seq = self._iter_channel_values(values)
        if not seq:
            return "[]"
        pieces = [self._format_circuit_value(value) for value in seq[:max_items]]
        if len(seq) > max_items:
            pieces.append("...")
            pieces.append(self._format_circuit_value(seq[-1]))
        return "[" + ", ".join(pieces) + "]"

    def _cfg_sequence(self, name: str) -> List[Any]:
        value = self._cfg_expt_value(name, None)
        if value is None:
            return []
        return self._iter_channel_values(value)

    def _cfg_expt_value(self, name: str, default: Any = None) -> Any:
        return _get_attr_or_key(self._cfg_expt(), name, default)

    def _pulse_pi_fraction(self) -> Optional[int]:
        pulse_type = self._cfg_expt_value("pulse_type", None)
        if not isinstance(pulse_type, (list, tuple)) or len(pulse_type) < 3:
            return None
        pulse_name = str(pulse_type[2])
        if pulse_name == "pi":
            return 1
        if pulse_name == "hpi":
            return 2
        match = re.match(r"^pi/(\d+)$", pulse_name)
        if match:
            return int(match.group(1))
        return None

    def _block_repeat_text(self, block: CircuitBlock) -> str:
        if block.count <= 1:
            return ""
        n_pulses = _try_float(self._cfg_expt_value("n_pulses", None), None)
        pi_frac = self._pulse_pi_fraction()
        if n_pulses is not None and pi_frac is not None and n_pulses > 0:
            per_body = int(round(2 * n_pulses * pi_frac))
            if per_body == block.count:
                n_text = self._format_circuit_value(n_pulses)
                sweep_param = self._cfg_expt_value("parameter_to_test", None)
                step = self._cfg_expt_value("step", None)
                expts = _try_float(self._cfg_expt_value("expts", None), None)
                suffix = ""
                if sweep_param is not None and step is not None:
                    if expts is not None and expts > 1:
                        suffix = f"; {sweep_param} += {self._format_circuit_value(step)} x {int(expts)} updates"
                    else:
                        suffix = f"; {sweep_param} += {self._format_circuit_value(step)}/update"
                return f"2 x {n_text} x {pi_frac} = {block.count}/body{suffix}"
        expts = _try_float(self._cfg_expt_value("expts", None), None)
        if expts is not None and expts > 1 and block.count % int(expts) == 0:
            per_update = block.count // int(expts)
            return f"{per_update}/body x {int(expts)} updates = {block.count}"
        return f"x{block.count}"

    def _fit_text_lines(self, text: str, *, width: int, max_lines: int) -> str:
        if not text:
            return ""
        parts = []
        for chunk in str(text).split("; "):
            if len(chunk) > width * 1.2 and ", " in chunk:
                parts.extend(chunk.split(", "))
            else:
                parts.append(chunk)
        lines: List[str] = []
        for part in parts:
            wrapped = textwrap.wrap(part, width=width, break_long_words=False, break_on_hyphens=False) or [part]
            lines.extend(wrapped)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            if len(lines[-1]) > max(3, width - 3):
                lines[-1] = lines[-1][: width - 3]
            lines[-1] = lines[-1].rstrip(". ") + "..."
        return "\n".join(lines)

    def _circuit_label_lines(self, label: str, max_label_chars: int) -> List[str]:
        label = self._normalize_circuit_label(label)
        if " " in label:
            parts = label.split()
            if len(parts) == 2 and len(parts[0]) <= 8 and len(parts[1]) <= 12:
                return parts
        if len(label) <= max_label_chars:
            return [label]
        return textwrap.wrap(label, width=max(8, max_label_chars // 2)) or [label]

    def _iter_channel_values(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            try:
                value = value.tolist()
            except Exception:
                pass
        if isinstance(value, (str, bytes)):
            return [value]
        if isinstance(value, (list, tuple, set)):
            out: List[Any] = []
            for item in value:
                out.extend(self._iter_channel_values(item))
            return out
        return [value]

    def _channel_key(self, value: Any) -> str:
        val = _try_float(value, None)
        if val is not None and stdlib_math.isfinite(val) and abs(val - round(val)) < 1e-9:
            return str(int(round(val)))
        return str(value)

    def _pretty_hardware_name(self, name: str) -> str:
        names = {
            "readout": "readout",
            "qubit": "qubit",
            "manipulate_in": "manipulate",
            "flux_low": "flux low",
            "flux_high": "flux high",
            "sideband": "f0g1",
            "storage_in": "storage",
        }
        return names.get(str(name), str(name).replace("_", " "))

    def _add_channel_alias(self, aliases: MutableMapping[str, List[str]], key: Any, label: str) -> None:
        if key is None:
            return
        key_str = self._channel_key(key)
        bucket = aliases.setdefault(key_str, [])
        if label not in bucket:
            bucket.append(label)

    def _hardware_channel_aliases(self) -> Dict[str, List[str]]:
        cfg = self.metadata.get("cfg")
        aliases: Dict[str, List[str]] = {}

        dacs = _nested_get(cfg, "hw.soc.dacs", None)
        if isinstance(dacs, Mapping):
            dac_items = list(dacs.items())
        else:
            dac_items = [
                (name, _get_attr_or_key(dacs, name, None))
                for name in ("readout", "qubit", "manipulate_in", "flux_low", "flux_high", "sideband", "storage_in")
            ]
        for name, dac_cfg in dac_items:
            ch = _get_attr_or_key(dac_cfg, "ch", None)
            label = self._pretty_hardware_name(str(name))
            for channel in self._iter_channel_values(ch):
                self._add_channel_alias(aliases, channel, label)

        adcs = _nested_get(cfg, "hw.soc.adcs", None)
        readout_adc = _get_attr_or_key(adcs, "readout", None)
        for channel in self._iter_channel_values(_get_attr_or_key(readout_adc, "ch", None)):
            self._add_channel_alias(aliases, f"ADC{self._channel_key(channel)}", "readout ADC")
            self._add_channel_alias(aliases, f"RO{self._channel_key(channel)}", "readout ADC")

        return aliases

    def _channel_label_parts(self, ch: str, aliases: Mapping[str, List[str]]) -> Tuple[str, str]:
        if str(ch) == "LOOP":
            return "PROGRAM", "control flow"
        labels = list(aliases.get(str(ch), []))
        if not labels and str(ch).startswith("ADC"):
            labels = ["readout ADC"]
        if not labels and str(ch).startswith("RO"):
            labels = ["readout ADC"]
        sub = ", ".join(labels[:3])
        return str(ch), sub

    def _circuit_label(self, event: PulseEvent) -> Tuple[str, str]:
        params = event.params or {}
        if event.kind in {"measure", "trigger"} or event.ch.startswith("ADC"):
            return "MEAS", event.ch
        if event.kind in {"sync", "wait"}:
            delay = params.get("delay", event.duration)
            return event.label, f"{self._format_circuit_value(delay)} {self.unit}"
        if event.kind in {"init", "loop", "update"}:
            return event.label, str(params.get("summary", ""))

        style = str(params.get("style", event.kind))
        waveform = params.get("waveform")
        display_label = params.get("display_label") or params.get("logical_label")
        if display_label:
            label = str(display_label)
        elif waveform == "pulse_to_test":
            label = self._pulse_to_test_label()
        elif waveform is not None:
            label = str(waveform)
        else:
            label = style
        ctx = params.get("_loop_context", {})
        if isinstance(ctx, Mapping) and ctx:
            label = self._label_with_context(str(label), ctx)
        config_label = self._label_from_config_match(params, str(label))
        if config_label:
            label = config_label
        label = self._normalize_circuit_label(label)

        bits = []
        for key, short in (("freq", "f"), ("gain", "g"), ("length", "len"), ("phase", "ph")):
            if key in params and params[key] is not None:
                bits.append(f"{short}={self._format_circuit_value(params[key])}")
        if style and style not in label:
            bits.insert(0, style)
        return label, " ".join(bits)

    def _circuit_signature(
        self,
        event: PulseEvent,
        *,
        group_sweeps: bool = True,
        group_param_changes: bool = True,
    ) -> Tuple[Any, ...]:
        params = event.params or {}
        waveform = params.get("waveform")
        style = params.get("style")
        display_label = params.get("display_label") or params.get("logical_label")
        if group_sweeps and waveform == "pulse_to_test":
            return (event.ch, event.kind, style, waveform)
        if event.kind in {"measure", "trigger"} or event.ch.startswith("ADC"):
            return (event.ch, "measure")
        if event.kind in {"sync", "wait"}:
            return (event.ch, event.kind, event.label)
        if event.kind in {"init", "loop", "update"}:
            return (event.ch, event.kind, event.label)
        variable_params = () if group_param_changes else (
            params.get("freq"),
            params.get("gain"),
            params.get("length"),
            params.get("phase"),
        )
        return (
            event.ch,
            event.kind,
            display_label,
            style,
            waveform,
            *variable_params,
        )

    def _cfg_expt(self) -> Any:
        cfg = self.metadata.get("cfg")
        return _get_attr_or_key(cfg, "expt", None)

    def _pulse_to_test_label(self) -> str:
        expt = self._cfg_expt()
        pulse_type = _get_attr_or_key(expt, "pulse_type", None)
        if isinstance(pulse_type, (list, tuple)) and len(pulse_type) >= 3:
            channel = pulse_type[0]
            target = pulse_type[1]
            pulse_name = pulse_type[2]
            if channel == "qubit":
                return f"{pulse_name}_{target}"
            return f"{pulse_name} {target}"
        return "pulse_to_test"

    def _global_sweep_text(self) -> str:
        expt = self._cfg_expt()
        param = _get_attr_or_key(expt, "parameter_to_test", None)
        if not param:
            return ""
        start = _get_attr_or_key(expt, "start", None)
        step = _get_attr_or_key(expt, "step", None)
        expts = _get_attr_or_key(expt, "expts", None)
        if start is None or step is None:
            return f"sweep: {param}"
        start_s = self._format_circuit_value(start)
        step_s = self._format_circuit_value(step)
        if expts is not None:
            return f"sweep: {param} = {start_s} + k*{step_s}, k=0..{int(expts) - 1}"
        return f"sweep: {param} = {start_s} + k*{step_s}"

    def _block_sweep_text(self, block: CircuitBlock) -> str:
        if not block.events:
            return ""
        params = block.events[0].params or {}
        if params.get("waveform") != "pulse_to_test":
            return ""
        return self._global_sweep_text().replace("sweep: ", "", 1)

    def _block_pattern_signature(self, block: CircuitBlock) -> Optional[Tuple[Any, ...]]:
        if block.kind in {"init", "loop", "update", "sync", "wait"}:
            return None
        if not block.events:
            return None
        params = block.events[0].params or {}
        display_label = params.get("display_label") or params.get("logical_label")
        waveform = params.get("waveform")
        style = params.get("style", block.kind)
        label = self._normalize_circuit_label(display_label or block.label)
        if block.kind in {"measure", "trigger"} or str(block.ch).startswith("ADC"):
            label = "MEAS"
            style = "measure"
            waveform = ""
        return (block.ch, block.kind, label, str(style), str(waveform))

    def _assign_circuit_steps(self, blocks: Sequence[CircuitBlock], *, max_group_gap: float = 1e-9) -> None:
        last_t: Optional[float] = None
        step = -1
        used_at_step: Dict[int, set] = defaultdict(set)
        for block in blocks:
            if last_t is None or abs(block.t0 - last_t) > max_group_gap:
                step += 1
                last_t = block.t0
            if block.ch in used_at_step[step]:
                step += 1
                last_t = block.t0
            block.step = step
            used_at_step[step].add(block.ch)

    def _block_position_variation_text(self, repeated_blocks: Sequence[CircuitBlock]) -> str:
        if len(repeated_blocks) <= 1:
            return ""
        pieces = []
        for key, short in (("freq", "f"), ("gain", "g"), ("length", "len"), ("phase", "ph")):
            values = []
            for block in repeated_blocks:
                if not block.events:
                    continue
                params = block.events[0].params or {}
                if key in params and params[key] is not None:
                    values.append(params[key])
            if len(values) <= 1:
                continue
            normalized = set()
            for value in values:
                val = _try_float(value, None)
                if val is not None and stdlib_math.isfinite(val):
                    normalized.add(round(float(val), 9))
                else:
                    normalized.add(_short_value(value, 24))
            if len(normalized) > 1:
                pieces.append(self._sequence_variation_text(short, values, index_var="i"))
        if not pieces:
            return ""
        return "varies: " + ", ".join(pieces[:2])

    def _loop_context(self, block: CircuitBlock) -> Dict[str, Any]:
        if not block.events:
            return {}
        ctx = (block.events[0].params or {}).get("_loop_context", {})
        return dict(ctx) if isinstance(ctx, Mapping) else {}

    def _event_loop_context_key(self, event: PulseEvent) -> Any:
        ctx = (event.params or {}).get("_loop_context", {})
        return self._context_value_key(ctx) if isinstance(ctx, Mapping) and ctx else None

    def _block_loop_context_key(self, block: CircuitBlock) -> Any:
        if not block.events:
            return None
        return self._event_loop_context_key(block.events[-1])

    def _context_value_key(self, value: Any) -> Any:
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        if hasattr(value, "tolist"):
            try:
                value = value.tolist()
            except Exception:
                pass
        if isinstance(value, Mapping):
            return tuple(sorted((str(k), self._context_value_key(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(self._context_value_key(v) for v in value)
        if isinstance(value, set):
            return tuple(sorted(self._context_value_key(v) for v in value))
        val = _try_float(value, None)
        if val is not None and stdlib_math.isfinite(val):
            return round(float(val), 9)
        return str(value)

    def _context_value_text(self, value: Any) -> str:
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, (list, tuple, set)) or hasattr(value, "tolist"):
            return self._compact_sequence_text(value, max_items=10)
        return self._format_circuit_value(value)

    def _context_scalar_value(self, value: Any) -> Optional[Any]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (str, int, float)):
            return value
        if hasattr(value, "item"):
            try:
                item = value.item()
                if isinstance(item, (str, int, float, bool)):
                    return item
            except Exception:
                pass
        return None

    def _label_context_value(self, ctx: Mapping[str, Any]) -> Optional[Any]:
        keys = list(ctx)
        exact_priority = {"stor", "storage", "mode", "target", "state"}
        for key in keys:
            low = str(key).lower()
            if low in exact_priority:
                value = self._context_scalar_value(ctx[key])
                if value is not None:
                    return value
        for key in keys:
            low = str(key).lower()
            if low.endswith(("_stor", "_storage", "_mode", "_target", "_state")) and not low.startswith(("i_", "j_", "k_")):
                value = self._context_scalar_value(ctx[key])
                if value is not None:
                    return value
        indexed = self._indexed_context_value(ctx)
        if indexed is not None:
            return indexed
        for key in sorted(keys, key=self._context_key_order):
            low = str(key).lower()
            if low in {"i", "j", "k", "kk"} or low.startswith(("i_", "j_", "k_")):
                continue
            value = self._context_scalar_value(ctx[key])
            if value is not None:
                return value
        return None

    def _label_with_context(self, label: str, ctx: Mapping[str, Any]) -> str:
        out = str(label)
        for key, raw_value in sorted(ctx.items(), key=lambda item: self._context_key_order(str(item[0]))):
            value = self._context_scalar_value(raw_value)
            if value is None:
                continue
            value_text = self._context_value_text(value)
            out = out.replace("{" + str(key) + "}", value_text)
            out = re.sub(rf"\${re.escape(str(key))}\b", value_text, out)

        value = self._label_context_value(ctx)
        if value is not None:
            value_text = self._context_value_text(value)
            # Covers compact symbolic labels such as M1-Si without touching "pi".
            out = re.sub(r"\b([A-Z][A-Za-z0-9]*-)S[i]\b", rf"\1S{value_text}", out)
            out = re.sub(r"\bS[i]\b", f"S{value_text}", out)
            compact = out.lower()
            if re.search(r"(^|[_\s-])m1si($|[_\s-])", compact):
                pulse = ""
                match = re.match(r"^(pi|hpi|pi2|hpi2)", compact)
                if match:
                    pulse = match.group(1) + " "
                return f"{pulse}M1-S{value_text}".strip()
        return out

    def _indexed_context_value(self, ctx: Mapping[str, Any]) -> Optional[Any]:
        for key, raw_idx in sorted(ctx.items(), key=lambda item: self._context_key_order(str(item[0]))):
            low = str(key).lower()
            if not low.startswith(("i_", "j_", "k_")):
                continue
            idx_f = _try_float(raw_idx, None)
            if idx_f is None or not stdlib_math.isfinite(idx_f):
                continue
            idx = int(round(idx_f))
            base = low.split("_", 1)[1]
            if not base:
                continue
            for seq_key, seq in ctx.items():
                seq_low = str(seq_key).lower()
                if seq_key == key or base not in seq_low:
                    continue
                if not (seq_low.endswith("s") or seq_low.endswith("_list") or "list" in seq_low):
                    continue
                values = self._context_sequence_values(seq)
                if values and 0 <= idx < len(values):
                    value = self._context_scalar_value(values[idx])
                    if value is not None:
                        return value
        return None

    def _context_sequence_values(self, value: Any) -> List[Any]:
        try:
            if hasattr(value, "tolist"):
                value = value.tolist()
        except Exception:
            pass
        if isinstance(value, (list, tuple)):
            return list(value)
        return []

    def _pulse_prefix_for_label(self, label: str) -> str:
        match = re.match(r"^(pi|hpi|pi2|hpi2)(?:\b|_)", str(label).strip().lower())
        return (match.group(1) + " ") if match else ""

    def _label_needs_config_match(self, label: str) -> bool:
        low = str(label or "").strip().lower()
        if not low:
            return True
        if low in {"pulse", "arb", "const", "flat", "flat_top", "flattop", "gaussian"}:
            return True
        return bool(re.search(r"m1[-_ ]?si|\bsi\b|s\{?i\}?", low))

    def _label_from_config_match(self, params: Mapping[str, Any], label: str) -> str:
        if not self._label_needs_config_match(label):
            return ""
        matched = self._label_from_dataset_match(params) or self._label_from_nested_config_match(params)
        if not matched:
            return ""
        prefix = self._pulse_prefix_for_label(label)
        if prefix and not matched.lower().startswith(prefix.strip().lower()):
            return (prefix + matched).strip()
        return matched

    def _numeric_values(self, value: Any, *, max_items: int = 16) -> List[float]:
        try:
            if hasattr(value, "tolist"):
                value = value.tolist()
        except Exception:
            pass
        if isinstance(value, (list, tuple)):
            out: List[float] = []
            for item in list(value)[:max_items]:
                out.extend(self._numeric_values(item, max_items=max_items))
            return out[:max_items]
        val = _try_float(value, None)
        if val is not None and stdlib_math.isfinite(val):
            return [float(val)]
        return []

    def _numbers_match(self, observed: Any, candidate: Any, *, kind: str = "") -> bool:
        obs = self._numeric_values(observed, max_items=1)
        cands = self._numeric_values(candidate)
        if not obs or not cands:
            return False
        target = obs[0]
        for cand in cands:
            if kind == "gain":
                if abs(target - cand) <= 0.51:
                    return True
            elif abs(target - cand) <= max(1e-3, 2e-6 * max(abs(target), abs(cand), 1.0)):
                return True
        return False

    def _record_match_score(self, params: Mapping[str, Any], record: Mapping[str, Any]) -> int:
        score = 0
        for key, value in record.items():
            low = str(key).lower()
            if "freq" in low and "freq" in params and self._numbers_match(params.get("freq"), value, kind="freq"):
                score += 3
            elif "gain" in low and "gain" in params and self._numbers_match(params.get("gain"), value, kind="gain"):
                score += 1
            elif any(token in low for token in ("length", "len", "pi (", "h_pi")) and "length" in params:
                if self._numbers_match(params.get("length"), value, kind="length"):
                    score += 1
        return score

    def _label_from_dataset_match(self, params: Mapping[str, Any]) -> str:
        cfg = self.metadata.get("cfg")
        datasets = [
            _nested_get(cfg, "device.storage._ds_storage", None),
            _nested_get(cfg, "device.storage._ds_floquet", None),
        ]
        best: Tuple[int, str] = (0, "")
        for ds in datasets:
            df = getattr(ds, "df", None)
            if df is None or not hasattr(df, "iterrows"):
                continue
            try:
                rows = df.iterrows()
            except Exception:
                continue
            for idx, row in rows:
                try:
                    record = {str(col): row[col] for col in row.index}
                except Exception:
                    continue
                score = self._record_match_score(params, record)
                if score < 3:
                    continue
                label = str(record.get("stor_name") or idx)
                if score > best[0]:
                    best = (score, label)
        return best[1]

    def _label_from_nested_config_match(self, params: Mapping[str, Any]) -> str:
        cfg = self.metadata.get("cfg")
        roots = [
            ("device.multiphoton", _nested_get(cfg, "device.multiphoton", None)),
            ("device.qubit", _nested_get(cfg, "device.qubit", None)),
            ("device.storage", _nested_get(cfg, "device.storage", None)),
        ]
        best: Tuple[int, str] = (0, "")
        for root_name, root in roots:
            for path, record in self._iter_config_records(root, root_name, max_depth=5):
                score = self._record_match_score(params, record)
                if score < 3 or score <= best[0]:
                    continue
                best = (score, self._pretty_config_path_label(path))
        return best[1]

    def _iter_config_records(self, obj: Any, path: str, *, max_depth: int, _depth: int = 0) -> Iterable[Tuple[str, Mapping[str, Any]]]:
        if obj is None or _depth > max_depth:
            return
        if isinstance(obj, Mapping):
            items = list(obj.items())
            if any(any(token in str(k).lower() for token in ("freq", "gain", "length", "len")) for k, _ in items):
                yield path, obj
            for key, value in items:
                if str(key).startswith("_"):
                    continue
                yield from self._iter_config_records(value, f"{path}.{key}", max_depth=max_depth, _depth=_depth + 1)
        elif isinstance(obj, (list, tuple)):
            for idx, value in enumerate(list(obj)[:16]):
                yield from self._iter_config_records(value, f"{path}[{idx}]", max_depth=max_depth, _depth=_depth + 1)
        else:
            raw = getattr(obj, "__dict__", None)
            if isinstance(raw, Mapping):
                public = {k: v for k, v in raw.items() if not str(k).startswith("_")}
                yield from self._iter_config_records(public, path, max_depth=max_depth, _depth=_depth + 1)

    def _pretty_config_path_label(self, path: str) -> str:
        parts = [part for part in re.split(r"\.", str(path)) if part and part not in {"device"}]
        if len(parts) >= 2 and parts[0] == "multiphoton":
            return " ".join(parts[1:3]).replace("_", " ")
        if parts:
            return parts[-1].replace("_", " ")
        return str(path)

    def _context_key_order(self, key: str) -> Tuple[int, str]:
        low = key.lower()
        priority_tokens = (
            "stor", "mode", "cycle", "kk", "phase", "freq", "detun",
            "gain", "length", "idx", "index", "update", "pulse",
        )
        if low in {"i", "j", "k", "kk"} or low.startswith(("i_", "j_", "k_")):
            return (0, low)
        for idx, token in enumerate(priority_tokens, start=1):
            if token in low:
                return (idx, low)
        return (99, low)

    def _context_pattern_width(
        self,
        blocks: Sequence[CircuitBlock],
        *,
        min_repeats: int,
        max_pattern_len: int,
    ) -> int:
        contexts = [self._loop_context(block) for block in blocks]
        if not any(contexts):
            return 0
        max_width = min(max_pattern_len, len(blocks) // max(min_repeats, 1))
        best_width = 0
        best_score = 0
        keys = sorted({key for ctx in contexts for key in ctx}, key=self._context_key_order)
        for width in range(2, max_width + 1):
            score = 0
            for key in keys:
                values = [ctx.get(key, None) for ctx in contexts]
                if any(value is None for value in values):
                    continue
                pattern = [self._context_value_key(value) for value in values[:width]]
                if len(set(pattern)) <= 1:
                    continue
                if all(self._context_value_key(values[idx]) == pattern[idx % width] for idx in range(len(values))):
                    score += 1
            if score > best_score:
                best_width = width
                best_score = score
        return best_width

    def _prefer_context_loop_pattern(
        self,
        best: Optional[Tuple[int, int, int]],
        blocks: Sequence[CircuitBlock],
        *,
        start: int,
        min_repeats: int,
        max_pattern_len: int,
        min_saved_blocks: int,
    ) -> Optional[Tuple[int, int, int]]:
        if best is None:
            return best
        pat_len, reps, _ = best
        if pat_len != 1:
            return best
        segment = list(blocks[start : start + reps])
        width = self._context_pattern_width(segment, min_repeats=min_repeats, max_pattern_len=max_pattern_len)
        if width <= 1 or reps % width != 0:
            return best
        context_reps = reps // width
        saved = (context_reps - 1) * width
        if context_reps >= min_repeats and saved >= min_saved_blocks:
            return (width, context_reps, saved)
        return best

    def _fold_pattern_label(self, reps: int, pattern_len: int) -> str:
        expt = self._cfg_expt()
        floquet_cycles = _get_attr_or_key(expt, "floquet_cycles", None)
        n_cycles = None
        if floquet_cycles is not None and not isinstance(floquet_cycles, (str, bytes)):
            try:
                n_cycles = len(floquet_cycles)
            except Exception:
                n_cycles = None
        if n_cycles == reps:
            return f"floquet pattern x{reps}"
        return f"pattern x{reps}"

    def _context_pattern_notes(self, segment: Sequence[CircuitBlock], pattern_len: int, reps: int) -> List[str]:
        contexts = [self._loop_context(block) for block in segment]
        if not any(contexts):
            return []
        items = [f"loop unit: {pattern_len} ops x {reps} repeats = {pattern_len * reps} pulses"]
        keys = sorted({key for ctx in contexts for key in ctx}, key=self._context_key_order)

        first = contexts[:pattern_len]
        for key in keys:
            values = [ctx.get(key) for ctx in first if key in ctx]
            if len(values) != pattern_len:
                continue
            normalized = {self._context_value_key(value) for value in values}
            if len(normalized) > 1:
                vals = ", ".join(self._context_value_text(value) for value in values[:8])
                if len(values) > 8:
                    vals += ", ..."
                items.append(f"{key}[j]: [{vals}], j=0..{pattern_len - 1}")

        for key in keys:
            values = []
            for rep in range(reps):
                idx = rep * pattern_len
                if idx >= len(contexts) or key not in contexts[idx]:
                    values = []
                    break
                values.append(contexts[idx][key])
            if len(values) <= 1:
                continue
            normalized = {self._context_value_key(value) for value in values}
            if len(normalized) > 1:
                summary = self._sequence_summary_text(key, values)
                if summary not in items:
                    items.append(summary)

        for key in keys:
            values = [ctx.get(key) for ctx in contexts if key in ctx]
            if len(values) != len(contexts):
                continue
            normalized = {self._context_value_key(value) for value in values}
            if len(normalized) == 1:
                value = values[0]
                if isinstance(value, bool) and value is False:
                    continue
                if isinstance(value, (int, float, str, bool)) or isinstance(value, (list, tuple, set)) or hasattr(value, "tolist"):
                    if isinstance(value, bool):
                        text = f"if {key}: {self._context_value_text(value)}"
                    else:
                        text = f"{key}: {self._context_value_text(value)}"
                    if text not in items:
                        items.append(text)
        return items[:12]

    def _annotate_context_pattern_blocks(self, pattern: Sequence[CircuitBlock]) -> None:
        contexts = [self._loop_context(block) for block in pattern]
        if not any(contexts):
            return
        keys = sorted({key for ctx in contexts for key in ctx}, key=self._context_key_order)
        varying_keys = []
        for key in keys:
            values = [ctx.get(key) for ctx in contexts if key in ctx]
            if len(values) != len(pattern):
                continue
            if len({self._context_value_key(value) for value in values}) > 1:
                varying_keys.append(key)
        if not varying_keys:
            return
        preferred = [key for key in varying_keys if not key.lower().startswith(("i_", "j_", "k_")) and key.lower() not in {"i", "j", "k", "kk"}]
        keys_to_show = (preferred or varying_keys)[:2]
        for block, ctx in zip(pattern, contexts):
            if ctx:
                block.label = self._label_with_context(block.label, ctx)
            bits = []
            for key in keys_to_show:
                if key in ctx:
                    bits.append(f"{key}={self._context_value_text(ctx[key])}")
            if not bits:
                continue
            detail = ", ".join(bits)
            if block.params_text:
                if detail not in block.params_text:
                    block.params_text = f"{detail}; {block.params_text}"
            else:
                block.params_text = detail

    def _pattern_context_items(self, block: CircuitBlock) -> List[str]:
        expt = self._cfg_expt()
        items = list(block.pattern_notes)
        reps = block.pattern_count
        swept_params = _get_attr_or_key(expt, "swept_params", None)
        if isinstance(swept_params, str):
            swept_params_list = [swept_params]
        elif swept_params is None:
            swept_params_list = []
        else:
            try:
                swept_params_list = [str(param) for param in swept_params]
            except Exception:
                swept_params_list = []
        if swept_params_list:
            items.append("swept params: " + ", ".join(swept_params_list[:4]))

        for name in ("floquet_cycles", "floquet_cycle"):
            value = _get_attr_or_key(expt, name, None)
            if value is not None:
                items.append(self._sequence_summary_text(name, value, limit_to=reps if name.endswith("s") else None))
                break

        for param in swept_params_list:
            plural = f"{param}s"
            values = _get_attr_or_key(expt, plural, None)
            if values is not None:
                summary = self._sequence_summary_text(plural, values, limit_to=reps)
                if summary not in items:
                    items.append(summary)
            else:
                value = _get_attr_or_key(expt, param, None)
                if value is not None:
                    summary = self._sequence_summary_text(param, value)
                    if summary not in items:
                        items.append(summary)
        return items[:12]

    def _fold_repeated_block_patterns(
        self,
        blocks: Sequence[CircuitBlock],
        *,
        min_repeats: int = 3,
        max_pattern_len: int = 24,
        min_saved_blocks: int = 8,
    ) -> List[CircuitBlock]:
        if len(blocks) < min_repeats:
            return list(blocks)
        sigs = [self._block_pattern_signature(block) for block in blocks]
        folded: List[CircuitBlock] = []
        i = 0
        n = len(blocks)
        while i < n:
            if sigs[i] is None:
                folded.append(blocks[i])
                i += 1
                continue

            best: Optional[Tuple[int, int, int]] = None
            max_len = min(max_pattern_len, (n - i) // min_repeats)
            for pat_len in range(1, max_len + 1):
                pattern = sigs[i : i + pat_len]
                if any(sig is None for sig in pattern):
                    break
                reps = 1
                while i + (reps + 1) * pat_len <= n and sigs[i + reps * pat_len : i + (reps + 1) * pat_len] == pattern:
                    reps += 1
                saved = (reps - 1) * pat_len
                if reps >= min_repeats and saved >= min_saved_blocks:
                    if best is None or saved > best[2] or (saved == best[2] and pat_len < best[0]):
                        best = (pat_len, reps, saved)

            if best is None:
                folded.append(blocks[i])
                i += 1
                continue

            best = self._prefer_context_loop_pattern(
                best,
                blocks,
                start=i,
                min_repeats=min_repeats,
                max_pattern_len=max_pattern_len,
                min_saved_blocks=min_saved_blocks,
            )
            pat_len, reps, _ = best
            first_pattern = list(blocks[i : i + pat_len])
            segment = list(blocks[i : i + pat_len * reps])
            first_pattern[0].pattern_count = reps
            first_pattern[0].pattern_size = pat_len
            first_pattern[0].pattern_text = self._fold_pattern_label(reps, pat_len)
            first_pattern[0].pattern_notes = tuple(self._context_pattern_notes(segment, pat_len, reps))
            self._annotate_context_pattern_blocks(first_pattern)
            for pos, block in enumerate(first_pattern):
                repeated_position = segment[pos : pat_len * reps : pat_len]
                variation_text = self._block_position_variation_text(repeated_position)
                if variation_text:
                    if block.params_text:
                        block.params_text = f"{block.params_text}; {variation_text}"
                    else:
                        block.params_text = variation_text
            folded.extend(first_pattern)
            i += pat_len * reps
        return folded

    def _elide_long_circuit_blocks(
        self,
        blocks: Sequence[CircuitBlock],
        *,
        max_visible_blocks: Optional[int],
    ) -> List[CircuitBlock]:
        if max_visible_blocks is None or len(blocks) <= max_visible_blocks:
            return list(blocks)
        max_visible = max(int(max_visible_blocks), 12)
        body_idxs = [idx for idx, block in enumerate(blocks) if self._block_section(block) == "body"]
        if len(body_idxs) <= max_visible:
            return list(blocks)

        keep_body = max(6, max_visible - (len(blocks) - len(body_idxs)) - 1)
        keep_head = max(3, int(keep_body * 0.65))
        keep_tail = max(3, keep_body - keep_head)
        if keep_head + keep_tail >= len(body_idxs):
            return list(blocks)

        start = body_idxs[keep_head]
        end = body_idxs[-keep_tail]
        omitted = max(0, end - start)
        summary = f"{omitted} body ops folded; set max_visible_blocks=None to expand"
        fold_event = PulseEvent(
            t0=blocks[start].t0,
            duration=0.02,
            ch="LOOP",
            kind="fold",
            label="body folded",
            params={"summary": summary, "omitted_blocks": omitted},
            source="body",
        )
        fold_block = CircuitBlock(ch="LOOP", kind="fold", label="body folded", params_text=summary, events=[fold_event])
        out = list(blocks[:start]) + [fold_block] + list(blocks[end:])
        self._assign_circuit_steps(out)
        return out

    def _metadata_name(self, key: str) -> str:
        value = self.metadata.get(key)
        if value is None:
            return ""
        if isinstance(value, str):
            return value.rsplit(".", 1)[-1]
        name = getattr(value, "__name__", None)
        if name:
            return str(name)
        cls = getattr(value, "__class__", None)
        return str(getattr(cls, "__name__", "")) if cls is not None else ""

    def _default_circuit_title(self) -> str:
        program = self._metadata_name("program_class") or self._metadata_name("program_name")
        experiment = self._metadata_name("experiment_class") or self._metadata_name("experiment_name")
        if program and experiment:
            return f"Pulse sequence blocks\nProgram: {program} | Experiment: {experiment}"
        if program:
            return f"Pulse sequence blocks\nProgram: {program}"
        if experiment:
            return f"Pulse sequence blocks\nExperiment: {experiment}"
        return "Pulse sequence blocks"

    def to_circuit_blocks(
        self,
        *,
        show_sync: bool = False,
        group_repeats: bool = True,
        group_sweeps: bool = True,
        group_param_changes: bool = True,
        fold_repeated_patterns: bool = True,
        max_group_gap: float = 1e-9,
    ) -> List[CircuitBlock]:
        """Return execution-order blocks for a circuit-style diagram."""
        selected = []
        indexed_events = list(enumerate(self.events))
        for _, event in sorted(indexed_events, key=lambda item: (item[1].t0, item[0])):
            if not show_sync and (event.ch in {"SYNC", "WAIT"} or event.kind in {"sync", "wait"}):
                continue
            selected.append(event)

        blocks: List[CircuitBlock] = []
        signatures: List[Tuple[Any, ...]] = []
        for event in selected:
            label, params_text = self._circuit_label(event)
            sig = self._circuit_signature(
                event,
                group_sweeps=group_sweeps,
                group_param_changes=group_param_changes,
            )
            if (
                group_repeats
                and blocks
                and signatures[-1] == sig
                and self._block_loop_context_key(blocks[-1]) == self._event_loop_context_key(event)
                and abs(event.t0 - blocks[-1].t1) <= max_group_gap
            ):
                blocks[-1].events.append(event)
                blocks[-1].count += 1
                variation_text = self._block_param_variation_text(blocks[-1])
                if variation_text:
                    blocks[-1].params_text = variation_text
                elif params_text and params_text not in blocks[-1].params_text:
                    blocks[-1].params_text = "varies"
                continue
            block = CircuitBlock(
                ch=event.ch,
                kind=event.kind,
                label=label,
                params_text=params_text,
                events=[event],
            )
            block.sweep_text = self._block_sweep_text(block)
            blocks.append(block)
            signatures.append(sig)

        if fold_repeated_patterns:
            blocks = self._fold_repeated_block_patterns(blocks)
        self._assign_circuit_steps(blocks, max_group_gap=max_group_gap)
        return blocks

    def _block_section(self, block: CircuitBlock) -> str:
        source = ""
        if block.events:
            source = str(block.events[0].source or "")
        if block.kind == "fold":
            return "body"
        if block.kind == "init" or "initialize" in source:
            return "initialize"
        if block.kind == "update" or "update" in source:
            return "update"
        if block.kind == "loop":
            return "loop"
        if "body" in source:
            return "body"
        return ""

    def _section_spans(self, blocks: Sequence[CircuitBlock]) -> List[Tuple[str, int, int]]:
        priority = {"update": 4, "body": 3, "initialize": 2, "loop": 1, "": 0}
        section_by_step: Dict[int, str] = {}
        for block in blocks:
            section = self._block_section(block)
            current = section_by_step.get(block.step, "")
            if priority.get(section, 0) >= priority.get(current, 0):
                section_by_step[block.step] = section

        spans: List[Tuple[str, int, int]] = []
        active = ""
        start: Optional[int] = None
        prev: Optional[int] = None
        for step in sorted(section_by_step):
            section = section_by_step[step]
            if not section:
                if active and start is not None and prev is not None:
                    spans.append((active, start, prev))
                active, start, prev = "", None, None
                continue
            if section != active or (prev is not None and step != prev + 1):
                if active and start is not None and prev is not None:
                    spans.append((active, start, prev))
                active, start = section, step
            prev = step
        if active and start is not None and prev is not None:
            spans.append((active, start, prev))
        return spans

    def _marker_step(self, marker: TraceMarker, blocks: Sequence[CircuitBlock]) -> int:
        if not blocks:
            return 0
        for block in blocks:
            if block.t0 - 1e-9 <= marker.t <= block.t1 + 1e-9:
                return block.step
        nearest = min(blocks, key=lambda block: min(abs(marker.t - block.t0), abs(marker.t - block.t1)))
        return nearest.step

    def _conditional_marker_text(self, marker: TraceMarker) -> str:
        params = marker.params or {}
        decoded = params.get("decoded")
        if decoded:
            return str(decoded)
        args = list(params.get("args", ()))
        if len(args) >= 5:
            _, reg, op, value, target = args[:5]
            return (
                f"if {_short_value(reg, 18)} {op} {_short_value(value, 18)}"
                f" -> {_short_value(target, 18)}"
            )
        if len(args) >= 4:
            return "if " + " ".join(_short_value(arg, 14) for arg in args[:4])
        return "conditional branch"

    def _conditional_summary_items(self, *, limit: int = 6) -> List[str]:
        items = []
        for marker in self.markers:
            if marker.kind != "condj":
                continue
            text = self._conditional_marker_text(marker)
            if text not in items:
                items.append(text)
            if len(items) >= limit:
                break
        return [f"conditional: {item}" for item in items]

    def _text_bounds(
        self,
        x: float,
        y: float,
        text: str,
        *,
        fontsize: float,
        step_spacing: float,
        va: str,
    ) -> Tuple[float, float, float, float]:
        lines = str(text).splitlines() or [str(text)]
        max_chars = max((len(line) for line in lines), default=1)
        width = max(0.28, min(2.35 * step_spacing, max_chars * 0.036 * step_spacing * (fontsize / 7.5)))
        height = max(0.14, len(lines) * 0.17 * (fontsize / 7.5))
        x0, x1 = x - width / 2, x + width / 2
        if va == "top":
            y0, y1 = y - height, y
        elif va == "center":
            y0, y1 = y - height / 2, y + height / 2
        else:
            y0, y1 = y, y + height
        return x0, x1, y0, y1

    def _boxes_overlap(self, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float], *, pad: float = 0.035) -> bool:
        return not (a[1] + pad < b[0] or b[1] + pad < a[0] or a[3] + pad < b[2] or b[3] + pad < a[2])

    def _pick_text_position(
        self,
        occupied: MutableMapping[str, List[Tuple[float, float, float, float]]],
        lane: str,
        x: float,
        candidates: Sequence[float],
        text: str,
        *,
        fontsize: float,
        step_spacing: float,
        va: str,
    ) -> float:
        boxes = occupied.setdefault(lane, [])
        last_y = candidates[-1] if candidates else 0.0
        last_box = self._text_bounds(x, last_y, text, fontsize=fontsize, step_spacing=step_spacing, va=va)
        for cand_y in candidates:
            box = self._text_bounds(x, cand_y, text, fontsize=fontsize, step_spacing=step_spacing, va=va)
            if not any(self._boxes_overlap(box, prior) for prior in boxes):
                boxes.append(box)
                return cand_y
            last_y, last_box = cand_y, box
        boxes.append(last_box)
        return last_y

    def _summary_item_text(self, text: Any) -> str:
        out = str(text).strip()
        out = re.sub(r"^pulse_regs\.([^.:]+)\.", r"pulse ch\1 ", out)
        out = re.sub(r"^default_regs\.([^.:]+)\.", r"default ch\1 ", out)
        register_match = re.match(r"^registers\.\((.+?)\)(.*)$", out)
        if register_match:
            tuple_text, suffix = register_match.groups()
            reg_name = tuple_text.split(",", 1)[-1].strip().strip("'\"")
            out = f"register {reg_name}{suffix}"
        out = out.replace("body not expanded: ", "")
        out = out.replace("update summarized", "update summarized")
        out = re.sub(r"\s+", " ", out)
        return self._compact_summary_transition(out)

    def _compact_summary_transition(self, text: str) -> str:
        match = re.match(r"^(.+?):\s*(\S+)\s*->\s*(\S+)$", text)
        if not match:
            return text
        name, old, new = match.groups()
        old_f = _try_float(old, None)
        new_f = _try_float(new, None)
        if old_f is None or new_f is None or not (stdlib_math.isfinite(old_f) and stdlib_math.isfinite(new_f)):
            return text
        if name.startswith("register "):
            delta = new_f - old_f
            delta_text = self._format_circuit_value(delta)
            if delta > 0:
                delta_text = "+" + delta_text
            return f"{name} {delta_text}"
        return f"{name}: {self._format_circuit_value(old_f)} -> {self._format_circuit_value(new_f)}"

    def _split_summary_items(self, summary: str) -> List[str]:
        pieces: List[str] = []
        for part in re.split(r";\s*|,\s*(?=[A-Za-z_]+=)", str(summary)):
            part = part.strip()
            if part:
                pieces.append(self._summary_item_text(part))
        return pieces

    def _loop_summary_cards(self, blocks: Sequence[CircuitBlock], *, width: int = 52) -> List[Tuple[str, str, List[str]]]:
        cards: List[Tuple[str, str, List[str]]] = []
        pattern_items: List[str] = []
        for idx, block in enumerate(blocks):
            if block.pattern_count <= 1:
                continue
            group = blocks[idx : idx + block.pattern_size]
            context_items = self._pattern_context_items(block)
            update_items = []
            for pattern_block in group:
                if "varies:" in pattern_block.params_text:
                    label = self._normalize_circuit_label(pattern_block.label)
                    variation = pattern_block.params_text.split("varies:", 1)[1].strip()
                    update_items.append(f"{label}: {variation}")
            pattern_items.append(block.pattern_text or f"pattern x{block.pattern_count}")
            loop_index_items = [
                item for item in context_items
                if item.startswith("loop index") or item.startswith("ops per")
            ]
            dynamic_context_items = [
                item for item in context_items
                if item not in loop_index_items and ("[j]" in item or "[i]" in item or item.startswith("loop unit"))
            ]
            static_context_items = [
                item for item in context_items
                if item not in loop_index_items and item not in dynamic_context_items
            ]
            pattern_items.extend(dynamic_context_items[:8])
            pattern_items.extend(update_items[:6])
            pattern_items.extend(self._conditional_summary_items(limit=4))
            pattern_items.extend(static_context_items[:5])
            pattern_items.extend(loop_index_items)
            pattern_items.append(f"shown: first {block.pattern_size} ops")
            break
        if not pattern_items:
            for block in blocks:
                if block.kind != "fold":
                    continue
                pattern_items.append("compact body view")
                pattern_items.extend(self._split_summary_items(block.params_text)[:3])
                break
        if not pattern_items:
            for block in blocks:
                if block.kind != "pulse" or block.count <= 1 or not block.params_text.startswith("varies:"):
                    continue
                label = self._normalize_circuit_label(block.label)
                pattern_items.append(f"{label} x{block.count}")
                pattern_items.append(block.params_text.replace("varies: ", "updates: "))
                break
        if not pattern_items:
            condition_items = self._conditional_summary_items(limit=6)
            if condition_items:
                cards.append(("body", "Conditions", condition_items))
        for block in blocks:
            if block.kind not in {"init", "loop", "update"}:
                continue
            summary = block.params_text.strip()
            if not summary:
                continue
            title = {"init": "Initialize", "loop": "Hardware Loop", "update": "Update"}.get(block.kind, block.label)
            items: List[str] = []
            if block.kind == "loop":
                for key in ("reps", "rounds", "expts"):
                    value = (block.events[0].params or {}).get(key) if block.events else None
                    if value is not None:
                        items.append(f"{key}: {self._format_circuit_value(value)}")
                if "update summarized" in summary:
                    items.append("update: summarized")
            elif block.kind == "update" and block.events:
                params = block.events[0].params or {}
                items.extend(self._summary_item_text(item) for item in params.get("changes", [])[:3])
                if not items:
                    items.extend(self._summary_item_text(item) for item in params.get("markers", [])[:3])
            elif block.kind == "init" and block.events:
                counts = (block.events[0].params or {}).get("counts", {})
                if isinstance(counts, Mapping):
                    for key, value in counts.items():
                        items.append(f"{key.replace('_', ' ')}: {value}")
            if not items:
                items = self._split_summary_items(summary)
            cleaned = []
            for item in items[:5]:
                wrapped = self._fit_text_lines(item, width=width, max_lines=2).replace("\n", "\n  ")
                cleaned.append(wrapped)
            cards.append((block.kind, title, cleaned or [summary]))
        if pattern_items:
            insert_at = 1 if cards and cards[0][0] == "init" else 0
            compact_pattern_items = [self._fit_text_lines(item, width=68, max_lines=1) for item in pattern_items[:14]]
            cards.insert(insert_at, ("body", "Body Pattern", compact_pattern_items))
        return cards

    def _draw_loop_summary_cards(
        self,
        fig: Any,
        cards: Sequence[Tuple[str, str, List[str]]],
        *,
        left: float,
        right: float,
        bottom: float = 0.035,
        height: float = 0.16,
    ) -> None:
        if not cards:
            return
        summary_ax = fig.add_axes([left, bottom, right - left, height], zorder=30)
        summary_ax.set_xlim(0, 1)
        summary_ax.set_ylim(0, 1)
        summary_ax.axis("off")

        card_styles = {
            "init": ("#eef6ff", "#bcd8f2", "#2468a8"),
            "body": ("#eef8f0", "#c6dfcc", "#2f7440"),
            "loop": ("#f3f5f8", "#cfd5dd", "#3f444a"),
            "update": ("#f6f0ff", "#d7c5f5", "#7750bd"),
        }
        visible_cards = list(cards[:4])
        gap = 0.012
        card_w = (1.0 - gap * (len(visible_cards) - 1)) / max(len(visible_cards), 1)
        for idx, (kind, card_title, items) in enumerate(visible_cards):
            face, edge, title_color = card_styles.get(kind, ("#ffffff", "#d7dce3", "#30343a"))
            x0 = idx * (card_w + gap)
            x1 = x0 + card_w
            y0, y1 = 0.08, 0.94
            summary_ax.fill_between([x0, x1], [y0, y0], [y1, y1], color=face, alpha=0.98, linewidth=0)
            summary_ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=edge, linewidth=0.9)
            summary_ax.text(
                x0 + 0.018,
                y1 - 0.10,
                card_title,
                ha="left",
                va="top",
                fontsize=8.0,
                color=title_color,
                fontweight="semibold",
            )

            if kind == "body":
                y_text = y1 - 0.20
                line_step = 0.058
                item_limit = 14
                font_size = 5.6
                max_lines_per_item = 1
            else:
                y_text = y1 - 0.28
                line_step = 0.145
                item_limit = 8
                font_size = 7.3
                max_lines_per_item = 2
            for item in items[:item_limit]:
                lines = str(item).splitlines() or [str(item)]
                for line_no, line in enumerate(lines[:max_lines_per_item]):
                    prefix = "- " if line_no == 0 else "  "
                    summary_ax.text(
                        x0 + 0.026,
                        y_text,
                        prefix + line.strip(),
                        ha="left",
                        va="top",
                        fontsize=font_size,
                        color="#30343a",
                    )
                    y_text -= line_step
                if y_text < y0 + 0.045:
                    break

    def _draw_circuit_box(self, ax: Any, x: float, y: float, w: float, h: float, color: str) -> None:
        x0, x1 = x - w / 2, x + w / 2
        y0, y1 = y - h / 2, y + h / 2
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=color, linewidth=1.2)

    def _draw_measure_symbol(self, ax: Any, x: float, y: float, w: float, h: float, color: str) -> None:
        box_w = w * 0.72
        box_h = h * 0.92
        self._draw_circuit_box(ax, x, y, box_w, box_h, color)
        arc_center_y = y - box_h * 0.18
        radius = min(box_w * 0.28, box_h * 0.42)
        angles = [stdlib_math.radians(a) for a in range(20, 161, 5)]
        xs = [x + radius * stdlib_math.cos(a) for a in angles]
        ys = [arc_center_y + radius * stdlib_math.sin(a) for a in angles]
        ax.plot(xs, ys, color=color, linewidth=1.1, solid_capstyle="round")
        ax.plot(
            [x, x + radius * 0.62],
            [arc_center_y, arc_center_y + radius * 1.22],
            color=color,
            linewidth=1.1,
            solid_capstyle="round",
        )

    def _lighten_color(self, color: str, amount: float = 0.72) -> Tuple[float, float, float]:
        try:
            import matplotlib.colors as mcolors

            r, g, b = mcolors.to_rgb(color)
            return (
                r + (1.0 - r) * amount,
                g + (1.0 - g) * amount,
                b + (1.0 - b) * amount,
            )
        except Exception:
            return (0.86, 0.91, 0.96)

    def _normalized_envelope_samples(self, samples: Any) -> List[float]:
        if samples is None:
            return []
        try:
            if hasattr(samples, "tolist"):
                samples = samples.tolist()
        except Exception:
            pass
        if not isinstance(samples, (list, tuple)):
            return []
        vals = []
        for value in samples:
            val = _try_float(value, None)
            if val is not None and stdlib_math.isfinite(val):
                vals.append(abs(float(val)))
        if len(vals) < 2:
            return []
        vmax = max(vals)
        vmin = min(vals)
        if vmax <= 0:
            return []
        span = vmax - vmin
        if span <= max(1e-12, 0.015 * vmax):
            return []
        vals = [(v - vmin) / span for v in vals]
        max_points = 80
        if len(vals) > max_points:
            step = (len(vals) - 1) / (max_points - 1)
            vals = [vals[int(round(i * step))] for i in range(max_points)]
        if len(vals) >= 7:
            for _ in range(2):
                vals = [vals[0]] + [
                    0.25 * vals[i - 1] + 0.5 * vals[i] + 0.25 * vals[i + 1]
                    for i in range(1, len(vals) - 1)
                ] + [vals[-1]]
            vmin = min(vals)
            vmax = max(vals)
            span = vmax - vmin
            if span > 1e-12:
                vals = [(v - vmin) / span for v in vals]
        return vals

    def _draw_envelope_samples(
        self,
        ax: Any,
        samples: Any,
        x0: float,
        x1: float,
        base: float,
        top: float,
        color: str,
        *,
        fill: bool = True,
    ) -> bool:
        vals = self._normalized_envelope_samples(samples)
        if not vals:
            return False
        xs = [x0 + (x1 - x0) * i / (len(vals) - 1) for i in range(len(vals))]
        ys = [base + (top - base) * v for v in vals]
        if fill:
            ax.fill_between(xs, ys, [base] * len(xs), color=self._lighten_color(color, 0.82), alpha=0.24, linewidth=0)
        ax.plot(xs, ys, color=color, linewidth=2.15, solid_capstyle="round")
        ax.plot([x0, x1], [base, base], color="#8b949e", linewidth=0.55, alpha=0.38)
        return True

    def _flat_top_uses_gaussian_ramp(self, params: Mapping[str, Any]) -> bool:
        kind = str(params.get("waveform_kind", "") or params.get("shape_kind", "") or "").lower()
        waveform = str(params.get("waveform", "") or "").lower()
        if kind in {"gaussian", "gauss", "drag"}:
            return True
        return any(token in waveform for token in ("gauss", "ramp", "temp_gaussian", "pi_qubit", "hpi_qubit"))

    def _gaussian_edge_samples(self, n: int = 30) -> List[float]:
        n = max(int(n), 3)
        vals = []
        for i in range(n):
            t = i / (n - 1)
            z = (1.0 - t) * 2.8
            vals.append(stdlib_math.exp(-0.5 * z * z))
        v0, v1 = vals[0], vals[-1]
        span = max(v1 - v0, 1e-12)
        return [(v - v0) / span for v in vals]

    def _draw_flat_top_envelope(self, ax: Any, params: Mapping[str, Any], x0: float, x1: float, y: float, color: str) -> None:
        w = x1 - x0
        base = y + 0.02
        top = y + 0.36
        gaussian_ramp = self._flat_top_uses_gaussian_ramp(params)
        flat_len = _try_float(params.get("length"), None)
        ramp_len = _try_float(params.get("envelope_length") or params.get("waveform_length"), None)
        if flat_len is not None and ramp_len is not None and flat_len + ramp_len > 0:
            physical_plateau = flat_len / (flat_len + ramp_len)
            plateau_frac = min(0.74, max(0.48, 0.30 + 0.46 * physical_plateau))
        else:
            plateau_frac = 0.58
        min_edge_frac = 0.23 if gaussian_ramp else 0.12
        edge_w = max(min_edge_frac * w, (1.0 - plateau_frac) * w / 2)
        edge_w = min(edge_w, 0.32 * w)
        left_edge = x0 + edge_w
        right_edge = x1 - edge_w

        ramp = self._normalized_envelope_samples(params.get("envelope_samples"))
        if gaussian_ramp:
            rise = self._gaussian_edge_samples(30)
            fall = list(reversed(rise))
            rx = [x0 + edge_w * i / (len(rise) - 1) for i in range(len(rise))]
            fx = [right_edge + edge_w * i / (len(fall) - 1) for i in range(len(fall))]
            ax.plot(rx, [base + (top - base) * v for v in rise], color=color, linewidth=2.1, solid_capstyle="round")
            ax.plot(fx, [base + (top - base) * v for v in fall], color=color, linewidth=2.1, solid_capstyle="round")
        elif ramp:
            peak = max(range(len(ramp)), key=lambda i: ramp[i])
            rise = ramp[: peak + 1] or ramp
            fall = ramp[peak:] or ramp
            rise = self._normalized_envelope_samples(rise)
            fall = self._normalized_envelope_samples(fall)
            if rise:
                rx = [x0 + edge_w * i / (len(rise) - 1) for i in range(len(rise))]
                ry = [base + (top - base) * v for v in rise]
                ax.plot(rx, ry, color=color, linewidth=2.1, solid_capstyle="round")
            if fall:
                fx = [right_edge + edge_w * i / (len(fall) - 1) for i in range(len(fall))]
                fy = [base + (top - base) * v for v in fall]
                ax.plot(fx, fy, color=color, linewidth=2.1, solid_capstyle="round")
        else:
            ax.plot([x0, left_edge], [base, top], color=color, linewidth=2.1, solid_capstyle="round")
            ax.plot([right_edge, x1], [top, base], color=color, linewidth=2.1, solid_capstyle="round")

        xs = [x0, left_edge, right_edge, x1]
        ys = [base, top, top, base]
        ax.fill_between(xs, ys, [base] * len(xs), color=self._lighten_color(color, 0.78), alpha=0.24, linewidth=0)
        ax.plot([left_edge, right_edge], [top, top], color=color, linewidth=3.0, solid_capstyle="butt", alpha=0.95)
        ax.plot([x0, x1], [base, base], color="#8b949e", linewidth=0.55, alpha=0.38)

    def _infer_circuit_waveform_kind(self, block: CircuitBlock, params: Mapping[str, Any]) -> str:
        style = str(params.get("style", block.kind)).lower()
        waveform = str(params.get("waveform", "") or "").lower()
        kind = str(params.get("waveform_kind", "") or params.get("shape_kind", "") or "").lower()
        if style in {"flat_top", "flattop", "flat"}:
            return "flat_top"
        if style == "const":
            return "const"
        if kind:
            return kind
        if style in {"gauss", "gaussian"}:
            return "gaussian"
        if style == "arb":
            if any(token in waveform for token in ("gauss", "pi_qubit", "pi_ge", "pi_ef", "drag")):
                return "gaussian"
            if any(token in waveform for token in ("opt", "oc", "gkp", "_q", "_m", "iq")):
                return "arbitrary"
            return "arbitrary"
        return style

    def _draw_circuit_waveform(self, ax: Any, block: CircuitBlock, x: float, y: float, w: float, h: float, color: str) -> None:
        event = block.events[0] if block.events else None
        params = event.params if event is not None else {}
        waveform_kind = self._infer_circuit_waveform_kind(block, params or {})
        x0, x1 = x - w / 2, x + w / 2
        mid = y + 0.04
        amp = h / 2

        if block.kind in {"measure", "trigger"} or block.ch.startswith("ADC"):
            self._draw_measure_symbol(ax, x, y, w, h, color)
            return
        if block.kind in {"sync", "wait"}:
            ax.plot([x0, x1], [y, y], color=color, linewidth=2.0)
            ax.plot([x0, x0], [y - h * 0.18, y + h * 0.18], color=color, linewidth=1.0)
            ax.plot([x1, x1], [y - h * 0.18, y + h * 0.18], color=color, linewidth=1.0)
            return
        if block.kind in {"init", "loop", "update", "fold"}:
            cy = y + 0.14
            self._draw_circuit_box(ax, x, cy, w * 0.88, h * 0.95, color)
            ax.plot([x - w * 0.18, x + w * 0.18], [cy, cy], color=color, linewidth=1.2)
            ax.plot([x + w * 0.10, x + w * 0.18, x + w * 0.10], [cy + h * 0.08, cy, cy - h * 0.08], color=color, linewidth=1.2)
            return

        if waveform_kind == "flat_top":
            self._draw_flat_top_envelope(ax, params or {}, x0, x1, y, color)
        elif waveform_kind in {"gaussian", "gauss"}:
            base = y + 0.02
            top = y + 0.36
            if not self._draw_envelope_samples(ax, params.get("envelope_samples"), x0, x1, base, top, color):
                xs = [x0 + w * i / 24 for i in range(25)]
                sigma = 0.16
                ys = []
                for xx in xs:
                    z = (xx - x) / max(w, 1e-9)
                    ys.append(base + (top - base) * stdlib_math.exp(-0.5 * (z / sigma) ** 2))
                ax.fill_between(xs, ys, [base] * len(xs), color=self._lighten_color(color, 0.82), alpha=0.22, linewidth=0)
                ax.plot(xs, ys, color=color, linewidth=2.0, solid_capstyle="round")
                ax.plot([x0, x1], [base, base], color="#8b949e", linewidth=0.55, alpha=0.38)
        elif waveform_kind == "drag":
            xs = [x0 + w * i / 28 for i in range(29)]
            sigma = 0.18
            gauss = []
            quad = []
            for xx in xs:
                z = (xx - x) / max(w, 1e-9)
                env = stdlib_math.exp(-0.5 * (z / sigma) ** 2)
                gauss.append(mid + amp * env)
                quad.append(mid + 0.45 * amp * (-z / sigma) * env)
            ax.plot(xs, gauss, color=color, linewidth=1.7)
            ax.plot(xs, quad, color=color, linewidth=1.0, linestyle="--", alpha=0.85)
            ax.plot([x0, x1], [mid, mid], color="0.55", linewidth=0.6, alpha=0.8)
        elif waveform_kind == "triangle":
            xs = [x0, x, x1]
            ys = [mid, mid + amp, mid]
            ax.plot(xs, ys, color=color, linewidth=1.8)
            ax.plot([x0, x1], [mid, mid], color="0.55", linewidth=0.6, alpha=0.8)
        elif waveform_kind in {"arbitrary", "custom", "iq", "opt_cont"}:
            base = y + 0.02
            top = y + 0.36
            if not self._draw_envelope_samples(ax, params.get("envelope_samples"), x0, x1, base, top, color):
                xs = [x0 + w * i / 16 for i in range(17)]
                pattern = [0.00, 0.38, 0.22, 0.68, 0.48, 0.85, 0.42, 0.72, 0.30, 0.62, 0.24, 0.76, 0.45, 0.58, 0.20, 0.36, 0.00]
                ys = [base + (top - base) * p for p in pattern]
                ax.plot(xs, ys, color=color, linewidth=1.65)
                qys = [base + 0.18 * stdlib_math.sin(i * stdlib_math.pi / 4) for i in range(17)]
                ax.plot(xs, qys, color=color, linewidth=0.9, linestyle=":", alpha=0.8)
                ax.plot([x0, x1], [base, base], color="#8b949e", linewidth=0.55, alpha=0.38)
        elif waveform_kind == "const":
            xs = [x0, x0, x1, x1]
            ys = [mid, mid + amp, mid + amp, mid]
            ax.plot(xs, ys, color=color, linewidth=1.8)
            ax.plot([x0, x1], [mid, mid], color="0.55", linewidth=0.6, alpha=0.8)
        else:
            self._draw_circuit_box(ax, x, y, w, h, color)

    def _draw_repeat_power(self, ax: Any, x: float, y: float, count: int, color: str, *, scale: float = 1.0,
                           text: Optional[str] = None) -> None:
        left = x - 0.58 * scale
        right = x + 0.58 * scale
        ax.text(left, y + 0.02, "(", ha="center", va="center", fontsize=28, color="#30343a", alpha=0.78, zorder=1)
        ax.text(right, y + 0.02, ")", ha="center", va="center", fontsize=28, color="#30343a", alpha=0.78, zorder=1)
        repeat_text = text or f"$\\times {count}$"
        if len(repeat_text) > 34:
            repeat_text = self._fit_text_lines(repeat_text, width=34, max_lines=2)
        ax.text(
            right + 0.03 * scale,
            y + 0.30,
            repeat_text,
            ha="left",
            va="bottom",
            fontsize=7.4 if len(repeat_text) > 16 else 8.6,
            color="#202327",
            zorder=6,
        )

    def _draw_pattern_fold_annotation(
        self,
        ax: Any,
        blocks: Sequence[CircuitBlock],
        start_idx: int,
        *,
        y_of: Mapping[str, float],
        step_spacing: float,
        plot_ymax: float,
        text_effects: Sequence[Any],
    ) -> None:
        start = blocks[start_idx]
        group = blocks[start_idx : start_idx + start.pattern_size]
        if start.pattern_count <= 1 or not group:
            return
        x0 = min(block.step for block in group) * step_spacing - 0.40 * step_spacing
        x1 = max(block.step for block in group) * step_spacing + 0.40 * step_spacing
        local_top = max(y_of.get(block.ch, 0.0) for block in group)
        y = min(plot_ymax - 0.52, local_top + 1.12)
        color = "#30343a"
        ax.plot([x0, x0, x1, x1], [y - 0.08, y, y, y - 0.08], color=color, linewidth=1.15, zorder=12)
        ax.text(
            (x0 + x1) / 2,
            y + 0.10,
            start.pattern_text or f"pattern x{start.pattern_count}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=color,
            fontweight="semibold",
            path_effects=text_effects,
            zorder=13,
        )

    def plot_circuit(
        self,
        *,
        show_sync: bool = False,
        group_repeats: bool = True,
        group_sweeps: bool = True,
        group_param_changes: bool = True,
        fold_repeated_patterns: bool = True,
        annotate_params: Union[bool, str] = True,
        annotate_sweep: bool = True,
        waveform_blocks: bool = True,
        detail: str = "phase",
        phase_mode: str = "relative",
        show_zero_phase: bool = False,
        max_label_chars: int = 18,
        step_spacing: float = 1.55,
        row_spacing: float = 1.60,
        detail_wrap_chars: int = 22,
        max_detail_lines: int = 2,
        loop_summary_panel: bool = True,
        annotate_conditionals: bool = True,
        max_visible_blocks: Optional[int] = 180,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
    ):
        """
        Plot an execution-order block diagram, similar to a quantum circuit.

        Unlike plot(), the x-axis is block order rather than elapsed time.  Adjacent
        repeated operations are collapsed into one block with an xN annotation.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as path_effects

        if isinstance(annotate_params, str):
            detail = annotate_params
            annotate_params_enabled = detail.lower() not in {"none", "off", "false"}
        else:
            annotate_params_enabled = bool(annotate_params)

        blocks = self.to_circuit_blocks(
            show_sync=show_sync,
            group_repeats=group_repeats,
            group_sweeps=group_sweeps,
            group_param_changes=group_param_changes,
            fold_repeated_patterns=fold_repeated_patterns,
        )
        blocks = self._elide_long_circuit_blocks(blocks, max_visible_blocks=max_visible_blocks)
        chans = sorted({b.ch for b in blocks})
        if not chans:
            fig, ax = plt.subplots(figsize=figsize or (10, 2))
            ax.set_title(title or "No pulse events recorded", fontsize=13, fontweight="semibold")
            return fig, ax

        step_spacing = max(_try_float(step_spacing, 1.55) or 1.55, 0.95)
        row_spacing = max(_try_float(row_spacing, 1.60) or 1.60, 1.05)
        y_of = {ch: i * row_spacing for i, ch in enumerate(chans)}
        max_step = max((b.step for b in blocks), default=0)
        top_y = (len(chans) - 1) * row_spacing
        summary_cards = self._loop_summary_cards(blocks) if loop_summary_panel else []
        if figsize is None:
            figsize = (
                max(11.5, 0.86 * step_spacing * (max_step + 1) + 3.2),
                max(3.5, 0.82 * row_spacing * max(len(chans), 1) + 2.8 + (1.35 if summary_cards else 0.0)),
            )
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#fbfcfd")
        text_effects = [path_effects.withStroke(linewidth=2.2, foreground=ax.get_facecolor())]

        xmin, xmax = -0.85 * step_spacing, (max_step + 0.85) * step_spacing
        plot_ymin = -0.85
        has_patterns = any(block.pattern_count > 1 for block in blocks)
        has_conditionals = annotate_conditionals and any(marker.kind == "condj" for marker in self.markers)
        top_padding = 0.48
        if annotate_sweep:
            top_padding = max(top_padding, 1.12)
        if has_patterns:
            top_padding = max(top_padding, 1.52)
        if has_conditionals:
            top_padding = max(top_padding, 1.72)
        plot_ymax = top_y + top_padding
        section_styles = {
            "initialize": "#e9f3ff",
            "body": "#eef7f0",
            "update": "#f3ecff",
            "loop": "#f1f3f6",
        }
        section_text_colors = {
            "initialize": "#2468a8",
            "body": "#2f7440",
            "update": "#7750bd",
            "loop": "#59616b",
        }
        for section, start, end in self._section_spans(blocks):
            color = section_styles.get(section)
            if not color:
                continue
            x0 = (start - 0.48) * step_spacing
            x1 = (end + 0.48) * step_spacing
            ax.fill_between(
                [x0, x1],
                [plot_ymin, plot_ymin],
                [plot_ymax, plot_ymax],
                color=color,
                alpha=0.46,
                linewidth=0,
                zorder=-5,
            )
            ax.text(
                (x0 + x1) / 2,
                plot_ymax - 0.08,
                section,
                ha="center",
                va="top",
                fontsize=7.5,
                color=section_text_colors.get(section, "#59616b"),
                fontweight="semibold",
                zorder=-1,
            )
        channel_aliases = self._hardware_channel_aliases()
        for ch, y in y_of.items():
            is_flow_row = str(ch) == "LOOP"
            if is_flow_row:
                ax.fill_between(
                    [xmin, xmax],
                    [y - 0.38, y - 0.38],
                    [y + 0.50, y + 0.50],
                    color="#fff1c9",
                    alpha=0.62,
                    linewidth=0,
                    zorder=-4,
                )
            ax.plot(
                [xmin, xmax],
                [y, y],
                color="#c08016" if is_flow_row else "#d5d9df",
                linewidth=1.35 if is_flow_row else 1.05,
                linestyle="--" if is_flow_row else "-",
                alpha=0.95 if is_flow_row else 1.0,
                zorder=0,
            )
            main_ch, sub_ch = self._channel_label_parts(ch, channel_aliases)
            ax.text(
                xmin - 0.20 * step_spacing,
                y + (0.10 if sub_ch else 0.0),
                main_ch,
                ha="right",
                va="center",
                fontsize=9.2,
                fontweight="semibold",
                color="#8a5700" if is_flow_row else "#24272b",
                bbox=(
                    {
                        "facecolor": "#fff1c9",
                        "edgecolor": "#d69a28",
                        "boxstyle": "square,pad=0.18",
                        "linewidth": 0.8,
                    }
                    if is_flow_row
                    else None
                ),
            )
            if sub_ch:
                ax.text(
                    xmin - 0.20 * step_spacing,
                    y - 0.18,
                    sub_ch,
                    ha="right",
                    va="center",
                    fontsize=7.0,
                    color="#9a6816" if is_flow_row else "#5f6873",
                )

        colors = {
            "pulse": "#1f77b4",
            "measure": "#d62728",
            "trigger": "#d62728",
            "sync": "#5b616a",
            "wait": "#5b616a",
            "readout": "#2ca02c",
            "init": "#2468a8",
            "loop": "#3f444a",
            "update": "#8a5ccf",
            "fold": "#59616b",
        }
        target_sweep_steps = []
        previous_phase_by_channel: Dict[str, float] = {}
        occupied_text: Dict[str, List[Tuple[float, float, float, float]]] = defaultdict(list)
        for block in blocks:
            x = float(block.step) * step_spacing
            y = y_of[block.ch]
            color = colors.get(block.kind, "C0")
            block_w = 0.80 * min(step_spacing, 1.45)
            block_h = 0.42 if block.kind in {"init", "loop", "update", "fold"} else 0.38
            if waveform_blocks:
                self._draw_circuit_waveform(ax, block, x, y, block_w, block_h, color)
            else:
                self._draw_circuit_box(ax, x, y, 0.76 * min(step_spacing, 1.45), 0.42, color)
            if block.count > 1:
                self._draw_repeat_power(
                    ax,
                    x,
                    y,
                    block.count,
                    color,
                    scale=min(step_spacing, 1.45),
                    text=self._block_repeat_text(block),
                )

            lines = self._circuit_label_lines(block.label, max_label_chars)
            if len(lines) > 2:
                lines = lines[:2]
                lines[-1] = lines[-1][: max(5, max_label_chars - 3)] + "..."
            label_y_offset = 0.56 if block.kind in {"init", "loop", "update", "fold"} else 0.50
            label_text = "\n".join(lines)
            label_y = self._pick_text_position(
                occupied_text,
                block.ch,
                x,
                [y + label_y_offset, y + label_y_offset + 0.22, y + label_y_offset + 0.44],
                label_text,
                fontsize=8.1,
                step_spacing=step_spacing,
                va="bottom",
            )
            ax.text(
                x,
                label_y,
                label_text,
                ha="center",
                va="bottom",
                fontsize=8.1,
                color="#202327",
                fontweight="semibold" if block.kind in {"init", "loop", "update", "fold"} else "normal",
                linespacing=0.92,
                path_effects=text_effects,
                zorder=4,
            )

            phase_text = self._block_phase_text(
                block,
                previous_phase_by_channel,
                phase_mode=phase_mode,
                show_zero_phase=show_zero_phase,
            )
            detail_text = self._block_detail_text(block, phase_text, detail=detail)
            if annotate_params_enabled and detail_text and not (loop_summary_panel and block.kind in {"init", "loop", "update", "fold"}):
                mode = str(detail).lower()
                is_loop_detail = block.kind in {"init", "loop", "update", "fold"}
                wrap_width = max(18, detail_wrap_chars)
                if is_loop_detail:
                    wrap_width = max(wrap_width, 36)
                elif mode in {"full", "all", "params"}:
                    wrap_width = min(max(wrap_width, 20), 24)
                max_lines = max(max_detail_lines, 4 if is_loop_detail else max_detail_lines)
                detail_text = self._fit_text_lines(detail_text, width=wrap_width, max_lines=max_lines)
                if is_loop_detail:
                    detail_candidates = [y - 0.52, y - 0.74]
                    detail_fontsize = 7.0
                else:
                    detail_candidates = [y - 0.50, y - 0.72, y + 0.76]
                    detail_fontsize = 7.0
                detail_y = self._pick_text_position(
                    occupied_text,
                    block.ch,
                    x,
                    detail_candidates,
                    detail_text,
                    fontsize=detail_fontsize,
                    step_spacing=step_spacing,
                    va="top",
                )
                ax.text(
                    x,
                    detail_y,
                    detail_text,
                    ha="center",
                    va="top",
                    fontsize=detail_fontsize,
                    linespacing=0.92,
                    color="#34383d",
                    path_effects=text_effects,
                    zorder=5,
                )

            if block.sweep_text:
                target_sweep_steps.append(block.step)

        for idx, block in enumerate(blocks):
            if block.pattern_count > 1:
                self._draw_pattern_fold_annotation(
                    ax,
                    blocks,
                    idx,
                    y_of=y_of,
                    step_spacing=step_spacing,
                    plot_ymax=plot_ymax,
                    text_effects=text_effects,
                )

        if annotate_sweep:
            sweep_text = self._global_sweep_text()
            if sweep_text:
                if target_sweep_steps:
                    x0 = min(target_sweep_steps) * step_spacing - 0.42 * step_spacing
                    x1 = max(target_sweep_steps) * step_spacing + 0.42 * step_spacing
                else:
                    x0, x1 = xmin + 0.2, xmax - 0.2
                y = (len(chans) - 1) * row_spacing + 0.56
                ax.plot([x0, x0, x1, x1], [y - 0.08, y, y, y - 0.08], color="#8a5ccf", linewidth=1.3)
                ax.text((x0 + x1) / 2, y + 0.09, sweep_text, ha="center", va="bottom", fontsize=9, color="#8a5ccf")

        if annotate_conditionals:
            conditionals = [marker for marker in self.markers if marker.kind == "condj"]
            for idx, marker in enumerate(conditionals[:8]):
                x = self._marker_step(marker, blocks) * step_spacing
                y = plot_ymax - (0.34 + 0.22 * (idx % 2))
                s = 0.10 * step_spacing
                color = "#b36b00"
                ax.plot([x, x + s, x, x - s, x], [y + s, y, y - s, y, y + s], color=color, linewidth=1.1, zorder=9)
                ax.text(
                    x,
                    y,
                    "if",
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    color=color,
                    fontweight="semibold",
                    path_effects=text_effects,
                    zorder=10,
                )
                ax.text(
                    x + 0.16 * step_spacing,
                    y + 0.08,
                    self._fit_text_lines(self._conditional_marker_text(marker), width=24, max_lines=1),
                    ha="left",
                    va="bottom",
                    fontsize=7.0,
                    color=color,
                    path_effects=text_effects,
                    zorder=10,
                )

        ax.set_xlim(xmin - 0.45 * step_spacing, xmax + 0.35 * step_spacing)
        ax.set_ylim(plot_ymin, plot_ymax)
        ax.set_yticks([])
        ax.set_xticks([i * step_spacing for i in range(max_step + 1)])
        ax.set_xticklabels([str(i) for i in range(max_step + 1)])
        ax.tick_params(axis="x", length=0, labelsize=9, colors="#34383d")
        ax.set_xlabel("operation order", fontsize=10.5, color="#24272b", labelpad=8)
        final_title = title or self._default_circuit_title()
        ax.set_title(final_title, fontsize=13.5, fontweight="semibold", color="#202327", pad=18, linespacing=1.28)
        for spine in ax.spines.values():
            spine.set_visible(False)
        top_margin = 0.82 if "\n" in final_title else 0.88
        left_margin = 0.115 if channel_aliases else 0.085
        right_margin = 0.99
        bottom_margin = 0.34 if summary_cards else 0.16
        fig.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin)
        if summary_cards:
            self._draw_loop_summary_cards(
                fig,
                summary_cards,
                left=left_margin,
                right=right_margin,
                bottom=0.035,
                height=0.22,
            )
        return fig, ax

    def plot(self, *, show_sync: bool = False, annotate: bool = True, min_label_width: float = 0.03,
             figsize: Optional[Tuple[float, float]] = None, title: Optional[str] = None):
        """Plot a Gantt-style pulse sequence. Returns (fig, ax)."""
        import matplotlib.pyplot as plt

        chans = self.channels(show_sync=show_sync)
        if not chans:
            fig, ax = plt.subplots(figsize=figsize or (10, 2))
            ax.set_title(title or "No pulse events recorded")
            return fig, ax

        y_of = {ch: i for i, ch in enumerate(chans)}
        if figsize is None:
            figsize = (14, max(2.5, 0.55 * len(chans) + 1.6))
        fig, ax = plt.subplots(figsize=figsize)
        bar_linewidth = 14
        max_t = 0.0
        for e in sorted(self.events, key=lambda x: (x.t0, x.ch, x.kind)):
            if e.ch not in y_of:
                continue
            y = y_of[e.ch]
            width = max(e.duration, 1e-12)
            ax.plot([e.t0, e.t0 + width], [y, y], linewidth=bar_linewidth, solid_capstyle="butt")
            max_t = max(max_t, e.t1)
            if annotate and e.duration >= min_label_width:
                txt = e.label
                if len(txt) > 34:
                    txt = txt[:31] + "..."
                ax.text(e.t0 + e.duration / 2, y, txt, ha="center", va="center", fontsize=8)

        for m in self.markers:
            ax.plot([m.t, m.t], [-0.8, len(chans) - 0.2], linewidth=0.8, linestyle="--", alpha=0.4)
            if annotate:
                ax.text(m.t, len(chans) - 0.4, m.label, rotation=90, va="top", ha="right", fontsize=7, alpha=0.7)
            max_t = max(max_t, m.t)

        ax.set_yticks(range(len(chans)))
        ax.set_yticklabels(chans)
        ax.set_xlabel(f"time [{self.unit}]")
        ax.set_ylabel("channel")
        ax.set_ylim(-0.8, len(chans) - 0.2)
        ax.set_xlim(0, max_t * 1.03 if max_t > 0 else 1)
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_title(title or "Traced pulse sequence")
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.9)
        return fig, ax


class PulseTraceMixin:
    """
    Mixin inserted before a QICK Program class to record pulse events.

    It intentionally overrides make_program(), pulse(), setup_and_pulse(),
    measure(), sync_all(), etc. so Program construction becomes a dry run.
    """

    _trace_options_default: Dict[str, Any] = {}

    def __init__(self, *args, **kwargs):
        trace_options = dict(getattr(self, "_trace_options_default", {}) or {})
        trace_options.update(kwargs.pop("trace_options", {}) or {})
        self._trace_init(trace_options)
        super().__init__(*args, **kwargs)

    def _trace_init(self, options: Optional[Mapping[str, Any]] = None) -> None:
        options = dict(options or {})
        self.trace = PulseTrace(
            unit=options.get("unit", "us"),
            show_zero_gain=bool(options.get("show_zero_gain", False)),
        )
        self._trace_options = options
        self._trace_current_regs: Dict[str, Dict[str, Any]] = {}
        self._trace_default_regs: Dict[str, Dict[str, Any]] = {}
        self._trace_readout_regs: Dict[str, Dict[str, Any]] = {}
        self._trace_waveforms: Dict[Tuple[str, str], float] = {}
        self._trace_waveforms_by_name: Dict[str, float] = {}
        self._trace_waveform_kinds: Dict[Tuple[str, str], str] = {}
        self._trace_waveform_kinds_by_name: Dict[str, str] = {}
        self._trace_waveform_samples: Dict[Tuple[str, str], List[float]] = {}
        self._trace_waveform_samples_by_name: Dict[str, List[float]] = {}
        self._trace_register_values: Dict[Tuple[str, Any], Any] = {}
        self._trace_section_stack: List[str] = []
        self._trace_errors: List[str] = []
        self._trace_custom_pulse_label_stack: List[List[str]] = []
        self._trace_pulse_label_registry: Dict[Tuple[Any, ...], List[str]] = {}

    def _trace_hashable_value(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return tuple(sorted((str(k), self._trace_hashable_value(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(self._trace_hashable_value(v) for v in value)
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        try:
            hash(value)
            return value
        except Exception:
            return _short_value(value, maxlen=80)

    def _trace_pulse_rows(self, pulse_data: Any) -> List[List[Any]]:
        if pulse_data is None:
            return []
        try:
            if hasattr(pulse_data, "tolist"):
                pulse_data = pulse_data.tolist()
        except Exception:
            pass
        if isinstance(pulse_data, tuple):
            pulse_data = list(pulse_data)
        if not isinstance(pulse_data, list):
            return []
        rows: List[List[Any]] = []
        for row in pulse_data:
            if hasattr(row, "tolist"):
                try:
                    row = row.tolist()
                except Exception:
                    pass
            if isinstance(row, tuple):
                row = list(row)
            if isinstance(row, list):
                rows.append(list(row))
            else:
                rows.append([row])
        return rows

    def _trace_pulse_columns(self, pulse_data: Any) -> List[List[Any]]:
        rows = self._trace_pulse_rows(pulse_data)
        if not rows:
            return []
        width = max((len(row) for row in rows), default=0)
        cols: List[List[Any]] = []
        for idx in range(width):
            cols.append([row[idx] if idx < len(row) else None for row in rows])
        return cols

    def _trace_pulse_data_signature(self, pulse_data: Any) -> Tuple[Any, ...]:
        rows = self._trace_pulse_rows(pulse_data)
        return tuple(tuple(self._trace_hashable_value(v) for v in row) for row in rows)

    def _trace_semantic_labels_from_pulse_spec(self, spec: Any) -> List[str]:
        try:
            parts = list(spec)
        except Exception:
            return [_short_value(spec, maxlen=32)]
        if not parts:
            return ["pulse"]
        channel = str(parts[0])
        target = str(parts[1]) if len(parts) > 1 else ""
        pulse_name = str(parts[2]) if len(parts) > 2 else ""

        if channel == "qubit":
            return [f"{pulse_name}_{target}".strip("_")]
        if channel in {"storage", "man", "multiphoton"}:
            return [f"{pulse_name} {target}".strip()]
        if channel == "floquet":
            if pulse_name in {"pi", "hpi"}:
                return [f"{pulse_name} {target}".strip()]
            return [f"floquet {target}".strip()]
        if channel in {"wait", "buffer"}:
            wait_len = target or pulse_name
            return [f"{channel} {wait_len}".strip()]
        if channel == "optimal_control":
            encoding = target
            state = pulse_name
            base = f"OC {encoding}/{state}".rstrip("/")
            return [f"{base} q", f"{base} m"]
        if pulse_name:
            return [f"{pulse_name} {target}".strip()]
        return [" ".join(str(p) for p in parts[1:3]).strip() or channel]

    def _trace_register_pulse_label_variant(self, pulse_rows: Any, labels: Sequence[str]) -> None:
        sig = self._trace_pulse_data_signature(pulse_rows)
        if sig:
            self._trace_pulse_label_registry[sig] = [str(label) for label in labels]

    def _trace_register_pulse_labels(self, pulse_data: Any, labels: Sequence[str]) -> None:
        rows = self._trace_pulse_rows(pulse_data)
        if not rows:
            return
        labels_list = [str(label) for label in labels]
        self._trace_register_pulse_label_variant(rows, labels_list)
        reversed_rows = [list(row)[::-1] for row in rows]
        self._trace_register_pulse_label_variant(reversed_rows, labels_list[::-1])
        sliced_reversed_rows = [list(row)[:0:-1] for row in rows]
        self._trace_register_pulse_label_variant(sliced_reversed_rows, labels_list[:0:-1])

    def _trace_labels_from_pulse_data(self, pulse_data: Any, prefix: str = "pulse") -> List[str]:
        cols = self._trace_pulse_columns(pulse_data)
        clean_prefix = str(prefix or "pulse").strip("_") or "pulse"
        labels: List[str] = []
        for idx, col in enumerate(cols):
            shape = col[5] if len(col) > 5 else None
            if isinstance(shape, (list, tuple)) and shape and shape[0] == "opt_cont":
                encoding = str(shape[1]) if len(shape) > 1 else ""
                state = str(shape[2]) if len(shape) > 2 else ""
                labels.append(f"OC {encoding}/{state}".rstrip("/"))
            elif len(cols) == 1:
                labels.append(clean_prefix)
            else:
                labels.append(f"{clean_prefix} {idx + 1}")
        return labels

    def _trace_filter_custom_labels(
        self,
        labels: Sequence[str],
        pulse_data: Any,
        sync_zero_const: bool,
    ) -> List[str]:
        cols = self._trace_pulse_columns(pulse_data)
        if not cols:
            return [str(label) for label in labels]
        out: List[str] = []
        for idx, col in enumerate(cols):
            label = str(labels[idx]) if idx < len(labels) else f"pulse {idx + 1}"
            gain = col[1] if len(col) > 1 else None
            shape = col[5] if len(col) > 5 else None
            shape_name = str(shape).lower() if not isinstance(shape, (list, tuple)) else str(shape[0]).lower()
            gain_zero = (_try_float(gain, None) == 0)
            creates_only_sync = (
                sync_zero_const
                and gain_zero
                and shape_name not in {"gaussian", "gauss", "g", "flat_top", "f", "opt_cont"}
            )
            if not creates_only_sync:
                out.append(label)
        return out

    def _trace_lookup_custom_labels(self, pulse_data: Any, prefix: str, sync_zero_const: bool) -> List[str]:
        labels = self._trace_pulse_label_registry.get(self._trace_pulse_data_signature(pulse_data))
        if labels is None:
            labels = self._trace_labels_from_pulse_data(pulse_data, prefix=prefix)
        return self._trace_filter_custom_labels(labels, pulse_data, sync_zero_const)

    def _trace_pop_custom_label(self) -> Optional[str]:
        while self._trace_custom_pulse_label_stack:
            labels = self._trace_custom_pulse_label_stack[-1]
            if labels:
                return labels.pop(0)
            self._trace_custom_pulse_label_stack.pop()
        return None

    def _trace_apply_next_custom_label(self, regs: MutableMapping[str, Any]) -> None:
        if regs.get("display_label") or regs.get("logical_label"):
            return
        label = self._trace_pop_custom_label()
        if label:
            regs["display_label"] = label
            regs["logical_label"] = label

    def _trace_context_scalar(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "item"):
            try:
                item = value.item()
                if isinstance(item, (str, int, float, bool)):
                    return item
            except Exception:
                pass
        return None

    def _trace_context_value(self, value: Any) -> Any:
        scalar = self._trace_context_scalar(value)
        if scalar is not None or value is None:
            return scalar
        try:
            if hasattr(value, "tolist"):
                value = value.tolist()
        except Exception:
            pass
        if isinstance(value, (list, tuple)):
            if len(value) > int(self._trace_options.get("loop_context_max_sequence", 16)):
                return None
            out = []
            for item in value:
                item_scalar = self._trace_context_scalar(item)
                if item_scalar is None and item is not None:
                    return None
                out.append(item_scalar)
            return tuple(out)
        return None

    def _trace_context_key_interesting(self, name: str, value: Any) -> bool:
        low = str(name).lower()
        if low in {"self", "cfg", "pulse_args", "all_pulse_args"}:
            return False
        if low.startswith("__"):
            return False
        if low in {"i", "j", "k", "kk"} or low.startswith(("i_", "j_", "k_")):
            return True
        tokens = (
            "stor", "mode", "cycle", "phase", "freq", "detun", "gain", "length",
            "idx", "index", "update", "pulse", "state", "target", "qubit",
        )
        if any(token in low for token in tokens):
            return True
        return isinstance(value, (int, float, bool)) and len(low) <= 3

    def _trace_capture_loop_context(self, frame: Any) -> Dict[str, Any]:
        if frame is None or not self._trace_options.get("capture_loop_context", True):
            return {}
        out: Dict[str, Any] = {}
        for name, value in frame.f_locals.items():
            if not self._trace_context_key_interesting(name, value):
                continue
            clean = self._trace_context_value(value)
            if clean is not None or value is None:
                out[str(name)] = clean
            if len(out) >= int(self._trace_options.get("loop_context_max_items", 14)):
                break
        self._trace_add_indexed_context_values(out)
        return out

    def _trace_add_indexed_context_values(self, ctx: MutableMapping[str, Any]) -> None:
        for idx_key, raw_idx in list(ctx.items()):
            low = str(idx_key).lower()
            if not low.startswith(("i_", "j_", "k_")):
                continue
            idx_f = _try_float(raw_idx, None)
            if idx_f is None or not stdlib_math.isfinite(idx_f):
                continue
            idx = int(round(idx_f))
            base = low.split("_", 1)[1]
            if not base or base in ctx:
                continue
            for seq_key, seq in list(ctx.items()):
                seq_low = str(seq_key).lower()
                if seq_key == idx_key or base not in seq_low:
                    continue
                if not (seq_low.endswith("s") or seq_low.endswith("_list") or "list" in seq_low):
                    continue
                values = list(seq) if isinstance(seq, (list, tuple)) else []
                if 0 <= idx < len(values):
                    value = self._trace_context_scalar(values[idx])
                    if value is not None:
                        ctx[base] = value
                        break

    def _trace_summary_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if hasattr(value, "tolist"):
            try:
                value = value.tolist()
            except Exception:
                pass
        if isinstance(value, (list, tuple)):
            if len(value) <= 6:
                return tuple(self._trace_summary_value(v) for v in value)
            return f"<{type(value).__name__} len={len(value)} first={_short_value(value[0], 24)} last={_short_value(value[-1], 24)}>"
        if isinstance(value, Mapping):
            return f"<mapping keys={len(value)}>"
        return _short_value(value, maxlen=60)

    def _trace_flatten_state(self, obj: Any, prefix: str, *, depth: int = 0, max_depth: int = 2) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if obj is None:
            return out
        if isinstance(obj, Mapping):
            items = list(obj.items())
        else:
            raw = getattr(obj, "__dict__", None)
            items = list(raw.items()) if isinstance(raw, Mapping) else []
        for key, value in items:
            key_s = str(key)
            path = f"{prefix}.{key_s}" if prefix else key_s
            if isinstance(value, Mapping) and depth < max_depth:
                out.update(self._trace_flatten_state(value, path, depth=depth + 1, max_depth=max_depth))
            else:
                out[path] = self._trace_summary_value(value)
        return out

    def _trace_program_state_snapshot(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        state.update(self._trace_flatten_state(self._trace_current_regs, "pulse_regs"))
        state.update(self._trace_flatten_state(self._trace_default_regs, "default_regs"))
        state.update(self._trace_flatten_state({str(k): v for k, v in self._trace_register_values.items()}, "registers"))
        expt = _get_attr_or_key(getattr(self, "cfg", None), "expt", None)
        state.update(self._trace_flatten_state(expt, "cfg.expt", max_depth=1))
        return state

    def _trace_changed_state_text(self, before: Mapping[str, Any], after: Mapping[str, Any], *, limit: int = 6) -> List[str]:
        changes = []
        keys = sorted(set(before) | set(after))
        for key in keys:
            old = before.get(key, "<unset>")
            new = after.get(key, "<unset>")
            if old != new:
                changes.append(f"{key}: {_short_value(old, 34)} -> {_short_value(new, 34)}")
            if len(changes) >= limit:
                break
        extra = max(0, len([k for k in keys if before.get(k, '<unset>') != after.get(k, '<unset>')]) - len(changes))
        if extra:
            changes.append(f"... {extra} more")
        return changes

    def _trace_marker_text(self, marker: TraceMarker) -> str:
        if marker.kind == "condj":
            return self.trace._conditional_marker_text(marker)
        if marker.kind == "reg":
            reg = marker.params.get("reg") if marker.params else None
            imm = marker.params.get("imm") if marker.params else None
            return f"{self._trace_register_display_name(reg)} = {_short_value(imm, 24)}"
        if marker.kind == "mathi":
            decoded = marker.params.get("decoded") if marker.params else None
            if decoded:
                return str(decoded)
        args = marker.params.get("args", ()) if marker.params else ()
        kwargs = marker.params.get("kwargs", {}) if marker.params else {}
        pieces = []
        for arg in list(args)[:5]:
            pieces.append(_short_value(arg, 24))
        if len(args) > 5:
            pieces.append("...")
        for key, value in list(kwargs.items())[:3]:
            pieces.append(f"{key}={_short_value(value, 24)}")
        suffix = f"({', '.join(pieces)})" if pieces else ""
        return f"{marker.label}{suffix}"

    def _trace_add_loop_event(
        self,
        label: str,
        summary: str,
        *,
        kind: str = "loop",
        params: Optional[Dict[str, Any]] = None,
        t_offset: Optional[Any] = None,
    ) -> None:
        start_offset = self.trace.now() - self.trace.t_ref if t_offset is None else (_try_float(t_offset, 0.0) or 0.0)
        payload = dict(params or {})
        payload["summary"] = summary
        self.trace.add_event(
            ch="LOOP",
            kind=kind,
            label=label,
            duration=self._trace_options.get("loop_summary_duration", 0.02),
            t0=start_offset,
            params=payload,
            source=self.trace.current_section,
        )

    def _trace_add_initialize_summary(self, marker_start: int = 0) -> None:
        if not self._trace_options.get("show_initialize_summary", True):
            return
        markers = self.trace.markers[marker_start:]
        counts: Dict[str, int] = {}
        waveform_count = len(getattr(self, "_trace_waveforms", {}))
        if waveform_count:
            counts["waveforms"] = waveform_count
        declare_count = sum(1 for marker in markers if marker.kind in {"declare_gen", "declare_readout"})
        if declare_count:
            counts["declarations"] = declare_count
        reg_count = len(getattr(self, "_trace_default_regs", {})) + len(getattr(self, "_trace_current_regs", {}))
        if reg_count:
            counts["pulse registers"] = reg_count
        if not counts:
            counts["setup"] = 1
        summary = "; ".join(f"{name}={value}" for name, value in counts.items())
        self._trace_add_loop_event(
            "initialize",
            summary,
            kind="init",
            params={"counts": counts},
            t_offset=-(self._trace_options.get("loop_summary_duration", 0.02) or 0.02),
        )

    def _trace_add_acquire_loop_summary(self) -> None:
        if not self._trace_options.get("show_loop_summary", True):
            return
        expt = _get_attr_or_key(getattr(self, "cfg", None), "expt", None)
        pieces = []
        loop_params: Dict[str, Any] = {}
        for name in ("reps", "rounds", "expts"):
            value = _get_attr_or_key(expt, name, None)
            val_f = _try_float(value, None)
            if val_f is not None and val_f > 1:
                pieces.append(f"{name}={self.trace._format_circuit_value(value)}")
                loop_params[name] = value
        if not pieces:
            return
        if hasattr(self, "update"):
            pieces.append("update summarized")
        summary = "body not expanded: " + ", ".join(pieces)
        self._trace_add_loop_event("hardware loop", summary, kind="loop", params=loop_params)

    def _trace_record_update_summary(self) -> None:
        if not self._trace_options.get("summarize_update", True):
            return
        if not hasattr(self, "update"):
            return
        before = self._trace_program_state_snapshot()
        marker_start = len(self.trace.markers)
        self._trace_enter("update")
        error_text = ""
        try:
            self.update()
        except Exception as exc:
            error_text = f"update dry-run failed: {type(exc).__name__}: {exc}"
            self._trace_errors.append(error_text)
            if self._trace_options.get("raise_update_errors", False):
                raise
        finally:
            self._trace_exit()
        after = self._trace_program_state_snapshot()
        changes = self._trace_changed_state_text(before, after)
        marker_texts = [
            self._trace_marker_text(marker)
            for marker in self.trace.markers[marker_start:]
            if marker.kind in {"math", "mathi", "bitwi", "reg", "memwi", "loopnz", "condj"}
        ]
        summary_pieces = []
        if changes:
            summary_pieces.extend(changes[:3])
        if marker_texts:
            summary_pieces.extend(marker_texts[:3])
        if error_text:
            summary_pieces.append(error_text)
        if not summary_pieces:
            summary_pieces.append("update executed; no traced state change")
        summary = "; ".join(summary_pieces)
        self._trace_add_loop_event(
            "update",
            summary,
            kind="update",
            params={"changes": changes, "markers": marker_texts, "error": error_text},
        )

    # ---- Program template ----------------------------------------------------
    def make_program(self):
        """Replacement for QICK AveragerProgram.make_program()."""
        if not hasattr(self, "trace"):
            self._trace_init({})
        run_initialize = self._trace_options.get("run_initialize", True)
        run_body = self._trace_options.get("run_body", True)
        if run_initialize and hasattr(self, "initialize"):
            marker_start = len(self.trace.markers)
            self._trace_enter("initialize")
            try:
                self.initialize()
            finally:
                self._trace_add_initialize_summary(marker_start)
                self._trace_exit()
        if run_body and hasattr(self, "body"):
            self.trace.add_marker("loop", "body/start")
            self._trace_enter("body")
            try:
                self.body()
            finally:
                self._trace_exit()
            self.trace.add_marker("loop", "body/end")
            self._trace_add_acquire_loop_summary()
        self._trace_record_update_summary()
        return None

    def _trace_enter(self, section: str) -> None:
        self._trace_section_stack.append(section)
        self.trace.current_section = "/".join(self._trace_section_stack)

    def _trace_exit(self) -> None:
        if self._trace_section_stack:
            self._trace_section_stack.pop()
        self.trace.current_section = "/".join(self._trace_section_stack)

    # ---- Unit converters: dry-run defaults ----------------------------------
    def us2cycles(self, us, *args, **kwargs):
        scale = _try_float(self._trace_options.get("cycles_per_us", 1.0), 1.0) or 1.0
        val = _try_float(us, 0.0) or 0.0
        if self._trace_options.get("unit", "us") == "cycles":
            return int(round(val * scale))
        # For visualization, keep the x-axis in microseconds by default.
        return val

    def cycles2us(self, cycles, *args, **kwargs):
        scale = _try_float(self._trace_options.get("cycles_per_us", 1.0), 1.0) or 1.0
        val = _try_float(cycles, 0.0) or 0.0
        if self._trace_options.get("unit", "us") == "cycles":
            return val / scale
        return val

    def freq2reg(self, freq, *args, **kwargs):
        return freq

    def reg2freq(self, reg, *args, **kwargs):
        return reg

    def deg2reg(self, deg, *args, **kwargs):
        return deg

    def reg2deg(self, reg, *args, **kwargs):
        return reg

    def ch_page(self, ch, *args, **kwargs):
        return f"page:{ch}"

    def ch_page_ro(self, ch, *args, **kwargs):
        return f"ro_page:{ch}"

    def sreg(self, ch, name, *args, **kwargs):
        return f"{ch}:{name}"

    def sreg_ro(self, ch, name, *args, **kwargs):
        return f"ro:{ch}:{name}"

    def _trace_base_method(self, name: str) -> Callable[..., Any]:
        wrapped = getattr(self, "_trace_wrapped_methods", {})
        if name in wrapped:
            return wrapped[name]
        return getattr(super(PulseTraceMixin, self), name)

    def get_prepulse_creator(self, sweep_pulse: Optional[List[List[Union[str, int]]]] = None, cfg: Any = None):
        method = self._trace_base_method("get_prepulse_creator")
        try:
            creator = method(sweep_pulse=sweep_pulse, cfg=cfg)
        except TypeError:
            creator = method(sweep_pulse, cfg)

        labels: List[str] = []
        for pulse_spec in sweep_pulse or []:
            labels.extend(self._trace_semantic_labels_from_pulse_spec(pulse_spec))
        pulse_data = getattr(creator, "pulse", None)
        if labels and pulse_data is not None:
            self._trace_register_pulse_labels(pulse_data, labels)
            try:
                setattr(creator, "_trace_labels", labels)
            except Exception:
                pass
        return creator

    def custom_pulse(
        self,
        cfg,
        pulse_data: Optional[Union[List[List[float]], Any]] = None,
        advance_qubit_phase: float = 0,
        sync_zero_const: bool = True,
        waveform_preload: Optional[List[str]] = None,
        prefix: str = "pre",
    ):
        method = self._trace_base_method("custom_pulse")
        labels = self._trace_lookup_custom_labels(pulse_data, prefix=prefix, sync_zero_const=sync_zero_const)
        self._trace_custom_pulse_label_stack.append(labels)
        try:
            try:
                return method(
                    cfg,
                    pulse_data=pulse_data,
                    advance_qubit_phase=advance_qubit_phase,
                    sync_zero_const=sync_zero_const,
                    waveform_preload=waveform_preload,
                    prefix=prefix,
                )
            except TypeError:
                return method(cfg, pulse_data, advance_qubit_phase, sync_zero_const, waveform_preload, prefix)
        finally:
            if self._trace_custom_pulse_label_stack and self._trace_custom_pulse_label_stack[-1] is labels:
                self._trace_custom_pulse_label_stack.pop()

    def custom_pulse_with_preloaded_wfm(
        self,
        cfg,
        pulse_data,
        advance_qubit_phase=None,
        sync_zero_const=False,
        prefix="pre",
        same_storage=False,
        same_qubit_pulse=False,
        storage_no=1,
    ):
        method = self._trace_base_method("custom_pulse_with_preloaded_wfm")
        labels = self._trace_lookup_custom_labels(pulse_data, prefix=prefix, sync_zero_const=sync_zero_const)
        self._trace_custom_pulse_label_stack.append(labels)
        try:
            return method(
                cfg,
                pulse_data,
                advance_qubit_phase=advance_qubit_phase,
                sync_zero_const=sync_zero_const,
                prefix=prefix,
                same_storage=same_storage,
                same_qubit_pulse=same_qubit_pulse,
                storage_no=storage_no,
            )
        finally:
            if self._trace_custom_pulse_label_stack and self._trace_custom_pulse_label_stack[-1] is labels:
                self._trace_custom_pulse_label_stack.pop()

    # ---- Declarations and waveforms -----------------------------------------
    def declare_gen(self, *args, **kwargs):
        self.trace.add_marker("declare_gen", "declare_gen", params={"args": args, "kwargs": kwargs})
        return None

    def declare_readout(self, *args, **kwargs):
        self.trace.add_marker("declare_readout", "declare_readout", params={"args": args, "kwargs": kwargs})
        return None

    def default_pulse_registers(self, ch, **kwargs):
        c = str(ch)
        self._trace_default_regs.setdefault(c, {}).update(kwargs)
        return None

    def set_pulse_registers(self, ch, **kwargs):
        for c in _as_list(ch):
            cstr = str(c)
            regs = dict(self._trace_default_regs.get(cstr, {}))
            regs.update(kwargs)
            regs["ch"] = c
            self._trace_current_regs[cstr] = regs
        return None

    def set_readout_registers(self, ch, **kwargs):
        for c in _as_list(ch):
            self._trace_readout_regs[str(c)] = dict(kwargs)
        return None

    def default_readout_registers(self, ch, **kwargs):
        self._trace_readout_regs.setdefault(str(ch), {}).update(kwargs)
        return None

    def _trace_store_waveform(self, ch: Any, name: Any, duration: Any, kind: str, samples: Optional[Sequence[Any]] = None) -> None:
        ch_s = str(ch)
        name_s = str(name)
        dur = _try_float(duration, 0.0) or 0.0
        self._trace_waveforms[(ch_s, name_s)] = dur
        self._trace_waveforms_by_name[name_s] = dur
        self._trace_waveform_kinds[(ch_s, name_s)] = kind
        self._trace_waveform_kinds_by_name[name_s] = kind
        clean_samples = self._trace_clean_waveform_samples(samples)
        if clean_samples:
            self._trace_waveform_samples[(ch_s, name_s)] = clean_samples
            self._trace_waveform_samples_by_name[name_s] = clean_samples

    def _trace_clean_waveform_samples(self, samples: Optional[Sequence[Any]], *, max_points: int = 320) -> List[float]:
        if samples is None:
            return []
        try:
            if hasattr(samples, "tolist"):
                samples = samples.tolist()
        except Exception:
            pass
        if not isinstance(samples, (list, tuple)):
            return []
        vals = []
        for value in samples:
            val = _try_float(value, None)
            if val is not None and stdlib_math.isfinite(val):
                vals.append(float(val))
        if len(vals) < 2:
            return []
        if len(vals) > max_points:
            step = (len(vals) - 1) / (max_points - 1)
            vals = [vals[int(round(i * step))] for i in range(max_points)]
        return vals

    def _trace_gaussian_samples(self, sigma: Any, length: Any) -> List[float]:
        length_f = max(_try_float(length, 0.0) or 0.0, 1e-9)
        n = int(round(length_f))
        n = min(max(n, 48), 320)
        sigma_f = max(_try_float(sigma, 1.0) or 1.0, 1e-9)
        scale = n / length_f
        sigma_pts = max(sigma_f * scale, 1e-9)
        mu = (n - 1) / 2
        return [stdlib_math.exp(-0.5 * ((i - mu) / sigma_pts) ** 2) for i in range(n)]

    def _trace_iq_envelope_samples(self, idata: Any = None, qdata: Any = None) -> List[float]:
        i_vals = self._trace_clean_waveform_samples(idata)
        q_vals = self._trace_clean_waveform_samples(qdata)
        if i_vals and q_vals:
            n = min(len(i_vals), len(q_vals))
            return [stdlib_math.hypot(i_vals[i], q_vals[i]) for i in range(n)]
        if i_vals:
            return [abs(v) for v in i_vals]
        if q_vals:
            return [abs(v) for v in q_vals]
        return []

    def add_gauss(self, ch, name, sigma, length, *args, **kwargs):
        dur = _try_float(length, 0.0) or 0.0
        self._trace_store_waveform(ch, name, dur, "gaussian", self._trace_gaussian_samples(sigma, length))
        self.trace.add_marker("waveform", f"add_gauss {name}", params={"ch": ch, "sigma": sigma, "length": length})
        return None

    def add_DRAG(self, ch, name, sigma, length, *args, **kwargs):
        dur = _try_float(length, 0.0) or 0.0
        self._trace_store_waveform(ch, name, dur, "drag", self._trace_gaussian_samples(sigma, length))
        self.trace.add_marker("waveform", f"add_DRAG {name}", params={"ch": ch, "sigma": sigma, "length": length})
        return None

    def add_triangle(self, ch, name, length, *args, **kwargs):
        dur = _try_float(length, 0.0) or 0.0
        n = int(min(max(dur, 3), 320))
        samples = [1.0 - abs((2 * i / max(n - 1, 1)) - 1.0) for i in range(n)]
        self._trace_store_waveform(ch, name, dur, "triangle", samples)
        self.trace.add_marker("waveform", f"add_triangle {name}", params={"ch": ch, "length": length})
        return None

    def add_pulse(self, ch, name, idata=None, qdata=None, *args, **kwargs):
        dur = 0.0
        try:
            if idata is not None:
                dur = float(len(idata))
            elif qdata is not None:
                dur = float(len(qdata))
        except Exception:
            dur = 0.0
        self._trace_store_waveform(ch, name, dur, "arbitrary", self._trace_iq_envelope_samples(idata, qdata))
        self.trace.add_marker("waveform", f"add_pulse {name}", params={"ch": ch, "length": dur})
        return None

    def add_envelope(self, ch, name, idata=None, qdata=None, *args, **kwargs):
        return self.add_pulse(ch, name, idata=idata, qdata=qdata, *args, **kwargs)

    # ---- Pulse timing --------------------------------------------------------
    def _trace_infer_duration(self, ch: Any, regs: Mapping[str, Any]) -> float:
        style = str(regs.get("style", "")).lower()
        length = _try_float(regs.get("length"), None)
        waveform = regs.get("waveform")
        wf_len = None
        if waveform is not None:
            wf_len = self._trace_waveforms.get((str(ch), str(waveform)))
            if wf_len is None:
                wf_len = self._trace_waveforms_by_name.get(str(waveform))
        if style == "const":
            return length if length is not None else (_try_float(self._trace_options.get("default_const_length", 0.02), 0.02) or 0.02)
        if style in {"arb", "gauss", "gaussian"}:
            return wf_len if wf_len is not None else (length if length is not None else (_try_float(self._trace_options.get("default_arb_length", 0.02), 0.02) or 0.02))
        if style in {"flat_top", "flattop", "flat"}:
            # QICK flat_top duration includes the flat part plus the envelope/ramp contribution.
            return (length or 0.0) + (wf_len or 0.0)
        if length is not None:
            return length
        if wf_len is not None:
            return wf_len
        return _try_float(self._trace_options.get("default_pulse_length", 0.02), 0.02) or 0.02

    def _trace_label_from_regs(self, regs: Mapping[str, Any]) -> str:
        display = regs.get("display_label") or regs.get("logical_label")
        if display:
            bits = [str(display)]
        else:
            bits = [str(regs.get("style", "pulse"))]
        style = regs.get("style", "pulse")
        wf = regs.get("waveform")
        freq = regs.get("freq")
        gain = regs.get("gain")
        phase = regs.get("phase")
        if wf is not None and not display:
            bits.append(str(wf))
        if freq is not None:
            bits.append(f"f={freq}")
        if gain is not None:
            bits.append(f"g={gain}")
        if phase is not None:
            bits.append(f"ph={phase}")
        return " ".join(bits)

    def _trace_apply_waveform_kind(self, ch: Any, regs: MutableMapping[str, Any]) -> None:
        waveform = regs.get("waveform")
        if waveform is None:
            return
        key = (str(ch), str(waveform))
        if not regs.get("waveform_kind"):
            kind = self._trace_waveform_kinds.get(key)
            if kind is None:
                kind = self._trace_waveform_kinds_by_name.get(str(waveform))
            if kind:
                regs["waveform_kind"] = kind
        samples = self._trace_waveform_samples.get(key)
        if samples is None:
            samples = self._trace_waveform_samples_by_name.get(str(waveform))
        if samples:
            regs["envelope_samples"] = samples
        wf_len = self._trace_waveforms.get(key)
        if wf_len is None:
            wf_len = self._trace_waveforms_by_name.get(str(waveform))
        if wf_len is not None:
            regs["envelope_length"] = wf_len

    def setup_and_pulse(self, ch, t="auto", **kwargs):
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        context = self._trace_capture_loop_context(caller)
        if context and "_loop_context" not in kwargs:
            kwargs["_loop_context"] = context
        self._trace_apply_next_custom_label(kwargs)
        self.set_pulse_registers(ch, **kwargs)
        self.pulse(ch, t=t)
        return None

    def pulse(self, ch, t="auto"):
        for c in _as_list(ch):
            cstr = str(c)
            regs = dict(self._trace_current_regs.get(cstr, {}))
            if not regs:
                regs = {"ch": c, "style": "unknown"}
            self._trace_apply_next_custom_label(regs)
            frame = inspect.currentframe()
            caller = frame.f_back if frame is not None else None
            context = self._trace_capture_loop_context(caller)
            if context and "_loop_context" not in regs:
                regs["_loop_context"] = context
            self._trace_apply_waveform_kind(c, regs)
            self._trace_current_regs[cstr] = regs
            dur = self._trace_infer_duration(c, regs)
            label = self._trace_label_from_regs(regs)
            self.trace.add_event(ch=c, kind="pulse", label=label, duration=dur, t0=t, params=regs)
        return None

    def readout(self, ch, t="auto"):
        for c in _as_list(ch):
            regs = self._trace_readout_regs.get(str(c), {})
            dur = _try_float(regs.get("length"), self._trace_options.get("default_readout_length", 0.5)) or 0.5
            self.trace.add_event(ch=f"RO{c}", kind="readout", label=f"readout {c}", duration=dur, t0=t, params=regs)
        return None

    # ---- Measurement and synchronization ------------------------------------
    def trigger(self, adcs=None, pins=None, adc_trig_offset=0, t=0, width=0.01, *args, **kwargs):
        t0 = (_try_float(t, 0.0) or 0.0) + (_try_float(adc_trig_offset, 0.0) or 0.0)
        width_f = _try_float(width, 0.01) or 0.01
        for adc in _as_list(adcs):
            self.trace.add_event(ch=f"ADC{adc}", kind="trigger", label=f"ADC{adc} trig", duration=width_f, t0=t0,
                                 params={"adc_trig_offset": adc_trig_offset, "pins": pins})
        return None

    def measure(self, adcs, pulse_ch, pins=None, adc_trig_offset=0, t="auto", wait=False, syncdelay=None, **kwargs):
        # If kwargs are supplied, behave like setup_and_measure.
        if kwargs:
            self.set_pulse_registers(pulse_ch, **kwargs)
        # Approximate ADC acquisition window.
        trig_offset = _try_float(adc_trig_offset, 0.0) or 0.0
        readout_len = self._infer_cfg_readout_length(adcs)
        if t == "auto":
            adc_t0 = self.trace.now() - self.trace.t_ref + trig_offset
        else:
            adc_t0 = (_try_float(t, 0.0) or 0.0) + trig_offset
        for adc in _as_list(adcs):
            self.trace.add_event(
                ch=f"ADC{adc}",
                kind="measure",
                label=f"ADC{adc} window",
                duration=readout_len,
                t0=adc_t0,
                params={"adc_trig_offset": adc_trig_offset},
            )
        self.pulse(pulse_ch, t=t)
        if wait:
            self.wait_all(0)
        if syncdelay is not None:
            self.sync_all(syncdelay)
        return None

    def setup_and_measure(self, adcs, pulse_ch, pins=None, adc_trig_offset=0, t="auto", wait=False, syncdelay=None, **kwargs):
        self.set_pulse_registers(pulse_ch, **kwargs)
        return self.measure(adcs=adcs, pulse_ch=pulse_ch, pins=pins, adc_trig_offset=adc_trig_offset, t=t, wait=wait, syncdelay=syncdelay)

    def _infer_cfg_readout_length(self, adcs=None) -> float:
        # Prefer common config paths from the uploaded package.
        candidates = [
            "device.readout.length",
            "device.readout.readout_length",
            "device.readout.relax_delay",
        ]
        cfg = getattr(self, "cfg", None)
        for path in candidates:
            val = _nested_get(cfg, path, None)
            if isinstance(val, (list, tuple)) and val:
                return _try_float(val[0], 0.5) or 0.5
            if val is not None:
                return _try_float(val, 0.5) or 0.5
        return _try_float(self._trace_options.get("default_readout_length", 0.5), 0.5) or 0.5

    def sync_all(self, t=0, *args, **kwargs):
        self.trace.sync_all(t, label="sync_all")
        return None

    def wait_all(self, t=0, *args, **kwargs):
        self.trace.wait_all(t, label="wait_all")
        return None

    def synci(self, t=0, *args, **kwargs):
        self.trace.sync_all(t, label="synci")
        return None

    def waiti(self, *args, **kwargs):
        self.trace.add_marker("waiti", "waiti", params={"args": args, "kwargs": kwargs})
        return None

    # ---- Register/branch operations: markers only ---------------------------
    def _trace_register_display_name(self, reg: Any) -> str:
        if isinstance(reg, str):
            match = re.match(r"^(.+):(freq|phase|gain|gain2|length)$", reg)
            if match:
                ch, field_name = match.groups()
                return f"ch{ch}.{field_name}"
        candidates = []
        for name, value in getattr(self, "__dict__", {}).items():
            if name.startswith("r_") and value == reg:
                candidates.append(name)
        if candidates:
            priority = ("freq", "phase", "gain", "length", "wait")
            candidates.sort(key=lambda name: next((i for i, p in enumerate(priority) if p in name), 99))
            return candidates[0]
        return str(reg)

    def _trace_register_value(self, rp: Any, reg_or_value: Any) -> Any:
        key = (str(rp), reg_or_value)
        if key in self._trace_register_values:
            return self._trace_register_values[key]
        return reg_or_value

    def _trace_apply_register_value(self, rp: Any, reg: Any, value: Any) -> None:
        self._trace_register_values[(str(rp), reg)] = value
        if not isinstance(reg, str):
            return
        match = re.match(r"^(.+):(freq|phase|gain|gain2|length)$", reg)
        if not match:
            return
        ch, field_name = match.groups()
        regs = dict(self._trace_current_regs.get(str(ch), {}))
        regs[field_name] = value
        regs["ch"] = ch
        self._trace_current_regs[str(ch)] = regs

    def _trace_eval_mathi(self, left: Any, op: Any, right: Any) -> Any:
        left_f = _try_float(left, None)
        right_f = _try_float(right, None)
        op_s = str(op)
        if left_f is not None and right_f is not None:
            if op_s == "+":
                return left_f + right_f
            if op_s == "-":
                return left_f - right_f
            if op_s == "*":
                return left_f * right_f
            if op_s == "<<":
                return int(left_f) << int(right_f)
            if op_s == "|":
                return int(left_f) | int(right_f)
        if op_s == "+" and right_f == 0:
            return left
        return left

    def safe_regwi(self, rp, reg, imm, *args, **kwargs):
        self._trace_apply_register_value(rp, reg, imm)
        self.trace.add_marker("reg", f"regwi {reg}={imm}", params={"page": rp, "reg": reg, "imm": imm})
        return None

    def regwi(self, rp, reg, imm, *args, **kwargs):
        return self.safe_regwi(rp, reg, imm, *args, **kwargs)

    def bitwi(self, *args, **kwargs):
        self.trace.add_marker("bitwi", "bitwi", params={"args": args, "kwargs": kwargs})
        return None

    def mathi(self, *args, **kwargs):
        decoded = ""
        if len(args) >= 5:
            rp, dst, src, op, operand = args[:5]
            src_value = self._trace_register_value(rp, src)
            operand_value = self._trace_register_value(rp, operand)
            result = self._trace_eval_mathi(src_value, op, operand_value)
            self._trace_apply_register_value(rp, dst, result)
            decoded = (
                f"{self._trace_register_display_name(dst)} = "
                f"{self._trace_register_display_name(src)} {op} {_short_value(operand_value, 24)}"
            )
        self.trace.add_marker("mathi", "mathi", params={"args": args, "kwargs": kwargs, "decoded": decoded})
        return None

    def math(self, *args, **kwargs):
        decoded = ""
        if len(args) >= 5:
            rp, dst, left, op, right = args[:5]
            left_value = self._trace_register_value(rp, left)
            right_value = self._trace_register_value(rp, right)
            result = self._trace_eval_mathi(left_value, op, right_value)
            self._trace_apply_register_value(rp, dst, result)
            decoded = (
                f"{self._trace_register_display_name(dst)} = "
                f"{self._trace_register_display_name(left)} {op} {self._trace_register_display_name(right)}"
            )
        self.trace.add_marker("math", "math", params={"args": args, "kwargs": kwargs, "decoded": decoded})
        return None

    def memwi(self, *args, **kwargs):
        self.trace.add_marker("memwi", "memwi", params={"args": args, "kwargs": kwargs})
        return None

    def loopnz(self, *args, **kwargs):
        self.trace.add_marker("loopnz", "loopnz", params={"args": args, "kwargs": kwargs})
        return None

    def label(self, name, *args, **kwargs):
        self.trace.add_marker("label", str(name), params={"args": args, "kwargs": kwargs})
        return None

    def condj(self, *args, **kwargs):
        self.trace.add_marker("condj", "conditional jump", params={"args": args, "kwargs": kwargs})
        return None

    def read(self, *args, **kwargs):
        self.trace.add_marker("read", "read ADC result", params={"args": args, "kwargs": kwargs})
        return None

    def end(self, *args, **kwargs):
        self.trace.add_marker("end", "program end")
        return None

    def set(self, *args, **kwargs):
        return None

    def seti(self, *args, **kwargs):
        return None

    def get_timestamp(self, gen_ch=None, ro_ch=None):
        if gen_ch is not None:
            return self.trace.channel_next.get(str(gen_ch), self.trace.t_ref)
        if ro_ch is not None:
            return self.trace.channel_next.get(f"RO{ro_ch}", self.trace.t_ref)
        return self.trace.now()

    def get_max_timestamp(self, *args, **kwargs):
        return self.trace.now()

    def set_timestamp(self, t, gen_ch=None, ro_ch=None):
        ch = str(gen_ch) if gen_ch is not None else f"RO{ro_ch}"
        self.trace.channel_next[ch] = self.trace.t_ref + (_try_float(t, 0.0) or 0.0)
        return None

    def reset_timestamps(self, *args, **kwargs):
        for ch in list(self.trace.channel_next.keys()):
            self.trace.channel_next[ch] = self.trace.t_ref
        return None


# Add simple no-op aliases for rarely used low-level QICK methods.
def _noop(self, *args, **kwargs):
    if hasattr(self, "trace"):
        name = inspect.currentframe().f_code.co_name if inspect.currentframe() else "noop"
        self.trace.add_marker("noop", name, params={"args": args, "kwargs": kwargs})
    return None


for _name in [
    "declare_readout_ch", "declare", "load_pulses", "config_bufs", "config_all", "sync", "reset_phase",
    "send_readoutconfig", "set_readout_registers_ro", "default_readout_registers_ro",
]:
    if not hasattr(PulseTraceMixin, _name):
        setattr(PulseTraceMixin, _name, _noop)


def make_traced_program_class(program_class: Type[Any], *, name: Optional[str] = None,
                              trace_options: Optional[Mapping[str, Any]] = None) -> Type[Any]:
    """Return a new class whose MRO is PulseTraceMixin -> program_class."""
    cls_name = name or f"Traced{program_class.__name__}"
    attrs = {
        "__module__": program_class.__module__,
        "_trace_options_default": dict(trace_options or {}),
        "__doc__": f"Dry-run traced wrapper for {program_class.__module__}.{program_class.__name__}",
    }
    return type(cls_name, (PulseTraceMixin, program_class), attrs)


def trace_program_class(program_class: Union[Type[Any], str], *args, soccfg: Any = None, cfg: Any = None,
                        trace_options: Optional[Mapping[str, Any]] = None, **kwargs) -> Tuple[Any, PulseTrace]:
    """
    Instantiate a traced Program class and return (program_instance, trace).

    You can pass either a class object or an import string.
    If args are omitted, soccfg/cfg are passed as keyword args, then as positional
    args if the constructor rejects keywords.
    """
    if isinstance(program_class, str):
        program_class = import_object(program_class)
    Traced = make_traced_program_class(program_class, trace_options=trace_options)
    if args:
        prog = Traced(*args, trace_options=dict(trace_options or {}), **kwargs)
    else:
        try:
            prog = Traced(soccfg=soccfg, cfg=cfg, trace_options=dict(trace_options or {}), **kwargs)
        except TypeError:
            prog = Traced(soccfg, cfg, trace_options=dict(trace_options or {}), **kwargs)
    if hasattr(prog, "trace"):
        prog.trace.metadata.update(
            {
                "cfg": cfg if cfg is not None else getattr(prog, "cfg", None),
                "soccfg": soccfg if soccfg is not None else getattr(prog, "soccfg", None),
                "program_class": program_class,
            }
        )
    return prog, prog.trace


def trace_existing_program(program: Any, method: str = "body", *, trace_options: Optional[Mapping[str, Any]] = None) -> PulseTrace:
    """
    Patch one existing Program instance in-place, execute method(), and return trace.

    This is useful when construction/initialize has already succeeded and you only
    want to trace body() or core_pulses(). It does not restore methods; use on a
    scratch instance.
    """
    PulseTraceMixin._trace_init(program, trace_options or {})
    program.trace.metadata.update(
        {
            "cfg": getattr(program, "cfg", None),
            "soccfg": getattr(program, "soccfg", None),
            "program_class": program.__class__,
        }
    )
    program._trace_wrapped_methods = {}
    for n in ("get_prepulse_creator", "custom_pulse", "custom_pulse_with_preloaded_wfm"):
        if hasattr(program, n):
            program._trace_wrapped_methods[n] = getattr(program, n)
    names = [n for n in dir(PulseTraceMixin) if not n.startswith("__") and callable(getattr(PulseTraceMixin, n))]
    for n in names:
        if n in {"__init__", "make_program"}:
            continue
        setattr(program, n, getattr(PulseTraceMixin, n).__get__(program, program.__class__))
    getattr(program, method)()
    return program.trace


# -----------------------------------------------------------------------------
# Station / CharacterizationRunner helpers
# -----------------------------------------------------------------------------

class _FallbackAttrDict(dict):
    """Small AttrDict fallback used only if slab.AttrDict is unavailable."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _attrdict(obj: Any) -> Any:
    """Return slab.AttrDict(obj) when available, otherwise a recursive fallback."""
    try:
        from slab import AttrDict  # type: ignore
        if isinstance(obj, AttrDict):
            return obj
        if isinstance(obj, Mapping):
            return AttrDict(obj)
        return obj
    except Exception:
        if isinstance(obj, Mapping):
            out = _FallbackAttrDict()
            for k, v in obj.items():
                out[k] = _attrdict(v)
            return out
        if isinstance(obj, list):
            return [_attrdict(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_attrdict(v) for v in obj)
        return obj


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _set_attr_or_key(obj: Any, name: str, value: Any) -> None:
    if isinstance(obj, MutableMapping):
        obj[name] = value
    else:
        setattr(obj, name, value)


def _ensure_child(obj: Any, name: str) -> Any:
    child = _get_attr_or_key(obj, name, None)
    if child is None:
        child = _attrdict({})
        _set_attr_or_key(obj, name, child)
    return child


def _set_nested(obj: Any, dotted_path: str, value: Any) -> None:
    cur = obj
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        cur = _ensure_child(cur, part)
    _set_attr_or_key(cur, parts[-1], value)


def _attach_station_datasets(cfg: Any, station: Any) -> Any:
    """
    Attach station datasets to cfg.device.storage, matching CharacterizationRunner.run_local().
    Missing fields are created instead of failing.
    """
    ds_storage = _get_attr_or_key(station, "ds_storage", None)
    ds_floquet = _get_attr_or_key(station, "ds_floquet", None)
    if ds_storage is not None:
        _set_nested(cfg, "device.storage._ds_storage", ds_storage)
    if ds_floquet is not None:
        _set_nested(cfg, "device.storage._ds_floquet", ds_floquet)
    return cfg


def _set_missing_attr_or_key(obj: Any, name: str, value: Any) -> None:
    if _get_attr_or_key(obj, name, None) is None:
        _set_attr_or_key(obj, name, value)


def _apply_trace_expt_defaults(cfg: Any) -> Any:
    """
    Fill optional experiment flags normally introduced later by Experiment.acquire().

    The tracer instantiates the Program directly and intentionally never calls
    acquire(), so Program.initialize()/body() can see missing attrs that the normal
    execution path would have patched in before acquisition.
    """
    expt = _get_attr_or_key(cfg, "expt", None)
    if expt is None:
        return cfg

    defaults = {
        "perform_wigner": False,
        "parity_check": False,
        "preloaded_pulses": False,
        "use_arb_waveform": False,
        "map_to_qubit_ge": False,
        "pre_selection_reset": False,
        "parity_fast": False,
        "phase_second_pulse": 180,
        "man_mode_no": 1,
    }
    for name, value in defaults.items():
        _set_missing_attr_or_key(expt, name, value)
    return cfg


def build_cfg_from_runner(runner: Any, **execute_kwargs: Any) -> Any:
    """
    Build the cfg that CharacterizationRunner would give to the Program, without running hardware.

    This mirrors the package's CharacterizationRunner.run_local() setup:
        cfg = deepcopy(station.hardware_cfg)
        cfg.device.storage._ds_storage = station.ds_storage
        cfg.device.storage._ds_floquet = station.ds_floquet
        cfg.expt = runner.preprocessor(station, runner.default_expt_cfg, **kwargs)
        cfg.device.readout.relax_delay = [cfg.expt.relax_delay]  # when present

    Parameters
    ----------
    runner:
        CharacterizationRunner-like object with station, default_expt_cfg, and preprocessor.
    **execute_kwargs:
        The same keyword arguments you would pass to runner.execute(...).

    Returns
    -------
    cfg:
        A cfg object ready to pass into trace_program_class(...).
    """
    station = _get_attr_or_key(runner, "station", None)
    if station is None:
        raise AttributeError("runner.station not found")

    hardware_cfg = _get_attr_or_key(station, "hardware_cfg", None)
    if hardware_cfg is None:
        raise AttributeError("runner.station.hardware_cfg not found")

    cfg = _attrdict(deepcopy(hardware_cfg))
    _attach_station_datasets(cfg, station)

    preprocessor = _get_attr_or_key(runner, "preprocessor", None)
    default_expt_cfg = _get_attr_or_key(runner, "default_expt_cfg", None)

    kwargs_for_preproc = dict(execute_kwargs)
    if preprocessor is not None and default_expt_cfg is not None:
        expt_cfg = preprocessor(station, default_expt_cfg, **kwargs_for_preproc)
    else:
        expt_cfg = deepcopy(default_expt_cfg) if default_expt_cfg is not None else {}
        try:
            expt_cfg.update(kwargs_for_preproc)
        except Exception:
            expt_cfg = dict(kwargs_for_preproc)

    _set_attr_or_key(cfg, "expt", _attrdict(expt_cfg))
    _apply_trace_expt_defaults(cfg)

    # CharacterizationRunner.run_local applies this special override.
    expt = _get_attr_or_key(cfg, "expt", None)
    relax_delay = _get_attr_or_key(expt, "relax_delay", None)
    if relax_delay is not None:
        _set_nested(cfg, "device.readout.relax_delay", [relax_delay])

    return cfg


def _sequence_value(seq: Any, idx: int, *, param: str) -> Any:
    if seq is None:
        raise KeyError(f"Cannot select sweep point for '{param}': cfg.expt.{param}s is missing")
    if isinstance(seq, (list, tuple)):
        return seq[idx]
    # Numpy arrays, pandas Series, and other sequence-like objects.
    try:
        return seq[idx]
    except Exception:
        pass
    # Scalar already: return it for index 0 only.
    if idx in (0, None):
        return seq
    raise TypeError(f"cfg.expt.{param}s is scalar-like, cannot index it with {idx}")


def select_sweep_point(
    cfg: Any,
    point: Optional[Union[int, Sequence[int], Mapping[str, int]]] = None,
    *,
    sweep_values: Optional[Mapping[str, Any]] = None,
    swept_params: Optional[Sequence[str]] = None,
    strict: bool = True,
) -> Any:
    """
    Convert sweep-list fields such as cfg.expt.detunes/lengths into scalar fields.

    Example
    -------
    If cfg.expt.swept_params == ['detune', 'length'], then

        select_sweep_point(cfg, point=(25, 0))

    sets

        cfg.expt.detune = cfg.expt.detunes[25]
        cfg.expt.length = cfg.expt.lengths[0]

    You can also bypass indices:

        select_sweep_point(cfg, sweep_values={'detune': 0.0, 'length': 0.25})

    The input cfg is modified in-place and also returned.
    """
    expt = _get_attr_or_key(cfg, "expt", None)
    if expt is None:
        raise AttributeError("cfg.expt not found")

    if swept_params is None:
        swept_params = _get_attr_or_key(expt, "swept_params", None)
    if swept_params is None:
        swept_params = list((sweep_values or {}).keys())
    if isinstance(swept_params, str):
        swept_params = [swept_params]
    else:
        swept_params = list(swept_params or [])

    sweep_values = dict(sweep_values or {})

    if isinstance(point, Mapping):
        point_by_param = dict(point)
    elif isinstance(point, Sequence) and not isinstance(point, (str, bytes)):
        point_by_param = {p: point[i] for i, p in enumerate(swept_params) if i < len(point)}
    elif point is None:
        point_by_param = {p: 0 for p in swept_params}
    else:
        # 1D shorthand. If there are multiple params, first index is `point`, rest default to 0.
        point_by_param = {p: (point if i == 0 else 0) for i, p in enumerate(swept_params)}

    for param in swept_params:
        if param in sweep_values:
            value = sweep_values[param]
        else:
            idx = int(point_by_param.get(param, 0))
            plural_name = f"{param}s"
            seq = _get_attr_or_key(expt, plural_name, None)
            if seq is None:
                if strict:
                    raise KeyError(
                        f"cfg.expt.{plural_name} not found. "
                        f"Pass sweep_values={{'{param}': value}} or set strict=False."
                    )
                continue
            value = _sequence_value(seq, idx, param=param)
        _set_attr_or_key(expt, param, value)

    # Also set any explicit sweep_values not listed in swept_params.
    for param, value in sweep_values.items():
        if param not in swept_params:
            _set_attr_or_key(expt, param, value)

    return cfg


def _program_class_from_runner(runner: Any, program_class: Optional[Union[Type[Any], str]] = None) -> Union[Type[Any], str]:
    if program_class is not None:
        return program_class

    candidates = [
        _get_attr_or_key(runner, "program", None),
        _get_attr_or_key(runner, "ExptProgram", None),
        _get_attr_or_key(runner, "ProgramClass", None),
    ]
    expt_cls = _get_attr_or_key(runner, "ExptClass", None)
    if expt_cls is not None:
        candidates.extend([
            getattr(expt_cls, "ProgramClass", None),
            getattr(expt_cls, "program", None),
        ])

    for cand in candidates:
        if cand is not None:
            return cand

    raise AttributeError(
        "Could not find Program class on runner. "
        "Pass program_class=meas.YourProgram explicitly."
    )


def _soccfg_from_runner(runner: Any) -> Any:
    station = _get_attr_or_key(runner, "station", None)
    for obj in (station, runner):
        for name in ("soc", "soccfg"):
            val = _get_attr_or_key(obj, name, None)
            if val is not None:
                return val
    return None


def trace_runner_point(
    runner: Any,
    point: Optional[Union[int, Sequence[int], Mapping[str, int]]] = None,
    *,
    sweep_values: Optional[Mapping[str, Any]] = None,
    swept_params: Optional[Sequence[str]] = None,
    program_class: Optional[Union[Type[Any], str]] = None,
    trace_options: Optional[Mapping[str, Any]] = None,
    strict: bool = True,
    **execute_kwargs: Any,
) -> Tuple[Any, PulseTrace, Any]:
    """
    Trace one concrete sweep point from a CharacterizationRunner-style object.

    This does NOT call runner.execute(), runner.run(), expt.go(), prog.acquire(), or hardware.
    It only builds the same cfg and instantiates a traced Program dry-run.

    Parameters
    ----------
    runner:
        Your CharacterizationRunner instance.
    point:
        Optional sweep index. If omitted together with sweep_values and
        swept_params, cfg.expt is left exactly as runner.execute(...) would
        build it. For 2D sweeps use a tuple, e.g. point=(25, 0). For
        swept_params=['detune', 'length'], that means detunes[25], lengths[0].
    sweep_values:
        Direct scalar overrides, e.g. {'detune': 0.0, 'length': 0.25}.
    swept_params:
        Override cfg.expt.swept_params when needed.
    program_class:
        Optional explicit Program class. Usually not needed if runner.program exists.
    trace_options:
        Options forwarded to trace_program_class().
    strict:
        If True, missing sweep lists raise. If False, missing params are skipped.
    **execute_kwargs:
        The same kwargs you would pass to runner.execute(...).

    Returns
    -------
    prog, trace, cfg
        The traced Program instance, its PulseTrace, and the concrete cfg used.
    """
    cfg_kwargs = dict(execute_kwargs)
    # Some preprocessors assert that swept_params was passed through the same
    # kwargs path as a real runner.execute(...).  trace_runner_point also needs
    # it locally for selecting one concrete point, so forward it to both places.
    if swept_params is not None and "swept_params" not in cfg_kwargs:
        cfg_kwargs["swept_params"] = [swept_params] if isinstance(swept_params, str) else list(swept_params)

    cfg = build_cfg_from_runner(runner, **cfg_kwargs)
    if point is not None or sweep_values is not None or swept_params is not None:
        select_sweep_point(
            cfg,
            point=point,
            sweep_values=sweep_values,
            swept_params=swept_params,
            strict=strict,
        )

    cls = _program_class_from_runner(runner, program_class=program_class)
    soccfg = _soccfg_from_runner(runner)
    prog, trace = trace_program_class(cls, soccfg=soccfg, cfg=cfg, trace_options=trace_options)
    if hasattr(trace, "metadata"):
        trace.metadata.update(
            {
                "runner_class": runner.__class__,
                "experiment_class": _get_attr_or_key(runner, "ExptClass", None),
                "preprocessor": _get_attr_or_key(runner, "preprocessor", None),
                "postprocessor": _get_attr_or_key(runner, "postprocessor", None),
            }
        )
    return prog, trace, cfg


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Scan/trace QICK-style pulse Program classes.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="Static scan of Program classes")
    p_scan.add_argument("roots", nargs="+", help="Repo roots, directories, or .py files")
    p_scan.add_argument("--base", default=None, help="Print inheritance tree/descendants for this base class")
    p_scan.add_argument("--calls", action="store_true", help="Print pulse/timing calls")
    p_scan.add_argument("--csv", default=None, help="Write class summary CSV")
    p_scan.add_argument("--include-tests", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "scan":
        idx = discover_program_classes(args.roots, include_tests=args.include_tests)
        if args.csv:
            idx.write_csv(args.csv)
        if args.base:
            idx.print_tree(args.base)
            classes = idx.descendants(args.base, include_base=True)
        else:
            classes = idx.program_classes()
        for c in classes:
            print(f"{c.filename}:{c.lineno}  class {c.name}({', '.join(c.bases)})  methods={', '.join(c.methods)}")
            if args.calls:
                for call in c.calls:
                    print("    " + call.format())
        print(f"\nclasses: {len(classes)}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
