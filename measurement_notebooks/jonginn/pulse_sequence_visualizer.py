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
- Conditional jumps are recorded as markers, not as alternate branch timelines.
"""
from __future__ import annotations

import ast
import csv
import importlib
import inspect
import json
import math
import re
import sys
import textwrap
import traceback
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

DEFAULT_PULSE_CALLS = {
    # low-level QICK timing/pulse/measurement calls
    "pulse", "setup_and_pulse", "set_pulse_registers", "measure", "setup_and_measure",
    "trigger", "sync_all", "wait_all", "readout", "read", "condj", "label", "mathi",
    "safe_regwi", "regwi", "memwi", "loopnz", "end", "add_gauss", "add_pulse", "add_envelope",
    # lab-specific higher-level calls from this package
    "custom_pulse", "custom_pulse_with_preloaded_wfm", "get_prepulse_creator",
    "play_parity_pulse", "play_joint_parity_pulse", "measure_wrapper", "reset_and_sync",
    "active_reset", "man_reset", "storage_reset", "coupler_reset", "displace_man",
    "core_pulses", "initialize", "body", "_prepare_dark_mode", "_read_dark_mode",
    "_read_large_dark", "_play_m1s_frac_train", "multi_parity_readout",
}

PROGRAM_BASE_HINTS = {
    "AveragerProgram", "RAveragerProgram", "NDAveragerProgram",
    "MMAveragerProgram", "MMRAveragerProgram", "MMDualRailAveragerProgram",
    "MMDualRailRAveragerProgram", "MMRBAveragerProgram",
    "QsimBaseProgram", "DarkBaseProgram",
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
        name = _is_self_attr_call(node)
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

    def __init__(self, classes: Sequence[ProgramClassInfo]):
        self.classes = list(classes)
        self.by_name: Dict[str, List[ProgramClassInfo]] = defaultdict(list)
        for c in self.classes:
            self.by_name[c.name].append(c)

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
        Class base names that should count as Program bases.
    pulse_calls:
        Method names to summarize as pulse/timing/measurement calls.
    include_tests:
        Include files under test/tests directories.
    include_all_program_suffix:
        Include classes whose name ends in 'Program' even if the base is unknown.
    """
    if isinstance(roots, (str, Path)):
        roots = [roots]
    root_paths = [Path(r).expanduser().resolve() for r in roots]
    hints = set(base_hints or PROGRAM_BASE_HINTS)
    target_calls = set(pulse_calls or DEFAULT_PULSE_CALLS)

    files: List[Path] = []
    for root in root_paths:
        if root.is_file() and root.suffix == ".py":
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.rglob("*.py")))

    out: List[ProgramClassInfo] = []
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
            tree = ast.parse(text, filename=str(p))
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = []
            for b in node.bases:
                bases.append(_safe_unparse(b))
            short_bases = {_basename(b) for b in bases}
            is_program = bool(short_bases & hints)
            if include_all_program_suffix and node.name.endswith("Program"):
                is_program = True
            if not is_program:
                continue
            methods = []
            calls: List[StaticCall] = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods.append(item.name)
                    visitor = _PulseCallVisitor(node.name, item.name, filename, target_calls)
                    visitor.visit(item)
                    calls.extend(visitor.calls)
            out.append(
                ProgramClassInfo(
                    name=node.name,
                    bases=tuple(bases),
                    filename=filename,
                    lineno=node.lineno,
                    methods=tuple(methods),
                    calls=tuple(calls),
                )
            )
    return ProgramIndex(sorted(out, key=lambda c: (c.filename, c.lineno, c.name)))


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
        if val is not None and math.isfinite(val):
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
        if val is None or not math.isfinite(val):
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
            if phase is not None and math.isfinite(phase):
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
        mode = str(detail or "phase").lower()
        if mode in {"none", "off", "false"}:
            return ""
        if mode in {"full", "all", "params"}:
            return block.params_text
        if mode in {"phase", "ph"}:
            return phase_text
        if mode in {"compact", "essential"}:
            params = block.events[0].params if block.events else {}
            pieces = []
            if phase_text:
                pieces.append(phase_text)
            if params.get("waveform") == "pulse_to_test":
                sweep = self._block_sweep_text(block)
                if sweep:
                    pieces.append(sweep)
            return " ".join(pieces)
        return phase_text

    def _fit_text_lines(self, text: str, *, width: int, max_lines: int) -> str:
        if not text:
            return ""
        lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [text]
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
        if val is not None and math.isfinite(val) and abs(val - round(val)) < 1e-9:
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
        label = self._normalize_circuit_label(label)

        bits = []
        for key, short in (("freq", "f"), ("gain", "g"), ("length", "len"), ("phase", "ph")):
            if key in params and params[key] is not None:
                bits.append(f"{short}={self._format_circuit_value(params[key])}")
        if style and style not in label:
            bits.insert(0, style)
        return label, " ".join(bits)

    def _circuit_signature(self, event: PulseEvent, *, group_sweeps: bool = True) -> Tuple[Any, ...]:
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
        return (
            event.ch,
            event.kind,
            display_label,
            style,
            waveform,
            params.get("freq"),
            params.get("gain"),
            params.get("length"),
            params.get("phase"),
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

    def to_circuit_blocks(
        self,
        *,
        show_sync: bool = False,
        group_repeats: bool = True,
        group_sweeps: bool = True,
        max_group_gap: float = 1e-9,
    ) -> List[CircuitBlock]:
        """Return execution-order blocks for a circuit-style diagram."""
        selected = []
        for event in sorted(self.events, key=lambda e: (e.t0, e.ch, e.kind, e.label)):
            if not show_sync and (event.ch in {"SYNC", "WAIT"} or event.kind in {"sync", "wait"}):
                continue
            selected.append(event)

        blocks: List[CircuitBlock] = []
        signatures: List[Tuple[Any, ...]] = []
        for event in selected:
            label, params_text = self._circuit_label(event)
            sig = self._circuit_signature(event, group_sweeps=group_sweeps)
            if (
                group_repeats
                and blocks
                and signatures[-1] == sig
                and abs(event.t0 - blocks[-1].t1) <= max_group_gap
            ):
                blocks[-1].events.append(event)
                blocks[-1].count += 1
                if params_text and params_text not in blocks[-1].params_text:
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
        return blocks

    def _draw_circuit_box(self, ax: Any, x: float, y: float, w: float, h: float, color: str) -> None:
        x0, x1 = x - w / 2, x + w / 2
        y0, y1 = y - h / 2, y + h / 2
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=color, linewidth=1.2)

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
        mid = y - 0.04
        amp = h / 2

        if block.kind in {"measure", "trigger"} or block.ch.startswith("ADC"):
            self._draw_circuit_box(ax, x, y, w * 0.72, h * 0.9, color)
            ax.plot([x - w * 0.18, x, x + w * 0.18], [y - h * 0.02, y + h * 0.16, y - h * 0.02], color=color, linewidth=1.1)
            return
        if block.kind in {"sync", "wait"}:
            ax.plot([x0, x1], [y, y], color=color, linewidth=2.0)
            ax.plot([x0, x0], [y - h * 0.18, y + h * 0.18], color=color, linewidth=1.0)
            ax.plot([x1, x1], [y - h * 0.18, y + h * 0.18], color=color, linewidth=1.0)
            return

        if waveform_kind == "flat_top":
            xs = [x0, x0 + 0.18 * w, x1 - 0.18 * w, x1]
            ys = [mid, mid + amp, mid + amp, mid]
            ax.plot(xs, ys, color=color, linewidth=1.8)
            ax.plot([x0, x1], [mid, mid], color="0.55", linewidth=0.6, alpha=0.8)
        elif waveform_kind in {"gaussian", "gauss"}:
            xs = [x0 + w * i / 24 for i in range(25)]
            sigma = 0.18
            ys = []
            for xx in xs:
                z = (xx - x) / max(w, 1e-9)
                ys.append(mid + amp * math.exp(-0.5 * (z / sigma) ** 2))
            ax.plot(xs, ys, color=color, linewidth=1.8)
            ax.plot([x0, x1], [mid, mid], color="0.55", linewidth=0.6, alpha=0.8)
        elif waveform_kind == "drag":
            xs = [x0 + w * i / 28 for i in range(29)]
            sigma = 0.18
            gauss = []
            quad = []
            for xx in xs:
                z = (xx - x) / max(w, 1e-9)
                env = math.exp(-0.5 * (z / sigma) ** 2)
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
            xs = [x0 + w * i / 16 for i in range(17)]
            pattern = [0.00, 0.38, 0.22, 0.68, 0.48, 0.85, 0.42, 0.72, 0.30, 0.62, 0.24, 0.76, 0.45, 0.58, 0.20, 0.36, 0.00]
            ys = [mid + amp * p for p in pattern]
            ax.plot(xs, ys, color=color, linewidth=1.65)
            qys = [mid - 0.30 * amp * math.sin(i * math.pi / 4) for i in range(17)]
            ax.plot(xs, qys, color=color, linewidth=0.9, linestyle=":", alpha=0.8)
            ax.plot([x0, x1], [mid, mid], color="0.55", linewidth=0.6, alpha=0.8)
        elif waveform_kind == "const":
            xs = [x0, x0, x1, x1]
            ys = [mid, mid + amp, mid + amp, mid]
            ax.plot(xs, ys, color=color, linewidth=1.8)
            ax.plot([x0, x1], [mid, mid], color="0.55", linewidth=0.6, alpha=0.8)
        else:
            self._draw_circuit_box(ax, x, y, w, h, color)

    def _draw_repeat_power(self, ax: Any, x: float, y: float, count: int, color: str, *, scale: float = 1.0) -> None:
        left = x - 0.55 * scale
        right = x + 0.55 * scale
        ax.text(left, y + 0.02, "(", ha="center", va="center", fontsize=27, color="black", alpha=0.72, zorder=1)
        ax.text(right, y + 0.02, ")", ha="center", va="center", fontsize=27, color="black", alpha=0.72, zorder=1)
        ax.text(right + 0.005 * scale, y + 0.29, f"$\\times {count}$", ha="left", va="bottom", fontsize=8.5, color="black", zorder=3)

    def plot_circuit(
        self,
        *,
        show_sync: bool = False,
        group_repeats: bool = True,
        group_sweeps: bool = True,
        annotate_params: Union[bool, str] = True,
        annotate_sweep: bool = True,
        waveform_blocks: bool = True,
        detail: str = "phase",
        phase_mode: str = "relative",
        show_zero_phase: bool = False,
        max_label_chars: int = 18,
        step_spacing: float = 1.35,
        row_spacing: float = 1.25,
        detail_wrap_chars: int = 22,
        max_detail_lines: int = 2,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
    ):
        """
        Plot an execution-order block diagram, similar to a quantum circuit.

        Unlike plot(), the x-axis is block order rather than elapsed time.  Adjacent
        repeated operations are collapsed into one block with an xN annotation.
        """
        import matplotlib.pyplot as plt

        if isinstance(annotate_params, str):
            detail = annotate_params
            annotate_params_enabled = detail.lower() not in {"none", "off", "false"}
        else:
            annotate_params_enabled = bool(annotate_params)

        blocks = self.to_circuit_blocks(
            show_sync=show_sync,
            group_repeats=group_repeats,
            group_sweeps=group_sweeps,
        )
        chans = sorted({b.ch for b in blocks})
        if not chans:
            fig, ax = plt.subplots(figsize=figsize or (10, 2))
            ax.set_title(title or "No pulse events recorded")
            return fig, ax

        step_spacing = max(_try_float(step_spacing, 1.35) or 1.35, 0.85)
        row_spacing = max(_try_float(row_spacing, 1.25) or 1.25, 0.9)
        y_of = {ch: i * row_spacing for i, ch in enumerate(chans)}
        max_step = max((b.step for b in blocks), default=0)
        if figsize is None:
            figsize = (
                max(11, 0.82 * step_spacing * (max_step + 1) + 3),
                max(3.2, 0.72 * row_spacing * max(len(chans), 1) + 2.2),
            )
        fig, ax = plt.subplots(figsize=figsize)

        xmin, xmax = -0.85 * step_spacing, (max_step + 0.85) * step_spacing
        channel_aliases = self._hardware_channel_aliases()
        for ch, y in y_of.items():
            ax.plot([xmin, xmax], [y, y], color="0.75", linewidth=0.9)
            main_ch, sub_ch = self._channel_label_parts(ch, channel_aliases)
            ax.text(xmin - 0.18 * step_spacing, y + (0.09 if sub_ch else 0.0), main_ch, ha="right", va="center", fontsize=9)
            if sub_ch:
                ax.text(xmin - 0.18 * step_spacing, y - 0.16, sub_ch, ha="right", va="center", fontsize=6.8, color="0.35")

        colors = {
            "pulse": "C0",
            "measure": "C3",
            "trigger": "C3",
            "sync": "0.35",
            "wait": "0.35",
            "readout": "C2",
        }
        target_sweep_steps = []
        previous_phase_by_channel: Dict[str, float] = {}
        last_label_layout: Dict[str, Tuple[float, float]] = {}
        detail_lane_by_channel: Dict[str, int] = defaultdict(int)
        for block in blocks:
            x = float(block.step) * step_spacing
            y = y_of[block.ch]
            color = colors.get(block.kind, "C0")
            if waveform_blocks:
                self._draw_circuit_waveform(ax, block, x, y, 0.80 * min(step_spacing, 1.45), 0.38, color)
            else:
                self._draw_circuit_box(ax, x, y, 0.76 * min(step_spacing, 1.45), 0.42, color)
            if block.count > 1:
                self._draw_repeat_power(ax, x, y, block.count, color, scale=min(step_spacing, 1.45))

            lines = self._circuit_label_lines(block.label, max_label_chars)
            if len(lines) > 2:
                lines = lines[:2]
                lines[-1] = lines[-1][: max(5, max_label_chars - 3)] + "..."
            label_nudge = 0.0
            last_layout = last_label_layout.get(block.ch)
            if last_layout is not None and x - last_layout[0] < 1.15 * step_spacing:
                label_nudge = 0.16 if last_layout[1] == 0 else 0.0
            last_label_layout[block.ch] = (x, label_nudge)
            ax.text(
                x,
                y + 0.36 + label_nudge,
                "\n".join(lines),
                ha="center",
                va="bottom",
                fontsize=8,
                linespacing=0.92,
            )

            phase_text = self._block_phase_text(
                block,
                previous_phase_by_channel,
                phase_mode=phase_mode,
                show_zero_phase=show_zero_phase,
            )
            detail_text = self._block_detail_text(block, phase_text, detail=detail)
            if annotate_params_enabled and detail_text:
                mode = str(detail).lower()
                wrap_width = max(detail_wrap_chars, 28 if mode in {"full", "all", "params"} else 16)
                detail_text = self._fit_text_lines(detail_text, width=wrap_width, max_lines=max_detail_lines)
                detail_lane = detail_lane_by_channel[block.ch] % 2
                detail_lane_by_channel[block.ch] += 1
                detail_y = y - (0.35 + 0.16 * detail_lane)
                ax.text(
                    x,
                    detail_y,
                    detail_text,
                    ha="center",
                    va="top",
                    fontsize=6.7,
                    linespacing=0.92,
                    color="0.22",
                    bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
                )

            if block.sweep_text:
                target_sweep_steps.append(block.step)

        if annotate_sweep:
            sweep_text = self._global_sweep_text()
            if sweep_text:
                if target_sweep_steps:
                    x0 = min(target_sweep_steps) * step_spacing - 0.42 * step_spacing
                    x1 = max(target_sweep_steps) * step_spacing + 0.42 * step_spacing
                else:
                    x0, x1 = xmin + 0.2, xmax - 0.2
                y = (len(chans) - 1) * row_spacing + 0.56
                ax.plot([x0, x0, x1, x1], [y - 0.08, y, y, y - 0.08], color="C4", linewidth=1.2)
                ax.text((x0 + x1) / 2, y + 0.08, sweep_text, ha="center", va="bottom", fontsize=9, color="C4")

        ax.set_xlim(xmin - 0.45 * step_spacing, xmax + 0.35 * step_spacing)
        top_y = (len(chans) - 1) * row_spacing
        ax.set_ylim(-0.85, top_y + (1.12 if annotate_sweep else 0.48))
        ax.set_yticks([])
        ax.set_xticks([i * step_spacing for i in range(max_step + 1)])
        ax.set_xticklabels([str(i) for i in range(max_step + 1)])
        ax.set_xlabel("operation order")
        ax.set_title(title or "Pulse sequence blocks")
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.subplots_adjust(left=0.11 if channel_aliases else 0.08, right=0.99, bottom=0.15, top=0.88)
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

    # ---- Program template ----------------------------------------------------
    def make_program(self):
        """Replacement for QICK AveragerProgram.make_program()."""
        if not hasattr(self, "trace"):
            self._trace_init({})
        run_initialize = self._trace_options.get("run_initialize", True)
        run_body = self._trace_options.get("run_body", True)
        if run_initialize and hasattr(self, "initialize"):
            self._trace_enter("initialize")
            try:
                self.initialize()
            finally:
                self._trace_exit()
        if run_body and hasattr(self, "body"):
            self.trace.add_marker("loop", "body/start")
            self._trace_enter("body")
            try:
                self.body()
            finally:
                self._trace_exit()
            self.trace.add_marker("loop", "body/end")
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

    def add_gauss(self, ch, name, sigma, length, *args, **kwargs):
        dur = _try_float(length, 0.0) or 0.0
        self._trace_waveforms[(str(ch), str(name))] = dur
        self._trace_waveforms_by_name[str(name)] = dur
        self._trace_waveform_kinds[(str(ch), str(name))] = "gaussian"
        self._trace_waveform_kinds_by_name[str(name)] = "gaussian"
        self.trace.add_marker("waveform", f"add_gauss {name}", params={"ch": ch, "sigma": sigma, "length": length})
        return None

    def add_DRAG(self, ch, name, sigma, length, *args, **kwargs):
        dur = _try_float(length, 0.0) or 0.0
        self._trace_waveforms[(str(ch), str(name))] = dur
        self._trace_waveforms_by_name[str(name)] = dur
        self._trace_waveform_kinds[(str(ch), str(name))] = "drag"
        self._trace_waveform_kinds_by_name[str(name)] = "drag"
        self.trace.add_marker("waveform", f"add_DRAG {name}", params={"ch": ch, "sigma": sigma, "length": length})
        return None

    def add_triangle(self, ch, name, length, *args, **kwargs):
        dur = _try_float(length, 0.0) or 0.0
        self._trace_waveforms[(str(ch), str(name))] = dur
        self._trace_waveforms_by_name[str(name)] = dur
        self._trace_waveform_kinds[(str(ch), str(name))] = "triangle"
        self._trace_waveform_kinds_by_name[str(name)] = "triangle"
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
        self._trace_waveforms[(str(ch), str(name))] = dur
        self._trace_waveforms_by_name[str(name)] = dur
        self._trace_waveform_kinds[(str(ch), str(name))] = "arbitrary"
        self._trace_waveform_kinds_by_name[str(name)] = "arbitrary"
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
        if waveform is None or regs.get("waveform_kind"):
            return
        kind = self._trace_waveform_kinds.get((str(ch), str(waveform)))
        if kind is None:
            kind = self._trace_waveform_kinds_by_name.get(str(waveform))
        if kind:
            regs["waveform_kind"] = kind

    def setup_and_pulse(self, ch, t="auto", **kwargs):
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
    def safe_regwi(self, rp, reg, imm, *args, **kwargs):
        self._trace_register_values[(str(rp), reg)] = imm
        self.trace.add_marker("reg", f"regwi {reg}={imm}", params={"page": rp, "reg": reg, "imm": imm})
        return None

    def regwi(self, rp, reg, imm, *args, **kwargs):
        return self.safe_regwi(rp, reg, imm, *args, **kwargs)

    def bitwi(self, *args, **kwargs):
        self.trace.add_marker("bitwi", "bitwi", params={"args": args, "kwargs": kwargs})
        return None

    def mathi(self, *args, **kwargs):
        self.trace.add_marker("mathi", "mathi", params={"args": args, "kwargs": kwargs})
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
        Sweep index. For 2D sweeps use a tuple, e.g. point=(25, 0). For
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
