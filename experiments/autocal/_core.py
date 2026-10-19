"""Step / phase model and the run loop of the automated calibration.

A Step is one experiment of the calibration notebooks: the experiment class,
its default settings (copied from the notebook cell), the run-time arguments
(scan windows centred on the current config), and an ``evaluate`` function that
fits the data, checks the fit (``_checks``) and returns the config changes.
The run loop executes the steps in order through the normal runners (job
queue), applies a change only when its check passes, and stops at the first
failed check. Everything lands in a run folder: one JSON record (job id, old /
new value, check numbers per step) and one PNG per step.

Review: ``<run folder>/index.html`` (``AutocalRun.report``) lists every step with its status, old / new values, the
check numbers, the job ids and the figure; it is rewritten after each step. The jobs are also logged to the lab
notebook as usual when the station has ``log_measurements=True`` (the experiments' own plots).

Changes are made in memory (``station.hardware_cfg`` and ``station.ds_storage``).
At the end of each phase the run saves config versions with
``update_main=False``: the files and DB records are written, the main pointer
does not move. Setting main is left to the user.
"""
import datetime
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from slab import AttrDict

from experiments.characterization_runner import CharacterizationRunner
from experiments.sweep_runner import SweepRunner

SECONDS_PER_JOB = 25.0          # queue + compile + load overhead per job (measured ~15-40 s on pippin)
SEQ_US = 30.0                   # pulse sequence + readout per shot, added to the relax delay


@dataclass
class Change:
    """One config value set by a step. `set` writes the new value; `old` / `new` are recorded."""
    label: str
    old: object
    new: object
    set: Callable[[], None]


@dataclass
class Outcome:
    ok: bool
    info: dict = field(default_factory=dict)
    changes: list = field(default_factory=list)
    plot: Optional[Callable] = None              # plot(ax) for the step figure


@dataclass
class Step:
    """One calibration experiment.

    expt: class name under ``experiments`` (as the notebooks use ``meas.<name>``), or a dotted path
        such as ``'single_qubit.t2_cavity.CavityRamseyExperiment'`` where a name is ambiguous.
        None = a config step without a job (e.g. seed a rung from the rung below); evaluate gets expt=None.
    defaults: the notebook's default cfg.expt (relax delay included).
    kwargs: kwargs(station) -> run-time arguments for the runner (the notebook's execute(...) call).
    evaluate: evaluate(station, expt, kw) -> Outcome.
    preprocessor: the notebook's preprocessor (station, defaults, **kw) -> cfg.expt; None = defaults updated by kw.
    sweep: SweepRunner parameter name (e.g. 'freq') for a 2D sweep step; None = one job.
    seconds: seconds(cfg_expt, kw) -> acquisition time estimate; None = the generic estimate.
    source: notebook and cell the step reproduces.
    """
    name: str
    expt: str
    defaults: dict
    kwargs: Callable
    evaluate: Callable
    preprocessor: Optional[Callable] = None
    sweep: Optional[str] = None
    seconds: Optional[Callable] = None
    source: str = ''


@dataclass
class Phase:
    name: str
    steps: list
    snapshot_storage: bool = False               # also save a man1_storage_swap version at the end


class AutocalStop(RuntimeError):
    """A step failed its check or its job; the run stopped there (values after it are unchanged)."""


def resolve(path):
    import experiments
    obj = experiments
    for part in path.split('.'):
        obj = getattr(obj, part)
    return obj


def default_pre(station, default_expt_cfg, **kw):
    cfg = AttrDict(dict(default_expt_cfg)); cfg.update(kw)
    return cfg


def n_points(cfg):
    """Shots per rep of one job: sweep points x error-amplification rows."""
    n = int(cfg.get('expts', 1) or 1)
    if 'n_pulses' in cfg and np.isscalar(cfg.get('n_pulses')):
        n0, ns = int(cfg.get('n_start', 0) or 0), int(cfg.get('n_step', 1) or 1)
        n *= len(np.arange(n0, int(cfg['n_pulses']) + ns, ns))
    return n


def generic_seconds(cfg, kw):
    reps = int(cfg.get('reps', 1) or 1) * int(cfg.get('rounds', 1) or 1)
    relax = float(cfg.get('relax_delay', 2500))
    return n_points(cfg) * reps * (relax + SEQ_US) * 1e-6


class AutocalRun:
    """Run calibration phases on a station.

    station: a MultimodeStation (real, or mock for a dry run).
    client: job_server.JobClient.
    phases: list of Phase (see experiments.autocal.default_phases).
    out_dir: parent folder; the run writes <out_dir>/autocal_<timestamp>/.
    coupler_A: coupler yoko current set in memory before the first job (None = keep the config value).
    log: lab-notebook logging per job (None = the station setting).
    """

    def __init__(self, station, client, phases, out_dir, coupler_A=None, log=None):
        self.station, self.client, self.phases, self.log = station, client, phases, log
        base = Path(out_dir) / f"autocal_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}"
        self.dir, k = base, 1
        while self.dir.exists():
            k += 1; self.dir = base.with_name(f'{base.name}_{k}')
        self.dir.mkdir(parents=True)
        self.coupler_A = coupler_A
        self.record = dict(started=datetime.datetime.now().isoformat(timespec='seconds'), user=station.user,
                           coupler_A=coupler_A, steps=[], snapshots=[])

    # ------------------------------------------------------------------ planning
    def steps(self, phases=None, start_at=None, stop_at=None):
        """(phase, step) pairs in order, restricted to `phases` (names) and the [start_at, stop_at] step names."""
        out, on = [], start_at is None
        for ph in self.phases:
            if phases is not None and ph.name not in phases:
                continue
            for st in ph.steps:
                on = on or st.name == start_at
                if on:
                    out.append((ph, st))
                if st.name == stop_at:
                    return out
        return out

    def build(self, step):
        """The run-time arguments and the cfg.expt the step would submit now (no job)."""
        kw = step.kwargs(self.station)
        pre = step.preprocessor or default_pre
        sweep_kw = {k: kw[k] for k in ('sweep_start', 'sweep_stop', 'sweep_npts', 'batch') if k in kw}
        cfg = pre(self.station, AttrDict(dict(step.defaults)), **{k: v for k, v in kw.items() if k not in sweep_kw})
        return kw, cfg, sweep_kw

    def budget(self, phases=None, start_at=None, stop_at=None, verbose=True):
        """Estimated time per step from the scan settings (points x reps x (relax + sequence) + job overhead)."""
        rows = []
        for ph, st in self.steps(phases, start_at, stop_at):
            if st.expt is None:
                continue
            kw, cfg, sweep_kw = self.build(st)
            sec = (st.seconds or generic_seconds)(cfg, kw)
            jobs = int(sweep_kw.get('sweep_npts', 1))
            rows.append((ph.name, st.name, jobs * sec + jobs * SECONDS_PER_JOB, jobs))
        if verbose:
            tot = 0.0
            for ph, name, s, jobs in rows:
                tot += s
                print(f'  {ph:12s} {name:34s} {jobs:3d} job(s) {s / 60:6.1f} min')
            print(f'  total {tot / 3600:.2f} h ({len(rows)} steps)')
        return rows

    # ------------------------------------------------------------------ running
    def _runner(self, step):
        cls = resolve(step.expt)
        pre = step.preprocessor or default_pre
        if step.sweep:
            return SweepRunner(station=self.station, ExptClass=cls, default_expt_cfg=AttrDict(dict(step.defaults)),
                               sweep_param=step.sweep, preprocessor=pre, postprocessor=None, live_plot=False,
                               job_client=self.client, use_queue=True)
        return CharacterizationRunner(station=self.station, ExptClass=cls, default_expt_cfg=AttrDict(dict(step.defaults)),
                                      preprocessor=pre, postprocessor=None, job_client=self.client, use_queue=True, show=False)

    def _save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / 'record.json').write_text(json.dumps(self.record, indent=1, default=_json_default))
        (self.dir / 'index.html').write_text(report_html(self.record), encoding='utf-8')

    @property
    def report(self):
        """The review page of this run (rewritten after every step; open it in a browser during the run)."""
        return self.dir / 'index.html'

    def _figure(self, name, outcome):
        if outcome.plot is None:
            return None
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            outcome.plot(ax)
            ax.set_title(f"{name}: {'OK' if outcome.ok else 'FAILED - ' + str(outcome.info.get('reason', ''))}", fontsize=11)
            p = self.dir / f"{len(self.record['steps']):02d}_{name.replace('.', '_')}.png"
            fig.tight_layout(); fig.savefig(p, dpi=90); plt.close(fig)
            return p.name
        except Exception:
            plt.close('all')
            return None

    def snapshot(self, phase, stopped_at=None):
        """Save the in-memory configs as versions (update_main=False: the main pointer does not move). stopped_at: the
        run stopped inside this phase; the version holds the steps of the phase that passed (resume from it)."""
        ids = dict(phase=phase.name, hardware=self.station.snapshot_hardware_config(update_main=False))
        if stopped_at:
            ids['partial, stopped at'] = stopped_at
        if phase.snapshot_storage:
            ids['man1_storage_swap'] = self.station.snapshot_man1_storage_swap(update_main=False)
        self.record['snapshots'].append(ids); self._save()
        print(f'  snapshot after {phase.name}: {ids}')
        return ids

    def _snapshot_on_stop(self, phase, step):
        """Keep the passed steps of the phase the run stopped in, unless the stop is at its first step."""
        done = [s for s in self.record['steps'] if s['phase'] == phase.name and s.get('ok')]
        if done:
            try:
                self.snapshot(phase, stopped_at=step.name)
            except Exception as exc:
                print(f'  snapshot on stop FAILED: {exc}')

    def run(self, phases=None, start_at=None, stop_at=None):
        """Execute the steps; stop (AutocalStop) at the first failed check or job. Returns the record."""
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.coupler_A is not None:
            self.station.hardware_cfg.hw.yoko_coupler.current = float(self.coupler_A)
        todo = self.steps(phases, start_at, stop_at)
        last_phase = None
        for i, (ph, st) in enumerate(todo):
            if last_phase is not None and ph is not last_phase:
                self.snapshot(last_phase)
            last_phase = ph
            rec = dict(phase=ph.name, step=st.name, source=st.source, time=datetime.datetime.now().isoformat(timespec='seconds'))
            t0 = time.time()
            try:
                if st.expt is None:
                    kw = st.kwargs(self.station)
                    out = st.evaluate(self.station, None, kw)
                else:
                    kw = st.kwargs(self.station)
                    runner = self._runner(st)
                    expt = runner.execute(postprocess=False, log=self.log, **kw)
                    rec['jobs'] = list(runner.last_job_ids)
                    out = st.evaluate(self.station, expt, kw)
            except Exception as exc:
                rec.update(ok=False, error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-2000:],
                           seconds=round(time.time() - t0, 1))
                self.record['stopped'] = rec['time']
                self.record['steps'].append(rec); self._save()
                print(f'[autocal] {st.name}: ERROR {exc}')
                self._snapshot_on_stop(ph, st)
                raise AutocalStop(f'{st.name}: {exc}') from exc
            rec.update(ok=bool(out.ok), info=out.info, seconds=round(time.time() - t0, 1),
                       changes=[dict(label=c.label, old=c.old, new=c.new) for c in out.changes])
            rec['figure'] = self._figure(st.name, out)
            if out.ok:
                for c in out.changes:
                    c.set()
            else:
                self.record['stopped'] = rec['time']
            self.record['steps'].append(rec); self._save()
            msg = ', '.join(f'{c.label} {_fmt(c.old)} -> {_fmt(c.new)}' for c in out.changes)
            print(f"[autocal] {st.name} {rec.get('jobs', '')}: {'OK' if out.ok else 'FAILED (' + str(out.info.get('reason')) + ')'}"
                  f"{' | ' + msg if msg else ''}{'' if out.ok else ' -- NOT applied'}")
            if not out.ok:
                self._snapshot_on_stop(ph, st)
                raise AutocalStop(f"{st.name}: {out.info.get('reason')}")
        if last_phase is not None:
            self.snapshot(last_phase)
        self.record['finished'] = datetime.datetime.now().isoformat(timespec='seconds'); self._save()
        return self.record


def _fmt(v):
    if isinstance(v, float):
        return f'{v:.6g}'
    return str(v)


def report_html(rec):
    """One static page per run: per step the status, the reason, the old -> new values, the check numbers, the jobs and
    the figure. Plain HTML next to the PNGs, so it opens from the run folder (local disk or the SMB share)."""
    esc = lambda s: str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    colour = {True: '#1b7f4b', False: '#b3261e'}
    rows = []
    for i, s in enumerate(rec['steps']):
        ok = s.get('ok'); status = 'ERROR' if 'error' in s else ('OK' if ok else 'FAILED')
        why = s.get('error') or (s.get('info') or {}).get('reason') or ''
        ch = ''.join(f"<tr><td>{esc(c['label'])}</td><td>{esc(_fmt(c['old']))}</td><td>{esc(_fmt(c['new']))}</td></tr>"
                     for c in s.get('changes', []))
        applied = 'applied' if ok else 'not applied'
        info = ', '.join(f'{esc(k)}={esc(_fmt(v))}' for k, v in (s.get('info') or {}).items() if k != 'reason' and not isinstance(v, (list, dict)))
        fig = f"<img src='{esc(s['figure'])}' loading='lazy'>" if s.get('figure') else ''
        rows.append(f"""<section id='s{i}'><h2><span style='color:{colour.get(bool(ok) and status == 'OK', '#b3261e')}'>{status}</span>
 {i:02d} {esc(s['step'])}</h2><p class='m'>{esc(s.get('time', ''))} &middot; {esc(s.get('seconds', ''))} s &middot; jobs {esc(', '.join(s.get('jobs', [])) or '-')}
 &middot; {esc(s.get('source', ''))}</p>{f"<p class='w'>{esc(why)}</p>" if why else ''}
 {f"<table><tr><th>value ({applied})</th><th>old</th><th>new</th></tr>{ch}</table>" if ch else ''}
 {f"<p class='m'>{info}</p>" if info else ''}{fig}</section>""")
    toc = ''.join(f"<a href='#s{i}' style='color:{colour.get(bool(s.get('ok')) and 'error' not in s, '#b3261e')}'>{i:02d} {esc(s['step'])}</a><br>"
                  for i, s in enumerate(rec['steps']))
    snaps = ''.join(f"<li>{esc(x)}</li>" for x in rec.get('snapshots', []))
    refresh = '' if rec.get('finished') or rec.get('stopped') else "<meta http-equiv='refresh' content='30'>"
    return f"""<!doctype html><meta charset='utf-8'>{refresh}<title>autocal {esc(rec.get('started', ''))}</title>
<style>body{{font:14px system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:0 16px;background:#fff;color:#222}}
h1{{font-size:20px}}h2{{font-size:16px;margin:24px 0 4px}}.m{{color:#666;margin:2px 0}}.w{{color:#b3261e;margin:4px 0}}
table{{border-collapse:collapse;margin:6px 0}}td,th{{border:1px solid #ddd;padding:2px 8px;text-align:left}}
img{{max-width:100%;border:1px solid #eee}}nav{{column-width:260px;font-size:13px}}section{{border-top:1px solid #eee}}</style>
<h1>Autocal run {esc(rec.get('started', ''))} ({esc(rec.get('user', ''))}, coupler {esc(rec.get('coupler_A'))} A)
{' &middot; finished ' + esc(rec['finished']) if rec.get('finished') else ' &middot; STOPPED ' + esc(rec['stopped']) if rec.get('stopped') else ' &middot; running (this page reloads every 30 s)'}</h1>
<p class='m'>Config versions saved with update_main=False:</p><ul>{snaps or '<li>none yet</li>'}</ul><nav>{toc}</nav>{''.join(rows)}"""


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
