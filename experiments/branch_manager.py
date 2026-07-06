"""Local, append-only branch/tag bookkeeping over the existing config version store.

A "branch" is purely a mnemonic: a name -> the 4 station config version IDs
(hardware / multiphoton / man1 / floquet). It stores NO config data of its own --
only pointers into configs/versions/, which stays the source of truth. The server
and worker never see branches; they keep receiving full station YAML as today.

History is preserved because the log is append-only (a reflog). The "current"
value of a branch is just the latest event for that name; never overwritten.

Cheapest-thing-that-works throughout; improve only if it becomes painful.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

# The 4 keys returned by station.update_all_station_snapshots(), in load order.
_ID_KEYS = ("hardware_config", "multiphoton_config", "man1_storage_swap", "floquet_storage_swap")


class BranchManager:
    def __init__(self, station, log_path=None):
        """station: a live MultimodeStation.

        log_path defaults to ``branches.jsonl`` in the current working directory --
        i.e. each user's sandbox notebook dir (measurement_notebooks/<user>/). That
        gives every user their own log, so concurrent users never contend for one
        file. Pass an explicit shared log_path if you want cross-user visibility
        (and accept the concurrency caveats in docs/config_branches.md).
        """
        self.station = station
        self.log_path = Path(log_path) if log_path else Path.cwd() / "branches.jsonl"
        # Set by checkout/commit: which branch we're on and a fingerprint of the
        # configs as loaded, so we can tell if the user has edited them since.
        self._loaded_name = None
        self._loaded_ids = None
        self._loaded_fingerprint = None

    # ---- log primitives (the only thing that touches the file) -------------

    def _append(self, event):
        """Append one JSON line. event always carries: branch, op, ts, + (ids or ref)."""
        event = {**event, "ts": datetime.now().isoformat(timespec="seconds")}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def _events(self):
        """Read all events oldest->newest. Tolerate a torn final line (ignore it)."""
        if not self.log_path.exists():
            return []
        events = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn/partial line -- skip
        return events

    def _state(self):
        """Fold the log -> {branch: latest_event}, dropping branches whose latest op == 'delete'."""
        state = {}
        for e in self._events():
            state[e["branch"]] = e
        return {k: v for k, v in state.items() if v.get("op") != "delete"}

    # ---- the four verbs ----------------------------------------------------

    def commit(self, name, note=""):
        """Snapshot current station state and advance branch `name` to it (one append).

        Refuses if `name`'s latest op is 'tag' (tags are frozen by convention).
        Returns the 4 version IDs recorded.
        """
        latest = self._state().get(name)
        if latest and latest.get("op") == "tag":
            raise ValueError(f"{name!r} is a tag (frozen); use a different name or branch() off it")
        # dedup in the version store makes unchanged configs free
        ids = self.station.update_all_station_snapshots(update_main=False)
        ids = {k: ids.get(k) for k in _ID_KEYS}
        self._append({"branch": name, "op": "commit", "ids": ids, "note": note})
        self._mark_loaded(name, ids)
        return ids

    def checkout(self, name, force=False):
        """Reload the station's 4 configs from branch `name`'s latest IDs.

        Guards against losing un-committed in-RAM edits: raises unless force=True
        (or the station is clean). Returns the IDs loaded.
        """
        latest = self._state().get(name)
        if latest is None:
            raise KeyError(f"no branch/tag named {name!r}")
        if not force and self._is_dirty():
            raise RuntimeError(
                f"station has uncommitted config edits (loaded from {self._loaded_name!r}); "
                f"commit() first or checkout(force=True) to discard"
            )
        ids = latest["ids"]
        # Reuse the station's own loader -- it accepts CFG- IDs and re-attaches datasets.
        self.station._initialize_configs(
            ids["hardware_config"], ids["multiphoton_config"],
            ids["man1_storage_swap"], ids["floquet_storage_swap"],
        )
        self._mark_loaded(name, ids)
        return ids

    def branch(self, new_name, from_name=None, note=""):
        """Create `new_name` pointing at `from_name`'s IDs (default: currently-loaded IDs).

        Records a `from` breadcrumb for lineage. Does not touch the station RAM.
        """
        if from_name is not None:
            src = self._state().get(from_name)
            if src is None:
                raise KeyError(f"no branch/tag named {from_name!r}")
            ids, origin = src["ids"], from_name
        else:
            if self._loaded_ids is None:
                raise RuntimeError("nothing loaded to branch from; checkout() or commit() first")
            ids, origin = self._loaded_ids, self._loaded_name
        self._append({"branch": new_name, "op": "branch", "from": origin, "ids": ids, "note": note})
        return ids

    def tag(self, name, note=""):
        """Freeze the currently-loaded IDs under `name` (commit() will later refuse this name)."""
        if self._loaded_ids is None:
            raise RuntimeError("nothing loaded to tag; checkout() or commit() first")
        self._append({"branch": name, "op": "tag", "ids": self._loaded_ids, "note": note})
        return self._loaded_ids

    def delete(self, name):
        """Tombstone `name` (history stays in the log; it just stops resolving)."""
        if name not in self._state():
            raise KeyError(f"no branch/tag named {name!r}")
        self._append({"branch": name, "op": "delete"})

    # ---- read-only views ---------------------------------------------------

    def list(self):
        """{branch: latest_event} for all live branches/tags."""
        return self._state()

    def show(self, name=None):
        """The 4 current IDs for `name`. If name==None, show the currently checked out branch."""
        if name is None:
            if self._loaded_name is None:
                raise TypeError("No branch checked out. Specify a branch name to show.")
            name = self._loaded_name
        return {'name': name, 'IDs': self._state()[name]["ids"]}

    def log(self, name=None):
        """Full history (all appends), optionally filtered to one branch -- oldest->newest."""
        events = self._events()
        return [e for e in events if name is None or e["branch"] == name]

    # ---- dirty detection (cheap fingerprint, no DB writes) -----------------

    def _mark_loaded(self, name, ids):
        self._loaded_name = name
        self._loaded_ids = ids
        self._loaded_fingerprint = self._fingerprint()

    def _fingerprint(self):
        """A cheap content hash of the 4 in-RAM configs.

        Not the same as ConfigVersionManager's checksum -- we only need to detect
        "did this change since I loaded it". Uses the station's own sanitizer to
        drop the non-serializable dataset objects before hashing.
        """
        st = self.station
        parts = [
            json.dumps(st._get_sanitized_config_copy(st.hardware_cfg), sort_keys=True, default=str),
            json.dumps(dict(st.multimode_cfg), sort_keys=True, default=str),
            st.ds_storage.df.to_csv(index=False),
            st.ds_floquet.df.to_csv(index=False) if st.ds_floquet is not None else "",
        ]
        return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()

    def _is_dirty(self):
        """True if in-RAM configs differ from what we last loaded/committed."""
        if self._loaded_fingerprint is None:
            return False  # nothing tracked yet -> nothing to lose
        return self._fingerprint() != self._loaded_fingerprint
