"""Bounded application checks and finalization, using the research execution kernel.

Programs run in the bubblewrap sandbox. ``apply`` can instead run one fixed program without it, but only after the
user names the program's sha256 (``trusted_program_sha256``): the run then records ``program_isolation: none``, opens
no editor kernel, and refuses any other program. ``synthesize`` always uses the sandbox.
"""

from __future__ import annotations

import platform
import os
import re
import shutil
import sys
import time
from importlib.metadata import version
from dataclasses import asdict
from pathlib import Path

import numpy as np

from weightcode.application.bundles import (bind, copy_task, dataset, digest, pack_program, program_bundle,
                                           program_source, read, sha, task_bundle, write)
from weightcode.application.resnet import export_checkpoint, forward_module, load_buffers, reference, reference_logits
from weightcode.common.identity import load_params, params_sha256, save_params
from weightcode.harness.program import ProgramProblem, Student, run_update, write_derived_patch
from weightcode.harness.session import SessionPaths
from weightcode.patch import Limits, PatchError, apply_patch
from weightcode.patch.limits import dense_delta_bytes, with_dense_file_cap
from weightcode.sandbox.host import DEFAULT_BWRAP, LocalKernel, SandboxConfig, SandboxKernel
from weightcode.sandbox.program import ISOLATION_NONE, ProgramSandbox, UnisolatedRunner

ISOLATION_BWRAP = "bubblewrap"
EDITOR_KERNEL_BWRAP = "bubblewrap"
EDITOR_KERNEL_LOCAL = "local"
PROGRAM_MEMORY_BYTES = 8 * 2**30

NO_NAMESPACES_NOTE = ("host disables unprivileged user namespaces, so no bubblewrap sandbox exists: the editor's "
                      "Python kernel and every update program run in plain processes with memory rlimits, wall "
                      "caps and the NumPy-only interpreter, but without filesystem or network isolation")


def trusted_sha256(value: str) -> str:
    """A sha256 the user vouches for, as 64 lowercase hex digits."""
    text = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"program sha256 must be 64 hex digits, got {value!r}")
    return text


def score(params: dict, data: dict) -> dict:
    forward = forward_module(len(params["fc.bias"])).forward
    result = {}
    for role in ("new", "old"):
        x, y = data[f"x_{role}"], data[f"y_{role}"]
        correct = 0
        # Small chunks keep host memory bounded independently of dataset length.
        for start in range(0, len(x), 4):
            pred = forward(params, x[start:start + 4]).argmax(axis=1)
            correct += int(np.count_nonzero(pred == y[start:start + 4]))
        result[role] = {"correct": correct, "n": len(y), "accuracy": correct / len(y)}
    return result


def decision(base: dict, child: dict, task: dict) -> dict:
    gain = child["new"]["accuracy"] - base["new"]["accuracy"]
    old = child["old"]["accuracy"] - base["old"]["accuracy"]
    return {"new_gain": gain, "old_change": old, "net_gain": gain + old,
            "eligible": gain > 0 and gain + 1e-12 >= task["min_new_gain"]
            and old + 1e-12 >= -task["retention_drop"]}


def deployment_score(params: dict, buffers: dict, data: dict) -> dict:
    model = reference(params, buffers)
    result = {}
    for role in ("new", "old"):
        logits = reference_logits(model, data[f"x_{role}"])
        y = data[f"y_{role}"]
        correct = int(np.count_nonzero(logits.argmax(1) == y))
        result[role] = {"correct": correct, "n": len(y), "accuracy": correct / len(y)}
    return result


def prepare_run(task: Path, out: Path, sandbox_python: Path | None, *, checks: int, cells: int,
                wall_s: float, program_wall_s: float = 300, threads: int = 4, bwrap: Path = DEFAULT_BWRAP,
                trusted_program_sha256: str | None = None,
                editor_isolation: str = EDITOR_KERNEL_BWRAP) -> dict:
    """Create the run directory. ``sandbox_python`` None means no sandbox (module docstring): it needs
    ``trusted_program_sha256`` and ``cells == 0``. ``editor_isolation`` "local" starts the editor kernel and
    runs programs without bubblewrap on hosts that disable unprivileged user namespaces; the run records
    ``program_isolation: none`` and a note, and ``sandbox_python`` stays required for the NumPy-only kernel."""
    if type(checks) is not int or checks < 1 or type(cells) is not int or cells < 0 \
            or not 0 < wall_s < float("inf") or not 0 < program_wall_s < float("inf") or threads < 1:
        raise ValueError("checks>=1, cells>=0, finite positive timeouts and threads>=1 required")
    if editor_isolation not in (EDITOR_KERNEL_BWRAP, EDITOR_KERNEL_LOCAL):
        raise ValueError(f"editor_isolation must be {EDITOR_KERNEL_BWRAP!r} or {EDITOR_KERNEL_LOCAL!r}")
    trusted = None if trusted_program_sha256 is None else trusted_sha256(trusted_program_sha256)
    unsandboxed_editor = editor_isolation == EDITOR_KERNEL_LOCAL
    if sandbox_python is None:
        if trusted is None or cells:
            raise ValueError("a run without the sandbox needs the trusted program's sha256 and no editor cells")
        probe = {"status": "not-run", "isolation": ISOLATION_NONE,
                 "reason": "the user named the program's sha256; it runs without bubblewrap"}
    elif unsandboxed_editor:
        probe = doctor_local(sandbox_python, threads=threads)
    else:
        # Includes real sandbox startup, imports and filesystem/network isolation checks.
        probe = doctor(sandbox_python, threads=threads, bwrap=bwrap)
    task_bundle(task)
    paths = SessionPaths(out.absolute())
    paths.root.parent.mkdir(parents=True, exist_ok=True)
    paths.create()
    copy_task(task, paths.private / "task")
    t, m, params = task_bundle(paths.private / "task")
    paths.trial.mkdir()
    for name in ("params.npz", "forward.py"):
        shutil.copyfile(paths.private / "task" / "model" / name, paths.trial / name)
    shutil.copyfile(paths.private / "task" / "experience.npz", paths.trial / "experience.npz")
    write(paths.trial / "layout.json", m)
    limits = with_dense_file_cap(Limits(max_ops=max(64, len(params))), dense_delta_bytes(v.shape for v in params.values()))
    public = {"goal": t["goal"], "retention_drop": t["retention_drop"], "min_new_gain": t["min_new_gain"],
              "limits": asdict(limits), "selection_rule": t["selection_rule"], "checks": checks}
    write(paths.trial / "task.json", public)
    base = score(params, dataset(paths.private / "task" / "selection.npz", len(m["labels"])))
    write(paths.private / "baseline.json", base)
    config = {"schema": "weightcode.application_run.v1", "task_manifest_sha256": sha(paths.private / "task" / "manifest.json"),
              "sandbox_python": None if sandbox_python is None else str(sandbox_python.absolute()),
              "program_isolation": ISOLATION_NONE if (sandbox_python is None or unsandboxed_editor) else ISOLATION_BWRAP,
              "editor_kernel": editor_isolation,
              "isolation_note": NO_NAMESPACES_NOTE if unsandboxed_editor else None,
              "trusted_program_sha256": trusted, "bwrap": str(bwrap), "checks": checks, "cells": cells,
              "wall_s": wall_s, "program_wall_s": program_wall_s, "threads": threads,
              "limits": asdict(limits), "started_unix": time.time(), "python": platform.python_version(),
              "numpy": np.__version__, "platform": platform.platform(), "sandbox_probe": probe,
              "torch": version("torch"), "torchvision": version("torchvision"),
              "invocation": [sys.executable, *sys.argv],
              "application_source_sha256": digest({p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}),
              "trial_files": bind(paths.trial,
                  ["params.npz", "forward.py", "experience.npz", "layout.json", "task.json"])}
    write(paths.root / "run.json", config)
    return config


def doctor(python: Path, *, threads: int = 4, bwrap: Path = DEFAULT_BWRAP) -> dict:
    """Start the real sandbox with ``python`` and check what a program inside can see; raise if anything fails."""
    import tempfile
    from weightcode.sandbox.host import check_bwrap, interpreter_mounts

    version_text = check_bwrap(bwrap)
    mounts = interpreter_mounts(python)
    with tempfile.TemporaryDirectory(prefix="weightcode-doctor-") as tmp:
        root = Path(tmp)
        for name in ("trial", "patches", "scratch"):
            (root / name).mkdir()
        (root / "hidden.txt").write_text("not mounted")
        config = SandboxConfig(root / "trial", root / "patches", root / "scratch", root / "cells.jsonl",
                               python.absolute(), threads=threads, max_cells=1, bwrap=bwrap)
        with SandboxKernel(config) as kernel:
            code = ("import os, pathlib, socket, numpy, sys\n"
                    f"assert not pathlib.Path({str(root / 'hidden.txt')!r}).exists()\n"
                    "assert pathlib.Path('/work/trial').is_dir()\n"
                    "assert socket.if_nameindex() == [(1, 'lo')]\n"
                    "assert 'torch' not in sys.modules\nprint(numpy.__version__)")
            result = kernel.run(code)
            if result.error or result.timed_out:
                raise RuntimeError(f"sandbox self-check failed: {result.error} {result.stderr}")
    return {"status": "ok", "bubblewrap": version_text, "bwrap": str(bwrap), "numpy": result.stdout.strip(),
            "sandbox_python": str(python.absolute()), "interpreter_binds": [str(p) for p in mounts.binds],
            "threads": threads}


def doctor_local(python: Path, *, threads: int = 4) -> dict:
    """Start the LocalKernel once and run one cell; raise if the kernel protocol fails. No isolation claims."""
    import tempfile

    from weightcode.sandbox.host import kernel_sha256

    with tempfile.TemporaryDirectory(prefix="weightcode-doctor-local-") as tmp:
        root = Path(tmp)
        for name in ("trial", "patches", "scratch"):
            (root / name).mkdir()
        config = SandboxConfig(root / "trial", root / "patches", root / "scratch", root / "cells.jsonl",
                               python.absolute(), threads=threads, max_cells=1)
        with LocalKernel(config) as kernel:
            code = "import sys, numpy\nassert 'torch' not in sys.modules\nprint(numpy.__version__)"
            result = kernel.run(code)
            if result.error or result.timed_out:
                raise RuntimeError(f"local kernel self-check failed: {result.error} {result.stderr}")
        return {"status": "ok", "isolation": ISOLATION_NONE, "kernel": "local", "numpy": result.stdout.strip(),
                "sandbox_python": str(python.absolute()), "kernel_sha256": kernel_sha256(), "threads": threads}


_QUOTED_ABSOLUTE_PATH = re.compile(r"[\"'](/[^\"']*)[\"']")


def forbidden_run_paths(code: str, paths) -> list[str]:
    """Quoted absolute paths inside the run directory but outside trial/patches/scratch (local-kernel guard).

    Without namespaces the editor's kernel can read the host; this static scan restores the sandbox's core
    confidentiality rule for the run's own data: selection.npz, final.npz and the baseline live under the run
    root outside the three editor-visible directories. Cell code that names such a path is refused before it
    runs. Paths outside the run directory are the host's business, not this guard's.
    """
    root = os.path.abspath(paths.root)
    allowed = [os.path.abspath(p) for p in (paths.trial, paths.patches, paths.scratch)]
    offenders = []
    for match in _QUOTED_ABSOLUTE_PATH.finditer(code):
        p = os.path.abspath(match.group(1))
        if not (p == root or p.startswith(root + os.sep)):
            continue
        if any(p == a or p.startswith(a + os.sep) for a in allowed):
            continue
        offenders.append(match.group(1))
    return offenders


class Application:
    """One run. All checks reserve a durable slot before running any candidate."""

    def __init__(self, root: Path):
        self.paths = SessionPaths(root.absolute())
        self.config = read(self.paths.root / "run.json")
        if self.config.get("schema") != "weightcode.application_run.v1":
            raise ValueError("unsupported application run")
        self.task_root = self.paths.private / "task"
        if sha(self.task_root / "manifest.json") != self.config["task_manifest_sha256"]:
            raise ValueError("run task changed")
        self.task, self.model, self.params = task_bundle(self.task_root)
        self.selection = dataset(self.task_root / "selection.npz", len(self.model["labels"]))
        self.base = read(self.paths.private / "baseline.json")
        self.limits = Limits(**self.config["limits"])
        self.isolation = self.config.get("program_isolation", ISOLATION_BWRAP)
        self.bwrap = Path(self.config.get("bwrap", DEFAULT_BWRAP))
        if self.isolation == ISOLATION_NONE:
            self.sandbox: ProgramSandbox | UnisolatedRunner = UnisolatedRunner(
                Path(sys.executable), PROGRAM_MEMORY_BYTES, self.config["threads"])
        else:
            self.sandbox = ProgramSandbox(Path(self.config["sandbox_python"]), PROGRAM_MEMORY_BYTES,
                                          self.config["threads"], bwrap=self.bwrap)
        self.kernel: SandboxKernel | None = None

    def attempts(self) -> list[Path]:
        return sorted(self.paths.attempts.glob("[0-9][0-9][0-9][0-9]"))

    def status(self) -> dict:
        elapsed = time.time() - self.config["started_unix"]
        return {"checks_used": len(self.attempts()), "checks_remaining": max(0, self.config["checks"] - len(self.attempts())),
                "wall_s_remaining": max(0.0, self.config["wall_s"] - elapsed), "selection_base": self.base,
                "frozen": (self.paths.root / "freeze.json").exists()}

    def require_open(self) -> None:
        s = self.status()
        if s["frozen"] or s["wall_s_remaining"] <= 0:
            raise ValueError("run is frozen or editor wall budget exhausted")

    def python(self, code: str) -> dict:
        self.require_open()
        if self.isolation == ISOLATION_NONE and self.config.get("editor_kernel") != EDITOR_KERNEL_LOCAL:
            raise ValueError("this run has no sandbox, so it runs no editor code")
        if self.config.get("editor_kernel") == EDITOR_KERNEL_LOCAL:
            bad = forbidden_run_paths(code, self.paths)
            if bad:
                raise ValueError("this cell names run paths outside trial/patches/scratch "
                                 f"({', '.join(sorted(set(bad))[:3])}); the selection and final data "
                                 "are not available to the editor")
        if self.kernel is None:
            p = self.paths
            # The bwrap sessions of record ran on fast servers; the local kernel on an HPC login node
            # needs longer cells for NumPy forward passes over a full experience split.
            cap = 600 if self.config.get("editor_kernel") == EDITOR_KERNEL_LOCAL else 120
            config = SandboxConfig(p.trial, p.patches, p.scratch, p.cells_log,
                Path(self.config["sandbox_python"]), max_cells=self.config["cells"], threads=self.config["threads"],
                cell_timeout_s=min(cap, self.status()["wall_s_remaining"]), bwrap=self.bwrap)
            self.kernel = (LocalKernel(config) if self.config.get("editor_kernel") == EDITOR_KERNEL_LOCAL
                           else SandboxKernel(config))
            self.kernel.start()
        return asdict(self.kernel.run(code))

    def check(self, name: str) -> dict:
        self.require_open()
        if not self.status()["checks_remaining"]:
            raise ValueError("check budget exhausted")
        if not name or Path(name).name != name or not name.endswith(".py"):
            raise ValueError("program_file must be a .py basename in /work/patches")
        source = self.paths.patches / name
        if source.is_symlink() or not source.is_file():
            raise ValueError("program must be a regular file in /work/patches")
        # Reserved before syntax validation/execution; invalid checks also consume a slot.
        attempt = self.paths.attempts / f"{len(self.attempts()) + 1:04d}"
        attempt.mkdir()
        record = {"index": len(self.attempts()), "status": "rejected"}
        try:
            (attempt / "program.py").write_bytes(program_source(source))
            record["program_sha256"] = sha(attempt / "program.py")
            trusted = self.config.get("trusted_program_sha256")
            if trusted is not None and record["program_sha256"] != trusted:
                raise ValueError(f"program sha256 {record['program_sha256']} is not the trusted {trusted}")
            student = Student(self.params, self.task_root / "model" / "params.npz",
                              self.task_root / "model" / "forward.py", {}, self.model)
            update = run_update(student, self.task_root / "experience.npz", attempt / "program.py",
                                attempt / "run", self.sandbox,
                                wall_s=min(self.config["program_wall_s"], self.status()["wall_s_remaining"]))
            write(attempt / "execution.json", update.record)
            patch = write_derived_patch(update, self.params, self.model["params_sha256"], attempt)
            result = apply_patch(self.params, patch, attempt, self.limits, self.model["params_sha256"])
            if any(np.any(v < 0) for k, v in result.child.items() if k.endswith("running_var")):
                raise ValueError("candidate has negative BatchNorm variance")
            measured = score(result.child, self.selection)
            record.update(status="checked", score=measured, decision=decision(self.base, measured, self.task),
                          patch=result.summary())
            save_params(attempt / "child.npz", result.child)
        except (ProgramProblem, PatchError, SyntaxError, ValueError) as exc:
            record["error"] = str(exc)
            if isinstance(exc, ProgramProblem) and exc.run is not None:
                write(attempt / "execution.json", dict(exc.run))
        write(attempt / "check.json", record)
        return record

    def freeze(self) -> dict:
        frozen = self.paths.root / "freeze.json"
        if frozen.exists():
            return read(frozen)
        eligible = []
        for path in self.attempts():
            if (path / "check.json").is_file():
                r = read(path / "check.json")
                if r["status"] == "checked" and r["decision"]["eligible"]:
                    eligible.append((r["decision"]["net_gain"], -r["index"], path, r))
        best = max(eligible, key=lambda item: item[:2]) if eligible else None
        record = {"schema": "weightcode.application_freeze.v1", "selected": best[2].name if best else None,
                  "program_sha256": best[3]["program_sha256"] if best else None,
                  "child_sha256": best[3]["patch"]["child_sha256"] if best else self.model["params_sha256"],
                  "checks_used": len(self.attempts())}
        write(frozen, record)
        return record

    def finalize(self) -> dict:
        self.close()
        frozen = self.freeze()  # No final-set scoring until after this write-once decision.
        if self.paths.result_json.exists():
            raise FileExistsError("run already finalized")
        selected = frozen["selected"]
        candidate = self.params
        if selected:
            pick = self.paths.attempts / selected
            candidate = load_params(pick / "child.npz")
            if params_sha256(candidate) != frozen["child_sha256"] or sha(pick / "program.py") != frozen["program_sha256"]:
                raise ValueError("frozen candidate changed")
        final = dataset(self.task_root / "final.npz", len(self.model["labels"]))
        base = score(self.params, final)
        measured = score(candidate, final) if selected else base
        verdict = decision(base, measured, self.task)
        buffers = load_buffers(self.task_root / "model")
        deployed_base = deployment_score(self.params, buffers, final)
        deployed_candidate = deployment_score(candidate, buffers, final) if selected else deployed_base
        deployed_verdict = decision(deployed_base, deployed_candidate, self.task)
        accepted = bool(selected and verdict["eligible"] and deployed_verdict["eligible"])
        released = candidate if accepted else self.params
        save_params(self.paths.final / "params.npz", released)
        export = export_checkpoint(released, buffers,
                                   self.paths.final / "model.pt", final["x_new"][:1])
        if selected:
            pack_program(pick / "program.py", self.task_root / "model", self.paths.final / "program",
                provenance={"kind": "application-run", "task_manifest_sha256": self.config["task_manifest_sha256"],
                            "source_model_sha256": self.model["params_sha256"], "final_accepted": accepted,
                            "final_decision": verdict, "frozen_program_sha256": frozen["program_sha256"]})
        report = {"schema": "weightcode.application_report.v1", "status": "accepted" if accepted else
                  "final-rejected" if selected else "base-kept", "freeze": frozen, "final_base": base,
                  "final_candidate": measured, "final_decision": verdict, "export": export,
                  "torchvision_final_base": deployed_base, "torchvision_final_candidate": deployed_candidate,
                  "torchvision_final_decision": deployed_verdict,
                  "output_is_updated_model": accepted, "run_manifest_sha256": sha(self.paths.root / "run.json"),
                  "program_isolation": self.isolation,
                  "note": "Application validation, not a reproduction of frozen paper experiments."}
        write(self.paths.result_json, report)
        write(self.paths.final / "report.json", report)
        return report

    def close(self) -> None:
        if self.kernel is not None:
            self.kernel.close()
            self.kernel = None


def apply_program(task: Path, program: Path, out: Path, sandbox_python: Path | None, *,
                  program_sha256: str | None = None, **kwargs) -> dict:
    """Reuse one program bundle on a task. ``sandbox_python`` None runs it without the sandbox; that needs
    ``program_sha256``, the sha256 of ``program.py`` the user trusts. A given ``program_sha256`` is always enforced."""
    _, model, _ = task_bundle(task)
    manifest = program_bundle(program, model)
    if program_sha256 is not None and manifest["files"]["program.py"] != trusted_sha256(program_sha256):
        raise ValueError(f"{program}/program.py has sha256 {manifest['files']['program.py']}, "
                         f"not the trusted {trusted_sha256(program_sha256)}")
    if sandbox_python is None and program_sha256 is None:
        raise ValueError("without --sandbox-python, pass --program-sha256 with the sha256 of the program you trust")
    prepare_run(task, out, sandbox_python, checks=1, cells=0, trusted_program_sha256=program_sha256, **kwargs)
    app = Application(out)
    shutil.copyfile(program / "program.py", app.paths.patches / "program.py")
    if sha(app.paths.patches / "program.py") != manifest["files"]["program.py"]:
        raise ValueError("program changed during snapshotting")
    app.check("program.py")
    return app.finalize()
