"""Render a fixed reference clip for each checkpoint of a running training process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil


def evaluate(checkpoint: Path, out: Path, npz_dir: Path, clip: str, device: str, max_steps: int) -> bool:
    out.mkdir(parents=True, exist_ok=True)
    trajectory = out / "rollout.npz"
    scripts = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    commands = [
        [sys.executable, str(scripts / "record_rollout.py"), "--ckpt", str(checkpoint),
         "--npz-dir", str(npz_dir), "--clips", clip, "--out", str(trajectory),
         "--device", device, "--max-steps", str(max_steps)],
        [sys.executable, str(scripts / "render_rollout.py"), "--traj", str(trajectory),
         "--out-dir", str(out), "--width", "640", "--height", "360", "--fps", "25"],
    ]
    try:
        with (out / "evaluation.log").open("w", encoding="utf-8") as log:
            for command in commands:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=1800)
        with np.load(trajectory) as data:
            summary = {"checkpoint": str(checkpoint), "clip": clip, "device": device,
                       "max_steps": max_steps, "steps": int(data["lengths"][0]),
                       "mean_return_per_executed_step": float(data["norm_return"][0])}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out / ".failed").unlink(missing_ok=True)
        (out / ".done").write_text("ok\n", encoding="utf-8")
        print(f"[eval] finished {checkpoint.name}: {out}", flush=True)
        return True
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        (out / ".failed").write_text(str(exc), encoding="utf-8")
        print(f"[eval] failed {checkpoint.name}: {exc}; see {out / 'evaluation.log'}", flush=True)
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--train-pid", type=int, required=True)
    parser.add_argument("--clip", default="walk1_subject1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    data = args.npz_dir.resolve()
    process = psutil.Process(args.train_pid)
    started = process.create_time()
    attempted: set[str] = set()
    print(f"[eval] watching {run} for process {args.train_pid}; clip={args.clip}, device={args.device}", flush=True)
    while True:
        for checkpoint in sorted(run.glob("ckpt_[0-9][0-9][0-9][0-9][0-9][0-9][0-9].pt")):
            out = run / "visual_eval" / checkpoint.stem
            if checkpoint.name in attempted or (out / ".done").exists():
                continue
            # Numbered checkpoints are written directly; wait until their mtime is quiet.
            if time.time() - checkpoint.stat().st_mtime < 10:
                continue
            attempted.add(checkpoint.name)
            evaluate(checkpoint, out, data, args.clip, args.device, args.max_steps)
        try:
            alive = process.is_running() and process.create_time() == started and process.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            alive = False
        if not alive:
            latest = run / "ckpt_latest.pt"
            if latest.exists():
                # The final save can occur between numbered checkpoints.
                evaluate(latest, run / "visual_eval" / "final", data, args.clip, args.device, args.max_steps)
            print("[eval] training ended; watcher exiting", flush=True)
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
