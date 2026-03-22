from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np


class ParticleMassTimeseriesCollector:
    def __init__(
        self,
        *,
        run_id: str,
        phase: str,
        enabled: bool,
        output_dir: str | Path = "temp/mass_timeseries",
        sample_interval: int = 1,
        flush_snapshots: int = 64,
    ) -> None:
        self.run_id = run_id
        self.phase = phase
        self.enabled = enabled
        self.sample_interval = max(int(sample_interval), 1)
        self.flush_snapshots = max(int(flush_snapshots), 1)
        self.output_dir = Path(output_dir) / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._pending_steps: list[int] = []
        self._pending_snapshots: list[np.ndarray] = []
        self._recorded_steps: list[int] = []
        self._recorded_snapshots: list[np.ndarray] = []
        self._artifact_paths: list[Path] = []
        self._chunk_index = 0
        self._queue: queue.Queue[tuple[int, np.ndarray, np.ndarray] | None] = queue.Queue()
        self._writer_thread: threading.Thread | None = None

        if self.enabled:
            self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._writer_thread.start()

    def should_record(self, step: int, *, force: bool = False) -> bool:
        if not self.enabled:
            return False
        return force or (step % self.sample_interval == 0)

    def record(self, step: int, particles: np.ndarray, *, force: bool = False) -> bool:
        if not self.should_record(step, force=force):
            return False
        snapshot = np.asarray(particles, dtype=np.float32).copy()
        self._pending_steps.append(int(step))
        self._pending_snapshots.append(snapshot)
        self._recorded_steps.append(int(step))
        self._recorded_snapshots.append(snapshot)
        if len(self._pending_steps) >= self.flush_snapshots:
            self._flush_pending()
        return True

    def finalize(self) -> tuple[np.ndarray, list[np.ndarray], list[Path]]:
        if self.enabled:
            self._flush_pending()
            self._queue.put(None)
            if self._writer_thread is not None:
                self._writer_thread.join()
        steps = np.asarray(self._recorded_steps, dtype=np.int32)
        return steps, list(self._recorded_snapshots), list(self._artifact_paths)

    def _flush_pending(self) -> None:
        if not self._pending_steps:
            return
        steps = np.asarray(self._pending_steps, dtype=np.int32)
        snapshots = np.stack(self._pending_snapshots, axis=0).astype(np.float32, copy=False)
        self._queue.put((self._chunk_index, steps, snapshots))
        self._chunk_index += 1
        self._pending_steps.clear()
        self._pending_snapshots.clear()

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            chunk_index, steps, snapshots = item
            output_path = self.output_dir / f"{self.phase}_chunk_{chunk_index:04d}.npz"
            np.savez_compressed(output_path, steps=steps, masses=snapshots)
            self._artifact_paths.append(output_path)
