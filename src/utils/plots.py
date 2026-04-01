import os
from pathlib import Path
import shutil
from typing import Any
from typing import Callable

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image
from PIL import ImageDraw
from scipy.stats import gaussian_kde

os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/mesa_shader_cache")

_TITLE_COLOR = "#1f2933"
_TEXT_COLOR = "#52606d"
_GRID_COLOR = "#7b8794"
_PRIMARY_COLOR = "#1d4ed8"
_PRIMARY_FILL = "#93c5fd"
_PRIMARY_MARKER = "#1e3a8a"
_REFERENCE_COLOR = "#ea580c"


def _patch_mujoco_renderer_close() -> None:
    renderer_cls = mujoco.Renderer
    if getattr(renderer_cls, "_simbay_safe_close_patched", False):
        return

    original_close = renderer_cls.close

    def _safe_close(self: Any) -> None:
        if not hasattr(self, "_gl_context"):
            self._gl_context = None
        if not hasattr(self, "_mjr_context"):
            self._mjr_context = None
        original_close(self)

    renderer_cls.close = _safe_close
    renderer_cls._simbay_safe_close_patched = True


_patch_mujoco_renderer_close()


def _style_axis(ax: Any) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c8c4ba")
    ax.spines["bottom"].set_color("#c8c4ba")
    ax.tick_params(colors=_TEXT_COLOR, labelsize=10.5)
    ax.grid(True, linestyle="--", linewidth=0.75, alpha=0.22, color=_GRID_COLOR)


def _set_panel_title(ax: Any, title: str) -> None:
    ax.set_title(title, fontsize=12.5, fontweight="semibold", color=_TITLE_COLOR, pad=10)


def _style_legend(ax: Any, *, loc: str = "best", ncol: int = 1, fontsize: float = 9.5) -> None:
    legend = ax.legend(
        loc=loc,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#d9e2ec",
        ncol=ncol,
        fontsize=fontsize,
    )
    if legend is not None:
        legend.get_frame().set_linewidth(0.8)


def _add_stat_block(ax: Any, text: str, *, x: float = 0.02, y: float = 0.04) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=9.4,
        color=_TEXT_COLOR,
        ha="left",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#d9e2ec",
            "linewidth": 0.8,
            "alpha": 0.92,
        },
    )


def _percentage_axis_limits(values: list[float]) -> tuple[float, float]:
    valid_values = [value for value in values if not np.isnan(value)]
    if not valid_values:
        return 0.0, 100.0
    data_min = min(valid_values)
    data_max = max(valid_values)
    span = max(data_max - data_min, 5.0)
    padding = max(span * 0.12, 2.0)
    lower = max(0.0, data_min - padding)
    upper = min(100.0, data_max + padding)
    if upper - lower < 5.0:
        midpoint = (upper + lower) / 2.0
        lower = max(0.0, midpoint - 2.5)
        upper = min(100.0, midpoint + 2.5)
    return lower, upper


def _backend_display_name(backend: str) -> str:
    if backend == "mujoco-warp":
        return "Warp-accelerated MuJoCo simulation"
    if backend == "mujoco":
        return "MuJoCo simulation"
    return backend.replace("-", " ").strip()


def _hardware_display_name(backend: str) -> str:
    if backend == "mujoco-warp":
        return "GPU"
    return "CPU"


def _plot_subtitle(
    backend: str,
    num_particles: int,
    *,
    extra: str | None = None,
) -> str:
    subtitle = (
        f"{_backend_display_name(backend)} • {num_particles} particles • "
        f"hardware: {_hardware_display_name(backend)}"
    )
    if extra:
        subtitle = f"{subtitle} • {extra}"
    return subtitle


def build_plot_grid(
    *,
    run_id: str,
    output_name: str,
    title: str,
    subtitle: str,
    figsize: tuple[float, float],
    nrows: int,
    ncols: int,
    plot_builders: list[Callable[[Any], None]],
) -> Path:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor="white")
    axes_array = np.atleast_1d(axes).reshape(-1)
    fig.suptitle(
        f"{title}\n{subtitle}",
        fontsize=15,
        fontweight="semibold",
        color=_TITLE_COLOR,
        y=0.98,
    )
    for ax, builder in zip(axes_array, plot_builders):
        _style_axis(ax)
        builder(ax)
    for ax in axes_array[len(plot_builders) :]:
        ax.set_visible(False)

    output_dir = _run_output_dir(run_id)
    output_path = output_dir / f"{output_name}.png"
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_single_plot(
    *,
    run_id: str,
    output_name: str,
    title: str,
    subtitle: str,
    figsize: tuple[float, float],
    plot_builder: Callable[[Any], None],
) -> Path:
    return build_plot_grid(
        run_id=run_id,
        output_name=output_name,
        title=title,
        subtitle=subtitle,
        figsize=figsize,
        nrows=1,
        ncols=1,
        plot_builders=[plot_builder],
    )


def _save_multi_axes_plot(
    *,
    run_id: str,
    output_name: str,
    title: str,
    subtitle: str,
    figsize: tuple[float, float],
    nrows: int,
    ncols: int,
    plot_builder: Callable[[Any], None],
) -> Path:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor="white")
    fig.suptitle(
        f"{title}\n{subtitle}",
        fontsize=15,
        fontweight="semibold",
        color=_TITLE_COLOR,
        y=0.98,
    )
    plot_builder(axes)
    output_path = _run_output_dir(run_id) / f"{output_name}.png"
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _reset_directory(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def _run_output_dir(run_id: str) -> Path:
    output_dir = Path("plots") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _particle_cloud_output_paths(run_id: str) -> tuple[Path, Path]:
    base_dir = _run_output_dir(run_id) / "particle_cloud"
    images_dir = base_dir / "images"
    _reset_directory(images_dir)
    gif_path = base_dir / "particle_cloud.gif"
    return images_dir, gif_path


def _posterior_evolution_output_paths(run_id: str) -> tuple[Path, Path]:
    base_dir = _run_output_dir(run_id) / "particle_posterior"
    images_dir = base_dir / "images"
    _reset_directory(images_dir)
    gif_path = base_dir / "particle_posterior.gif"
    return images_dir, gif_path


def _real_robot_execution_output_paths(run_id: str) -> tuple[Path, Path]:
    base_dir = _run_output_dir(run_id) / "real_robot_execution"
    images_dir = base_dir / "images"
    _reset_directory(images_dir)
    gif_path = base_dir / "real_robot_execution.gif"
    return images_dir, gif_path


def _posterior_kde_output_paths(run_id: str) -> tuple[Path, Path]:
    base_dir = _run_output_dir(run_id) / "particle_posterior_kde"
    images_dir = base_dir / "images"
    _reset_directory(images_dir)
    gif_path = base_dir / "particle_posterior_kde.gif"
    return images_dir, gif_path


def _likelihood_landscape_output_paths(run_id: str) -> tuple[Path, Path]:
    base_dir = _run_output_dir(run_id) / "likelihood_landscape"
    images_dir = base_dir / "images"
    _reset_directory(images_dir)
    gif_path = base_dir / "likelihood_landscape.gif"
    return images_dir, gif_path


def _wall_duration_ess_output_paths(run_id: str) -> tuple[Path, Path]:
    base_dir = _run_output_dir(run_id) / "wall_duration_vs_ess"
    images_dir = base_dir / "images"
    _reset_directory(images_dir)
    gif_path = base_dir / "wall_duration_vs_ess.gif"
    return images_dir, gif_path


def _build_execution_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.azimuth = 132.0
    camera.elevation = -22.0
    camera.distance = 1.75
    camera.lookat[:] = np.array([0.45, 0.0, 0.32], dtype=np.float64)
    return camera


def _concatenate_trajectories(
    trajectories: list[tuple[str, list[np.ndarray] | np.ndarray]],
) -> list[tuple[str, int, np.ndarray]]:
    timeline: list[tuple[str, int, np.ndarray]] = []
    for phase_name, trajectory in trajectories:
        for step_index, qpos in enumerate(trajectory):
            timeline.append(
                (phase_name, step_index, np.asarray(qpos, dtype=np.float64))
            )
    return timeline


def _evaluate_particle_kde(
    particle_values: np.ndarray,
    x_grid: np.ndarray,
    *,
    bw_method: str | float,
) -> np.ndarray:
    if particle_values.size == 0:
        return np.zeros_like(x_grid)
    if particle_values.size == 1 or np.allclose(
        particle_values, particle_values[0], atol=1e-9
    ):
        sigma = max((x_grid[-1] - x_grid[0]) * 0.01, 1e-3)
        density = np.exp(-0.5 * ((x_grid - particle_values[0]) / sigma) ** 2)
        return density / np.trapezoid(density, x_grid)
    try:
        kde = gaussian_kde(particle_values, bw_method=bw_method)
        density = kde(x_grid)
    except (np.linalg.LinAlgError, ValueError):
        sigma = max(np.std(particle_values), 1e-3)
        density = np.exp(-0.5 * ((x_grid - np.mean(particle_values)) / sigma) ** 2)
    area = np.trapezoid(density, x_grid)
    if area <= 0.0 or not np.isfinite(area):
        return np.zeros_like(x_grid)
    return density / area


def _smooth_likelihood_curve(
    masses: np.ndarray,
    likelihoods: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    masses = np.asarray(masses, dtype=np.float64).reshape(-1)
    likelihoods = np.asarray(likelihoods, dtype=np.float64).reshape(-1)
    valid_mask = np.isfinite(masses) & np.isfinite(likelihoods)
    masses = masses[valid_mask]
    likelihoods = likelihoods[valid_mask]
    if masses.size == 0:
        return np.zeros_like(x_grid)
    span = max(float(np.max(masses) - np.min(masses)), 1e-3)
    bandwidth = max(float(np.std(masses)) * 0.25, span / 35.0, 0.03)
    distances = (x_grid[:, None] - masses[None, :]) / bandwidth
    kernels = np.exp(-0.5 * distances**2)
    weight_sum = np.sum(kernels, axis=1)
    smoothed = np.divide(
        kernels @ likelihoods,
        weight_sum,
        out=np.zeros_like(x_grid),
        where=weight_sum > 1e-12,
    )
    peak = float(np.max(smoothed)) if smoothed.size else 0.0
    if peak > 0.0:
        smoothed = smoothed / peak
    return smoothed


def plot_mass_estimation_evolution(
    ax: Any,
    *,
    history_estimates: list[float],
    initial_particles: np.ndarray,
    particle_history: list[np.ndarray],
    true_mass: float,
    env: Any,
) -> None:
    num_steps = len(history_estimates)
    steps_to_plot = min(num_steps, len(particle_history))
    initial_snapshot = np.asarray(initial_particles, dtype=np.float64).reshape(-1)

    if initial_snapshot.size > 0:
        ax.scatter(
            np.zeros(initial_snapshot.shape[0], dtype=np.float64),
            initial_snapshot,
            color=_PRIMARY_COLOR,
            alpha=0.35,
            s=14,
            zorder=3,
            label="Initial particle values",
        )

    for t in range(steps_to_plot):
        snapshot = np.asarray(particle_history[t], dtype=np.float64).reshape(-1)
        if snapshot.size == 0:
            continue
        ax.scatter(
            [t] * snapshot.shape[0], snapshot, color=_PRIMARY_FILL, alpha=0.08, s=12
        )

    mass_steps = range(num_steps)
    ax.plot(
        mass_steps,
        history_estimates,
        color=_PRIMARY_COLOR,
        linewidth=2.8,
        label="Filter Estimate (Mean)",
    )
    ax.fill_between(
        mass_steps,
        history_estimates,
        true_mass,
        color=_PRIMARY_FILL,
        alpha=0.18,
    )
    ax.axhline(
        y=true_mass,
        color=_REFERENCE_COLOR,
        linestyle="--",
        linewidth=1.8,
        label=f"True Mass ({true_mass} kg)",
    )
    if history_estimates:
        ax.scatter(
            [num_steps - 1],
            [history_estimates[-1]],
            color=_PRIMARY_MARKER,
            s=44,
            zorder=4,
        )

    _set_panel_title(ax, "Mass Estimate")
    ax.set_xlabel("Simulation Step (lifting phase)", fontsize=11)
    ax.set_ylabel("Estimated mass (kg)", fontsize=11)
    ax.set_ylim(env.min, env.max)
    _style_legend(ax, loc="lower right")
    if history_estimates:
        final_error_kg = abs(true_mass - history_estimates[-1])
        _add_stat_block(
            ax,
            (
                f"Final estimate: {history_estimates[-1]:.3f} kg\n"
                f"Absolute error: {final_error_kg:.3f} kg"
            ),
        )


def generate_particle_cloud_gif(
    *,
    particle_history: list[np.ndarray],
    history_estimates: list[float],
    likelihood_diagnostics_history: list[dict[str, float]],
    true_mass: float,
    env: Any,
    run_id: str,
    backend: str,
    num_particles: int,
    frame_duration_ms: int = 120,
    max_step: int | None = 300,
) -> Path:
    images_dir, gif_path = _particle_cloud_output_paths(run_id)
    frame_paths: list[Path] = []
    primary_color = "#1d4ed8"
    primary_fill = "#93c5fd"
    reference_color = "#ea580c"

    final_step = len(particle_history) - 1
    if max_step is not None:
        final_step = min(final_step, max_step)

    for step, snapshot in enumerate(particle_history[: final_step + 1]):
        frame_path = images_dir / f"frame_{step:04d}.png"
        particle_values = np.asarray(snapshot, dtype=np.float64).reshape(-1)
        fig, ax = plt.subplots(figsize=(8.5, 6), facecolor="white")
        _style_axis(ax)

        if particle_values.size > 0:
            y_jitter = np.linspace(-0.15, 0.15, particle_values.size)
            ax.scatter(
                particle_values,
                y_jitter,
                color=primary_fill,
                edgecolors=primary_color,
                linewidths=0.6,
                alpha=0.8,
                s=36,
            )

        estimate = (
            history_estimates[min(step, len(history_estimates) - 1)]
            if history_estimates
            else true_mass
        )
        diagnostics = (
            likelihood_diagnostics_history[step]
            if step < len(likelihood_diagnostics_history)
            else {}
        )
        ax.axvline(
            x=true_mass,
            color=reference_color,
            linestyle="--",
            linewidth=2.0,
            label=f"True mass ({true_mass:.3f} kg)",
        )
        ax.axvline(
            x=estimate,
            color=primary_color,
            linewidth=2.4,
            label=f"Estimate ({estimate:.3f} kg)",
        )
        ax.set_title(
            f"Particle Cloud\nStep {step + 1} • {backend} backend • {num_particles} particles",
            fontsize=15,
            fontweight="bold",
            color="#1f2933",
        )
        ax.set_xlabel("Particle Mass (kg)", fontsize=12)
        ax.set_ylabel("Particle Spread", fontsize=12)
        ax.set_xlim(env.min, env.max)
        ax.set_ylim(-0.3, 0.3)
        ax.set_yticks([])
        ax.legend(loc="upper right", frameon=False)
        if particle_values.size > 0:
            likelihood_lines: list[str] = []
            if diagnostics:
                likelihood_lines = [
                    f"Lik mean/std: {diagnostics.get('likelihood_mean', float('nan')):.3f} / {diagnostics.get('likelihood_std', float('nan')):.3f}",
                    f"Lik min/max: {diagnostics.get('likelihood_min', float('nan')):.3f} / {diagnostics.get('likelihood_max', float('nan')):.3f}",
                    f"Finite ratio: {diagnostics.get('likelihood_finite_ratio', float('nan')):.3f}",
                ]
            ax.text(
                0.02,
                0.04,
                "\n".join(
                    [
                        f"Particle mean: {np.mean(particle_values):.3f} kg",
                        f"Particle std: {np.std(particle_values):.3f} kg",
                        *likelihood_lines,
                    ]
                ),
                transform=ax.transAxes,
                fontsize=10.5,
                color="#52606d",
                ha="left",
                va="bottom",
            )
        fig.tight_layout()
        fig.savefig(frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)

    if not frame_paths:
        raise ValueError(
            "particle_history is empty; cannot generate particle cloud GIF."
        )

    frames = [Image.open(frame_path).copy() for frame_path in frame_paths]
    try:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
            optimize=False,
        )
    finally:
        for frame in frames:
            frame.close()
    return gif_path


def generate_posterior_evolution_gif(
    *,
    particle_history: list[np.ndarray],
    history_estimates: list[float],
    resample_events: list[bool],
    true_mass: float,
    env: Any,
    run_id: str,
    backend: str,
    num_particles: int,
    frame_duration_ms: int = 120,
    max_step: int | None = 300,
    bins: int = 200,
) -> Path:
    images_dir, gif_path = _posterior_evolution_output_paths(run_id)
    frame_paths: list[Path] = []

    final_step = len(particle_history) - 1
    if max_step is not None:
        final_step = min(final_step, max_step)

    for step, snapshot in enumerate(particle_history[: final_step + 1]):
        frame_path = images_dir / f"frame_{step:04d}.png"
        particle_values = np.asarray(snapshot, dtype=np.float64).reshape(-1)
        estimate = (
            history_estimates[min(step, len(history_estimates) - 1)]
            if history_estimates
            else true_mass
        )
        did_resample = (
            bool(resample_events[step]) if step < len(resample_events) else False
        )
        rel_error_pct = (
            abs(estimate - true_mass) / true_mass * 100.0 if true_mass != 0 else 0.0
        )

        fig, ax = plt.subplots(figsize=(9.5, 6.2), facecolor="white")
        _style_axis(ax)

        fill_color = "#dc2626" if did_resample else "#93c5fd"
        edge_color = "#16a34a" if did_resample else "#1d4ed8"
        estimate_color = "#16a34a" if did_resample else "#1d4ed8"
        true_mass_color = "#ea580c"

        if particle_values.size > 0:
            ax.hist(
                particle_values,
                bins=bins,
                range=(env.min, env.max),
                density=True,
                color=fill_color,
                edgecolor=edge_color,
                linewidth=1.0,
                alpha=0.82,
            )

        ax.axvline(
            x=true_mass,
            color=true_mass_color,
            linestyle="--",
            linewidth=2.0,
            label=f"True mass ({true_mass:.3f} kg)",
        )
        ax.axvline(
            x=estimate,
            color=estimate_color,
            linewidth=2.4,
            label=f"Estimate ({estimate:.3f} kg)",
        )
        ax.set_title(
            "Particle Posterior Evolution",
            fontsize=15,
            fontweight="bold",
            color="#1f2933",
        )
        ax.set_xlabel("Particle Mass (kg)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_xlim(env.min, env.max)
        ax.legend(loc="upper right", frameon=False)
        ax.text(
            0.02,
            0.96,
            f"Step {step + 1} • {backend} backend • {num_particles} particles",
            transform=ax.transAxes,
            fontsize=11,
            color="#52606d",
            ha="left",
            va="top",
        )
        ax.text(
            0.02,
            0.89,
            f"Resampled: {'Yes' if did_resample else 'No'} • Relative error: {rel_error_pct:.2f}%",
            transform=ax.transAxes,
            fontsize=11,
            color="#52606d",
            ha="left",
            va="top",
        )
        if particle_values.size > 0:
            ax.text(
                0.02,
                0.04,
                f"Mean: {np.mean(particle_values):.3f} kg\nStd: {np.std(particle_values):.3f} kg",
                transform=ax.transAxes,
                fontsize=10.5,
                color="#52606d",
                ha="left",
                va="bottom",
            )

        fig.tight_layout()
        fig.savefig(frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)

    if not frame_paths:
        raise ValueError(
            "particle_history is empty; cannot generate posterior evolution GIF."
        )

    frames = [Image.open(frame_path).copy() for frame_path in frame_paths]
    try:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
            optimize=False,
        )
    finally:
        for frame in frames:
            frame.close()
    return gif_path


def generate_posterior_kde_gif(
    *,
    particle_history: list[np.ndarray],
    ess_history: list[float],
    true_mass: float,
    env: Any,
    run_id: str,
    backend: str,
    num_particles: int,
    frame_duration_ms: int = 120,
    max_step: int | None = 300,
    bw_method: str | float = "silverman",
    grid_points: int = 512,
) -> Path:
    images_dir, gif_path = _posterior_kde_output_paths(run_id)
    frame_paths: list[Path] = []

    final_step = len(particle_history) - 1
    if max_step is not None:
        final_step = min(final_step, max_step)

    x_grid = np.linspace(env.min, env.max, grid_points, dtype=np.float64)
    max_density = 0.0
    cached_densities: list[np.ndarray] = []
    for snapshot in particle_history[: final_step + 1]:
        particle_values = np.asarray(snapshot, dtype=np.float64).reshape(-1)
        density = _evaluate_particle_kde(
            particle_values,
            x_grid,
            bw_method=bw_method,
        )
        cached_densities.append(density)
        if density.size:
            max_density = max(max_density, float(np.max(density)))

    for step, snapshot in enumerate(particle_history[: final_step + 1]):
        frame_path = images_dir / f"frame_{step:04d}.png"
        particle_values = np.asarray(snapshot, dtype=np.float64).reshape(-1)
        density = cached_densities[step]
        ess_value = ess_history[min(step, len(ess_history) - 1)] if ess_history else float("nan")

        fig, ax = plt.subplots(figsize=(9.5, 6.2), facecolor="white")
        _style_axis(ax)

        ax.fill_between(
            x_grid,
            density,
            0,
            color="#93c5fd",
            alpha=0.32,
        )
        ax.plot(
            x_grid,
            density,
            color="#1d4ed8",
            linewidth=2.8,
            label="Posterior KDE",
        )
        ax.axvline(
            x=true_mass,
            color="#ea580c",
            linestyle="--",
            linewidth=2.0,
            label=f"True mass ({true_mass:.3f} kg)",
        )

        ax.set_title(
            "Animated Posterior KDE",
            fontsize=15,
            fontweight="bold",
            color="#1f2933",
        )
        ax.set_xlabel("Particle Mass (kg)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_xlim(env.min, env.max)
        ax.set_ylim(0.0, max(max_density * 1.08, 1e-6))
        ax.legend(loc="upper right", frameon=False)
        ax.text(
            0.02,
            0.96,
            f"Step {step + 1} • {backend} backend • {num_particles} particles",
            transform=ax.transAxes,
            fontsize=11,
            color="#52606d",
            ha="left",
            va="top",
        )
        ax.text(
            0.02,
            0.89,
            f"ESS: {ess_value:.1f}" if np.isfinite(ess_value) else "ESS: unavailable",
            transform=ax.transAxes,
            fontsize=11,
            color="#52606d",
            ha="left",
            va="top",
        )
        if particle_values.size > 0:
            ax.text(
                0.02,
                0.04,
                f"Mean: {np.mean(particle_values):.3f} kg\nStd: {np.std(particle_values):.3f} kg",
                transform=ax.transAxes,
                fontsize=10.5,
                color="#52606d",
                ha="left",
                va="bottom",
            )

        fig.tight_layout()
        fig.savefig(frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)

    if not frame_paths:
        raise ValueError("particle_history is empty; cannot generate posterior KDE GIF.")

    frames = [Image.open(frame_path).copy() for frame_path in frame_paths]
    try:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
            optimize=False,
        )
    finally:
        for frame in frames:
            frame.close()
    return gif_path


def generate_real_robot_execution_gif(
    *,
    model: mujoco.MjModel,
    trajectories: list[tuple[str, list[np.ndarray] | np.ndarray]],
    dt: float,
    run_id: str,
    backend: str,
    frame_duration_ms: int = 80,
    width: int = 960,
    height: int = 720,
) -> Path:
    images_dir, gif_path = _real_robot_execution_output_paths(run_id)
    timeline = _concatenate_trajectories(trajectories)
    if not timeline:
        raise ValueError(
            "No robot trajectory steps available for execution GIF generation."
        )

    frame_paths: list[Path] = []
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = _build_execution_camera()

    try:
        for global_step, (phase_name, phase_step, qpos) in enumerate(timeline):
            data.ctrl[: qpos.shape[0]] = qpos
            mujoco.mj_step(model, data)

            renderer.update_scene(data, camera=camera)
            rgb_image = renderer.render()
            frame_image = Image.fromarray(rgb_image)
            draw = ImageDraw.Draw(frame_image)
            timestamp_seconds = global_step * dt
            overlay_lines = [
                "Real Robot MuJoCo Execution",
                f"Phase: {phase_name}  Step: {phase_step + 1}",
                f"Timestamp: {timestamp_seconds:.3f}s  Backend: {backend}",
            ]
            draw.rectangle((18, 18, 460, 106), fill=(255, 255, 255))
            draw.multiline_text(
                (34, 30),
                "\n".join(overlay_lines),
                fill=(31, 41, 51),
                spacing=6,
            )

            frame_path = images_dir / f"frame_{global_step:04d}.png"
            frame_image.save(frame_path)
            frame_paths.append(frame_path)
    finally:
        renderer.close()

    frames = [Image.open(frame_path).copy() for frame_path in frame_paths]
    try:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
            optimize=False,
        )
    finally:
        for frame in frames:
            frame.close()

    return gif_path


def generate_likelihood_landscape_gif(
    *,
    likelihood_history: list[np.ndarray],
    likelihood_particle_history: list[np.ndarray],
    particle_history: list[np.ndarray],
    ess_history: list[float],
    true_mass: float,
    env: Any,
    run_id: str,
    backend: str,
    num_particles: int,
    frame_duration_ms: int = 120,
    max_step: int | None = 300,
    grid_points: int = 512,
) -> Path:
    images_dir, gif_path = _likelihood_landscape_output_paths(run_id)
    frame_paths: list[Path] = []
    final_step = min(len(likelihood_history), len(likelihood_particle_history), len(particle_history)) - 1
    if max_step is not None:
        final_step = min(final_step, max_step)
    if final_step < 0:
        raise ValueError("No likelihood history available; cannot generate likelihood landscape GIF.")

    x_grid = np.linspace(env.min, env.max, grid_points, dtype=np.float64)
    cached_curves: list[np.ndarray] = []
    max_curve_value = 0.0
    for step in range(final_step + 1):
        curve = _smooth_likelihood_curve(
            np.asarray(likelihood_particle_history[step], dtype=np.float64),
            np.asarray(likelihood_history[step], dtype=np.float64),
            x_grid,
        )
        cached_curves.append(curve)
        max_curve_value = max(max_curve_value, float(np.max(curve, initial=0.0)))

    for step in range(final_step + 1):
        frame_path = images_dir / f"frame_{step:04d}.png"
        current_particles = np.asarray(particle_history[step], dtype=np.float64).reshape(-1)
        current_curve = cached_curves[step]
        ess_value = ess_history[min(step, len(ess_history) - 1)] if ess_history else float("nan")

        fig, ax = plt.subplots(figsize=(10.2, 6.2), facecolor="white")
        _style_axis(ax)
        ax.fill_between(x_grid, current_curve, 0.0, color="#93c5fd", alpha=0.28)
        ax.plot(x_grid, current_curve, color="#1d4ed8", linewidth=2.8, label="Likelihood landscape")
        ax.axvline(
            x=true_mass,
            color="#ea580c",
            linestyle="--",
            linewidth=2.0,
            label=f"True mass ({true_mass:.3f} kg)",
        )
        if current_particles.size > 0:
            ax.scatter(
                current_particles,
                np.zeros_like(current_particles) - 0.025,
                marker="|",
                color="#111827",
                s=180,
                linewidths=1.2,
                alpha=0.75,
                label="Particles",
            )
        ax.set_title(
            "Likelihood Landscape Sweep",
            fontsize=15,
            fontweight="bold",
            color="#1f2933",
        )
        ax.set_xlabel("Mass (kg)", fontsize=12)
        ax.set_ylabel("Normalized likelihood", fontsize=12)
        ax.set_xlim(env.min, env.max)
        ax.set_ylim(-0.06, max(max_curve_value * 1.08, 1.0))
        ax.legend(loc="upper right", frameon=False)
        ax.text(
            0.02,
            0.96,
            f"Step {step + 1} • {backend} backend • {num_particles} particles",
            transform=ax.transAxes,
            fontsize=11,
            color="#52606d",
            ha="left",
            va="top",
        )
        ax.text(
            0.02,
            0.89,
            f"ESS: {ess_value:.1f}" if np.isfinite(ess_value) else "ESS: unavailable",
            transform=ax.transAxes,
            fontsize=11,
            color="#52606d",
            ha="left",
            va="top",
        )
        if current_curve.size > 0:
            peak_index = int(np.argmax(current_curve))
            ax.text(
                0.02,
                0.04,
                f"Peak mass: {x_grid[peak_index]:.3f} kg\nPeak value: {current_curve[peak_index]:.3f}",
                transform=ax.transAxes,
                fontsize=10.5,
                color="#52606d",
                ha="left",
                va="bottom",
            )
        fig.tight_layout()
        fig.savefig(frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)

    frames = [Image.open(frame_path).copy() for frame_path in frame_paths]
    try:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
            optimize=False,
        )
    finally:
        for frame in frames:
            frame.close()
    return gif_path


def generate_wall_duration_vs_ess_gif(
    *,
    ess_history: list[float],
    pf_wall_durations: list[float],
    resample_events: list[bool],
    num_particles: int,
    run_id: str,
    backend: str,
    frame_duration_ms: int = 120,
    max_step: int | None = 300,
) -> Path:
    images_dir, gif_path = _wall_duration_ess_output_paths(run_id)
    frame_paths: list[Path] = []
    final_step = min(len(ess_history), len(pf_wall_durations)) - 1
    if max_step is not None:
        final_step = min(final_step, max_step)
    if final_step < 0:
        raise ValueError("No ESS or wall-duration history available; cannot generate joint GIF.")

    steps = np.arange(final_step + 1)
    ess_values = np.asarray(ess_history[: final_step + 1], dtype=np.float64)
    wall_ms = np.asarray(pf_wall_durations[: final_step + 1], dtype=np.float64) * 1000.0
    resample_steps = [idx for idx, flag in enumerate(resample_events[: final_step + 1]) if flag]
    ess_threshold = num_particles / 2.0

    for step in range(final_step + 1):
        frame_path = images_dir / f"frame_{step:04d}.png"
        fig, (ax_ess, ax_wall) = plt.subplots(1, 2, figsize=(13.6, 5.4), facecolor="white")
        _style_axis(ax_ess)
        _style_axis(ax_wall)

        ax_ess.fill_between(steps, ess_values, 0, color="#93c5fd", alpha=0.22)
        ax_ess.plot(steps, ess_values, color="#1d4ed8", linewidth=2.6, label="ESS")
        ax_ess.axhline(
            y=ess_threshold,
            color="#ea580c",
            linewidth=1.8,
            linestyle="--",
            label="Resample threshold",
        )
        if resample_steps:
            ax_ess.scatter(resample_steps, ess_values[resample_steps], color="#dc2626", s=28, zorder=3, label="Resample event")
        ax_ess.axvline(step, color="#111827", linewidth=1.6, alpha=0.85)
        ax_ess.scatter([step], [ess_values[step]], color="#111827", s=44, zorder=4)
        ax_ess.set_title("Effective Sample Size", fontsize=14, fontweight="bold", color="#1f2933")
        ax_ess.set_xlabel("Simulation Step", fontsize=12)
        ax_ess.set_ylabel("ESS", fontsize=12)
        ax_ess.set_xlim(0, max(final_step, 1))
        ax_ess.set_ylim(0, max(float(num_particles), float(np.max(ess_values, initial=0.0))) * 1.05)
        ax_ess.legend(loc="upper right", frameon=False, fontsize=9)

        bar_colors = np.array(["#93c5fd"] * len(steps), dtype=object)
        for idx in resample_steps:
            bar_colors[idx] = "#fca5a5"
        bar_colors[step] = "#1d4ed8"
        ax_wall.bar(steps, wall_ms, color=bar_colors, width=0.9, edgecolor="none")
        ax_wall.axvline(step, color="#111827", linewidth=1.6, alpha=0.85)
        ax_wall.set_title("Per-step Wall Duration", fontsize=14, fontweight="bold", color="#1f2933")
        ax_wall.set_xlabel("Simulation Step", fontsize=12)
        ax_wall.set_ylabel("Wall time (ms)", fontsize=12)
        ax_wall.set_xlim(-0.5, max(final_step, 1) + 0.5)
        ax_wall.set_ylim(0, max(float(np.max(wall_ms, initial=0.0)) * 1.12, 1.0))

        fig.suptitle(
            f"Wall Duration vs ESS\n{backend} backend • {num_particles} particles • step {step + 1}",
            fontsize=17,
            fontweight="bold",
            color="#1f2933",
            y=0.99,
        )
        fig.text(
            0.5,
            0.02,
            (
                f"Current ESS: {ess_values[step]:.1f} • Current wall: {wall_ms[step]:.2f} ms • "
                f"Resampled: {'Yes' if step in resample_steps else 'No'}"
            ),
            ha="center",
            va="bottom",
            fontsize=10.5,
            color="#52606d",
        )
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
        fig.savefig(frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)

    frames = [Image.open(frame_path).copy() for frame_path in frame_paths]
    try:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
            optimize=False,
        )
    finally:
        for frame in frames:
            frame.close()
    return gif_path


def plot_effective_sample_size(
    ax: Any,
    *,
    ess_history: list[float],
    num_particles: int,
) -> None:
    ess_steps = range(len(ess_history))
    threshold = num_particles / 2

    ax.fill_between(
        ess_steps,
        ess_history,
        0,
        color=_PRIMARY_FILL,
        alpha=0.18,
    )
    ax.plot(ess_steps, ess_history, color=_PRIMARY_COLOR, linewidth=2.5, label="ESS")
    ax.axhline(
        y=threshold,
        color=_REFERENCE_COLOR,
        linestyle="--",
        linewidth=1.6,
        label="Resample Threshold",
    )
    if ess_history:
        ax.scatter(
            [len(ess_history) - 1],
            [ess_history[-1]],
            color=_PRIMARY_MARKER,
            s=44,
            zorder=4,
        )

    _set_panel_title(ax, "Effective Sample Size")
    ax.set_xlabel("Simulation Step (lifting phase)", fontsize=11)
    ax.set_ylabel("ESS particles", fontsize=11)
    ax.set_xlim(0, max(len(ess_history) - 1, 0))
    ax.set_ylim(
        0,
        (
            max(float(num_particles), max(ess_history, default=0.0)) * 1.05
            if ess_history
            else float(num_particles)
        ),
    )
    _style_legend(ax, loc="lower right")
    if ess_history:
        _add_stat_block(
            ax,
            f"Final ESS: {ess_history[-1]:.1f}\nResample threshold: {threshold:.1f}",
        )


def generate_particle_filter_plot(
    *,
    history_estimates: list[float],
    ess_history: list[float],
    initial_particles: np.ndarray,
    particle_history: list[np.ndarray],
    true_mass: float,
    env: Any,
    backend: str,
    num_particles: int,
    run_id: str,
) -> Path:
    return build_plot_grid(
        run_id=run_id,
        output_name="particle_filter_evolution",
        title="Particle Filter Summary",
        subtitle=_plot_subtitle(backend, num_particles),
        figsize=(17, 6.8),
        nrows=1,
        ncols=2,
        plot_builders=[
            lambda ax: plot_mass_estimation_evolution(
                ax,
                history_estimates=history_estimates,
                initial_particles=initial_particles,
                particle_history=particle_history,
                true_mass=true_mass,
                env=env,
            ),
            lambda ax: plot_effective_sample_size(
                ax,
                ess_history=ess_history,
                num_particles=num_particles,
            ),
        ],
    )


def plot_resample_events_timeline(
    ax: Any,
    *,
    resample_events: list[bool],
) -> None:
    event_steps = [
        step for step, did_resample in enumerate(resample_events) if did_resample
    ]
    ax.grid(True, axis="x", linestyle="--", linewidth=0.75, alpha=0.22, color=_GRID_COLOR)

    if event_steps:
        ax.vlines(
            event_steps, ymin=0.0, ymax=1.0, color="#ea580c", linewidth=2.2, alpha=0.9
        )
        ax.scatter(
            event_steps, [1.0] * len(event_steps), color="#1d4ed8", s=42, zorder=3
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No resample events recorded",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color=_TEXT_COLOR,
        )

    _set_panel_title(ax, "Resample Events")
    ax.set_xlabel("Simulation Step (lifting phase)", fontsize=11)
    ax.set_ylabel("Resample", fontsize=12)
    ax.set_xlim(0, max(len(resample_events) - 1, 0))
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["No", "Yes"])
    _add_stat_block(ax, f"Resample count: {len(event_steps)}")


def generate_resample_events_timeline_plot(
    *,
    resample_events: list[bool],
    backend: str,
    num_particles: int,
    run_id: str,
) -> Path:
    return build_plot_grid(
        run_id=run_id,
        output_name="resample_events_timeline",
        title="Resampling Timeline",
        subtitle=_plot_subtitle(backend, num_particles),
        figsize=(12, 3.8),
        nrows=1,
        ncols=1,
        plot_builders=[
            lambda ax: plot_resample_events_timeline(
                ax,
                resample_events=resample_events,
            )
        ],
    )


def plot_relative_error(
    ax: Any,
    *,
    history_estimates: list[float],
    true_mass: float,
) -> None:
    rel_error_pct = [
        (abs(estimate - true_mass) / true_mass * 100.0) if true_mass != 0 else 0.0
        for estimate in history_estimates
    ]
    steps = range(len(rel_error_pct))
    if rel_error_pct:
        ax.fill_between(steps, rel_error_pct, 0, color=_PRIMARY_FILL, alpha=0.18)
        ax.plot(
            steps,
            rel_error_pct,
            color=_PRIMARY_COLOR,
            linewidth=2.5,
            label="Relative error",
        )
        ax.scatter(
            [len(rel_error_pct) - 1],
            [rel_error_pct[-1]],
            color=_PRIMARY_MARKER,
            s=40,
            zorder=3,
        )

    _set_panel_title(ax, "Relative Error")
    ax.set_xlabel("Simulation Step (lifting phase)", fontsize=11)
    ax.set_ylabel("Error (%)", fontsize=12)
    ax.set_xlim(0, max(len(rel_error_pct), 1) - 1)
    if rel_error_pct:
        _style_legend(ax, loc="lower right")

    if rel_error_pct:
        _add_stat_block(
            ax,
            (
                f"Final error: {rel_error_pct[-1]:.2f}%\n"
                f"Peak error: {np.max(rel_error_pct):.2f}%"
            ),
        )


def plot_gpu_usage(
    ax: Any,
    *,
    gpu_utilization_history: list[float],
) -> None:
    gpu_steps = range(len(gpu_utilization_history))
    valid_gpu_values = [
        value for value in gpu_utilization_history if not np.isnan(value)
    ]
    if valid_gpu_values:
        ax.fill_between(
            gpu_steps, gpu_utilization_history, 0, color=_PRIMARY_FILL, alpha=0.18
        )
        ax.plot(
            gpu_steps,
            gpu_utilization_history,
            color=_PRIMARY_COLOR,
            linewidth=2.5,
            label="GPU utilization",
        )
        ax.scatter(
            [len(gpu_utilization_history) - 1],
            [gpu_utilization_history[-1]],
            color=_PRIMARY_MARKER,
            s=40,
            zorder=3,
        )
        ax.axhline(
            y=np.mean(valid_gpu_values),
            color=_REFERENCE_COLOR,
            linewidth=1.6,
            linestyle="--",
            label="Average",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No GPU usage samples recorded",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color=_TEXT_COLOR,
        )

    _set_panel_title(ax, "GPU Utilization")
    ax.set_xlabel("Simulation Step (lifting phase)", fontsize=11)
    ax.set_ylabel("Utilization (%)", fontsize=12)
    ax.set_xlim(0, max(len(gpu_utilization_history), 1) - 1)
    ax.set_ylim(*_percentage_axis_limits(gpu_utilization_history))
    if valid_gpu_values:
        _style_legend(ax, loc="lower right")
        _add_stat_block(
            ax,
            (
                f"Peak GPU: {np.max(valid_gpu_values):.1f}%\n"
                f"Average GPU: {np.mean(valid_gpu_values):.1f}%"
            ),
        )


def plot_vram_usage(
    ax: Any,
    *,
    gpu_vram_utilization_history: list[float],
) -> None:
    vram_steps = range(len(gpu_vram_utilization_history))
    valid_vram_values = [
        value for value in gpu_vram_utilization_history if not np.isnan(value)
    ]
    if valid_vram_values:
        ax.fill_between(
            vram_steps,
            gpu_vram_utilization_history,
            0,
            color=_PRIMARY_FILL,
            alpha=0.18,
        )
        ax.plot(
            vram_steps,
            gpu_vram_utilization_history,
            color=_PRIMARY_COLOR,
            linewidth=2.5,
            label="VRAM utilization",
        )
        ax.scatter(
            [len(gpu_vram_utilization_history) - 1],
            [gpu_vram_utilization_history[-1]],
            color=_PRIMARY_MARKER,
            s=40,
            zorder=3,
        )
        ax.axhline(
            y=np.mean(valid_vram_values),
            color=_REFERENCE_COLOR,
            linewidth=1.6,
            linestyle="--",
            label="Average",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No VRAM usage samples recorded",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color=_TEXT_COLOR,
        )

    _set_panel_title(ax, "VRAM Utilization")
    ax.set_xlabel("Simulation Step (lifting phase)", fontsize=11)
    ax.set_ylabel("VRAM Utilization (%)", fontsize=12)
    ax.set_xlim(0, max(len(gpu_vram_utilization_history), 1) - 1)
    ax.set_ylim(*_percentage_axis_limits(gpu_vram_utilization_history))
    if valid_vram_values:
        _style_legend(ax, loc="lower right")
        _add_stat_block(
            ax,
            (
                f"Peak VRAM: {np.max(valid_vram_values):.1f}%\n"
                f"Average VRAM: {np.mean(valid_vram_values):.1f}%"
            ),
        )


def plot_sensor_comparison(
    ax: Any,
    *,
    real_sensor_history: list[np.ndarray],
    mean_particle_sensor_history: list[np.ndarray],
) -> None:
    if not real_sensor_history or not mean_particle_sensor_history:
        ax.text(
            0.5,
            0.5,
            "No sensor comparison samples recorded",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color=_TEXT_COLOR,
        )
        return

    real_values = np.asarray(real_sensor_history, dtype=np.float64)
    particle_values = np.asarray(mean_particle_sensor_history, dtype=np.float64)
    steps = range(min(len(real_values), len(particle_values)))
    axis_colors = ["#1d4ed8", "#dc2626", "#16a34a"]
    axis_labels = ["Sensor X", "Sensor Y", "Sensor Z"]

    for sensor_idx, (color, label) in enumerate(zip(axis_colors, axis_labels)):
        ax.plot(
            steps,
            real_values[: len(steps), sensor_idx],
            color=color,
            linewidth=2.4,
            label=f"Real {label}",
        )
        ax.plot(
            steps,
            particle_values[: len(steps), sensor_idx],
            color=color,
            linewidth=1.8,
            linestyle="--",
            alpha=0.9,
            label=f"Particles {label}",
        )

    _set_panel_title(ax, "Sensor Comparison")
    ax.set_xlabel("Simulation Step (lifting phase)", fontsize=11)
    ax.set_ylabel("Force Reading", fontsize=12)
    ax.set_xlim(0, max(len(steps), 1) - 1)
    _style_legend(ax, loc="upper right", ncol=2, fontsize=8.8)
    mean_abs_gap = np.mean(
        np.abs(real_values[: len(steps)] - particle_values[: len(steps)]), axis=0
    )
    _add_stat_block(
        ax,
        (
            f"Mean |gap| X/Y/Z: {mean_abs_gap[0]:.3f}, "
            f"{mean_abs_gap[1]:.3f}, {mean_abs_gap[2]:.3f}"
        ),
    )


def build_sensor_comparison_panels(
    axes: Any,
    *,
    real_sensor_history: list[np.ndarray],
    mean_particle_sensor_history: list[np.ndarray],
) -> None:
    axes_array = np.atleast_1d(axes).reshape(-1)
    for ax in axes_array:
        _style_axis(ax)

    if not real_sensor_history or not mean_particle_sensor_history:
        for ax in axes_array:
            ax.text(
                0.5,
                0.5,
                "No sensor comparison samples recorded",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color=_TEXT_COLOR,
            )
        return

    real_values = np.asarray(real_sensor_history, dtype=np.float64)
    particle_values = np.asarray(mean_particle_sensor_history, dtype=np.float64)
    step_count = min(len(real_values), len(particle_values))
    if step_count == 0:
        for ax in axes_array:
            ax.set_visible(False)
        return
    real_values = real_values[:step_count]
    particle_values = particle_values[:step_count]
    residuals = real_values - particle_values
    steps = np.arange(step_count)
    axis_specs = [
        ("Sensor X", "#1d4ed8"),
        ("Sensor Y", "#dc2626"),
        ("Sensor Z", "#16a34a"),
    ]

    for axis_index, (label, color) in enumerate(axis_specs):
        ax = axes_array[axis_index]
        ax.plot(
            steps,
            real_values[:, axis_index],
            color=color,
            linewidth=2.6,
            label=f"Real {label}",
        )
        ax.plot(
            steps,
            particle_values[:, axis_index],
            color="#111827",
            linewidth=1.8,
            linestyle="--",
            alpha=0.9,
            label=f"Particle mean {label}",
        )
        axis_gap = np.abs(residuals[:, axis_index])
        ax.fill_between(
            steps,
            real_values[:, axis_index],
            particle_values[:, axis_index],
            color=color,
            alpha=0.12,
        )
        ax.set_ylabel(f"{label}\nForce", fontsize=11)
        _set_panel_title(ax, label)
        _style_legend(ax, loc="upper right", fontsize=8.6)
        _add_stat_block(ax, f"Mean |gap|: {np.mean(axis_gap):.3f}", x=0.01, y=0.04)

    residual_ax = axes_array[3]
    residual_colors = ["#1d4ed8", "#dc2626", "#16a34a"]
    residual_labels = ["X residual", "Y residual", "Z residual"]
    for axis_index, (color, label) in enumerate(zip(residual_colors, residual_labels)):
        residual_ax.plot(
            steps,
            residuals[:, axis_index],
            color=color,
            linewidth=2.0,
            label=label,
        )
    residual_ax.axhline(0.0, color="#111827", linewidth=1.2, linestyle="--", alpha=0.8)
    _set_panel_title(residual_ax, "Residuals")
    residual_ax.set_xlabel("Simulation Step (lifting phase)", fontsize=11)
    residual_ax.set_ylabel("Residual\nReal - Mean", fontsize=11)
    _style_legend(residual_ax, loc="upper right", ncol=3, fontsize=8.6)
    _add_stat_block(
        residual_ax,
        "Residuals near zero indicate the particle mean tracks the real sensor.",
        x=0.01,
        y=0.04,
    )


def generate_sensor_comparison_plot(
    *,
    real_sensor_history: list[np.ndarray],
    mean_particle_sensor_history: list[np.ndarray],
    backend: str,
    num_particles: int,
    run_id: str,
) -> Path:
    return _save_multi_axes_plot(
        run_id=run_id,
        output_name="sensor_comparison",
        title="Force Sensor Comparison",
        subtitle=_plot_subtitle(
            backend,
            num_particles,
            extra="three axes with residual summary",
        ),
        figsize=(13, 10.5),
        nrows=4,
        ncols=1,
        plot_builder=lambda axes: build_sensor_comparison_panels(
            axes,
            real_sensor_history=real_sensor_history,
            mean_particle_sensor_history=mean_particle_sensor_history,
        ),
    )


def plot_steps_per_second(
    ax: Any,
    *,
    pf_wall_durations: list[float],
    num_particles: int,
) -> None:
    durations = np.asarray(pf_wall_durations, dtype=np.float64)
    valid_mask = np.isfinite(durations) & (durations > 0.0)
    steps_per_second = np.zeros_like(durations)
    steps_per_second[valid_mask] = 1.0 / durations[valid_mask]
    valid_sps = steps_per_second[valid_mask]

    if valid_sps.size > 0:
        steps = np.arange(len(steps_per_second))
        ax.fill_between(steps, steps_per_second, 0, color="#93c5fd", alpha=0.22)
        ax.plot(
            steps,
            steps_per_second,
            color=_PRIMARY_COLOR,
            linewidth=2.5,
            label="PF update steps/s",
        )
        ax.scatter(
            [len(steps_per_second) - 1],
            [steps_per_second[-1]],
            color=_PRIMARY_MARKER,
            s=40,
            zorder=3,
        )
        ax.axhline(
            y=float(np.mean(valid_sps)),
            color=_REFERENCE_COLOR,
            linewidth=1.6,
            linestyle="--",
            label="Average",
        )
        _style_legend(ax, loc="upper right")
        _add_stat_block(
            ax,
            (
                f"Particles: {num_particles}\n"
                f"Average: {np.mean(valid_sps):.2f} steps/s\n"
                f"Peak: {np.max(valid_sps):.2f} steps/s"
            ),
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No per-step runtime samples recorded",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color=_TEXT_COLOR,
        )

    _set_panel_title(ax, "Particle Filter Throughput")
    ax.set_xlabel("Simulation Step (lifting phase)", fontsize=11)
    ax.set_ylabel("Steps / second", fontsize=12)
    ax.set_xlim(0, max(len(pf_wall_durations), 1) - 1)


def generate_update_duration_per_step_plot(
    *,
    history_estimates: list[float],
    true_mass: float,
    backend: str,
    num_particles: int,
    run_id: str,
) -> Path:
    return build_plot_grid(
        run_id=run_id,
        output_name="relative_error",
        title="Relative Error Over Time",
        subtitle=_plot_subtitle(backend, num_particles),
        figsize=(12, 4.5),
        nrows=1,
        ncols=1,
        plot_builders=[
            lambda ax: plot_relative_error(
                ax,
                history_estimates=history_estimates,
                true_mass=true_mass,
            )
        ],
    )


def generate_particle_filter_overview_plot(
    *,
    history_estimates: list[float],
    ess_history: list[float],
    resample_events: list[bool],
    gpu_utilization_history: list[float],
    gpu_vram_utilization_history: list[float],
    real_sensor_history: list[np.ndarray],
    mean_particle_sensor_history: list[np.ndarray],
    initial_particles: np.ndarray,
    particle_history: list[np.ndarray],
    true_mass: float,
    env: Any,
    backend: str,
    num_particles: int,
    run_id: str,
) -> Path:
    return build_plot_grid(
        run_id=run_id,
        output_name="particle_filter_overview",
        title="Particle Filter Overview",
        subtitle=_plot_subtitle(backend, num_particles),
        figsize=(18, 19),
        nrows=4,
        ncols=2,
        plot_builders=[
            lambda ax: plot_mass_estimation_evolution(
                ax,
                history_estimates=history_estimates,
                initial_particles=initial_particles,
                particle_history=particle_history,
                true_mass=true_mass,
                env=env,
            ),
            lambda ax: plot_effective_sample_size(
                ax,
                ess_history=ess_history,
                num_particles=num_particles,
            ),
            lambda ax: plot_relative_error(
                ax,
                history_estimates=history_estimates,
                true_mass=true_mass,
            ),
            lambda ax: plot_resample_events_timeline(
                ax,
                resample_events=resample_events,
            ),
            lambda ax: plot_gpu_usage(
                ax,
                gpu_utilization_history=gpu_utilization_history,
            ),
            lambda ax: plot_vram_usage(
                ax,
                gpu_vram_utilization_history=gpu_vram_utilization_history,
            ),
            lambda ax: plot_sensor_comparison(
                ax,
                real_sensor_history=real_sensor_history,
                mean_particle_sensor_history=mean_particle_sensor_history,
            ),
        ],
    )


def generate_particle_filter_plots(
    *,
    history_estimates: list[float],
    ess_history: list[float],
    resample_events: list[bool],
    gpu_utilization_history: list[float],
    gpu_vram_utilization_history: list[float],
    real_sensor_history: list[np.ndarray],
    mean_particle_sensor_history: list[np.ndarray],
    initial_particles: np.ndarray,
    particle_history: list[np.ndarray],
    pf_wall_durations: list[float],
    true_mass: float,
    env: Any,
    backend: str,
    num_particles: int,
    run_id: str,
) -> dict[str, Path]:
    subtitle = _plot_subtitle(backend, num_particles)
    return {
        "mass_estimation": _save_single_plot(
            run_id=run_id,
            output_name="mass_estimation_evolution",
            title="Mass Estimate Over Time",
            subtitle=subtitle,
            figsize=(12, 4.8),
            plot_builder=lambda ax: plot_mass_estimation_evolution(
                ax,
                history_estimates=history_estimates,
                initial_particles=initial_particles,
                particle_history=particle_history,
                true_mass=true_mass,
                env=env,
            ),
        ),
        "effective_sample_size": _save_single_plot(
            run_id=run_id,
            output_name="effective_sample_size",
            title="Effective Sample Size Over Time",
            subtitle=subtitle,
            figsize=(12, 4.8),
            plot_builder=lambda ax: plot_effective_sample_size(
                ax,
                ess_history=ess_history,
                num_particles=num_particles,
            ),
        ),
        "relative_error": _save_single_plot(
            run_id=run_id,
            output_name="relative_error",
            title="Relative Error Over Time",
            subtitle=subtitle,
            figsize=(12, 4.8),
            plot_builder=lambda ax: plot_relative_error(
                ax,
                history_estimates=history_estimates,
                true_mass=true_mass,
            ),
        ),
        "resample_events": _save_single_plot(
            run_id=run_id,
            output_name="resample_events_timeline",
            title="Resampling Timeline",
            subtitle=subtitle,
            figsize=(12, 4.8),
            plot_builder=lambda ax: plot_resample_events_timeline(
                ax,
                resample_events=resample_events,
            ),
        ),
        "gpu_usage": _save_single_plot(
            run_id=run_id,
            output_name="gpu_usage",
            title="GPU Utilization Over Time",
            subtitle=subtitle,
            figsize=(12, 4.8),
            plot_builder=lambda ax: plot_gpu_usage(
                ax,
                gpu_utilization_history=gpu_utilization_history,
            ),
        ),
        "vram_usage": _save_single_plot(
            run_id=run_id,
            output_name="vram_usage",
            title="GPU Memory Utilization Over Time",
            subtitle=subtitle,
            figsize=(12, 4.8),
            plot_builder=lambda ax: plot_vram_usage(
                ax,
                gpu_vram_utilization_history=gpu_vram_utilization_history,
            ),
        ),
        "steps_per_second": _save_single_plot(
            run_id=run_id,
            output_name="steps_per_second",
            title="Particle Filter Throughput",
            subtitle=_plot_subtitle(
                backend,
                num_particles,
                extra="measured from per-step wall time",
            ),
            figsize=(12, 4.8),
            plot_builder=lambda ax: plot_steps_per_second(
                ax,
                pf_wall_durations=pf_wall_durations,
                num_particles=num_particles,
            ),
        ),
        "sensor_comparison": generate_sensor_comparison_plot(
            run_id=run_id,
            backend=backend,
            num_particles=num_particles,
            real_sensor_history=real_sensor_history,
            mean_particle_sensor_history=mean_particle_sensor_history,
        ),
    }
