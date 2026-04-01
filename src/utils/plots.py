import os
from pathlib import Path
import shutil
from typing import Any
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/mesa_shader_cache")

_TITLE_COLOR = "#1f2933"
_TEXT_COLOR = "#52606d"
_GRID_COLOR = "#7b8794"
_PRIMARY_COLOR = "#1d4ed8"
_PRIMARY_FILL = "#93c5fd"
_PRIMARY_MARKER = "#1e3a8a"
_REFERENCE_COLOR = "#ea580c"


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


def _posterior_evolution_output_paths(run_id: str) -> tuple[Path, Path]:
    base_dir = _run_output_dir(run_id) / "particle_posterior"
    images_dir = base_dir / "images"
    _reset_directory(images_dir)
    gif_path = base_dir / "particle_posterior.gif"
    return images_dir, gif_path


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

        fill_color = "#d4d4d8"
        edge_color = "#52525b"
        estimate_color = "#111827"
        true_mass_color = "#6b7280"

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
        figsize=(18, 14),
        nrows=3,
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
        ],
    )


def generate_particle_filter_plots(
    *,
    history_estimates: list[float],
    ess_history: list[float],
    resample_events: list[bool],
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
    }
