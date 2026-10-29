"""Named phase timers and warm-up-aware estimates for training runs."""

from statistics import median

from codetiming import Timer


def phase_timer(name: str) -> Timer:
    return Timer(name=name, logger=None)


def estimate_runtime(epoch_seconds: list[float], epochs: int, elapsed_seconds: float):
    """Use the median of up to five recent epochs, excluding the first warm-up."""
    if not epoch_seconds:
        return {}
    recent = (epoch_seconds[1:] or epoch_seconds)[-5:]
    remaining = max(0, epochs - len(epoch_seconds))
    typical = median(recent)
    return {
        "steady_epoch_seconds": typical,
        "estimated_remaining_seconds": typical * remaining,
        "estimated_total_seconds": elapsed_seconds + typical * remaining,
        "estimated_remaining_low_seconds": min(recent) * remaining,
        "estimated_remaining_high_seconds": max(recent) * remaining,
        "estimate_epochs": len(recent),
    }
