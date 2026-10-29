"""Named phase timers and warm-up-aware estimates for training runs."""

from statistics import median

from codetiming import Timer


def phase_timer(name: str) -> Timer:
    return Timer(name=name, logger=None)


def estimate_runtime(
    epoch_seconds: list[float | None], epochs: int, elapsed_seconds: float
):
    """Use the median of up to five recent epochs, excluding the first warm-up."""
    valid = [value for value in epoch_seconds if value is not None]
    if not valid:
        return {}
    recent = [value for value in epoch_seconds[1:] if value is not None] or valid
    recent = recent[-5:]
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
