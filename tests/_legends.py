"""Finds legend keys that draw the same symbol for different labels."""

from matplotlib.figure import Figure


def marker_clashes(fig: Figure) -> list[tuple[str, str]]:
    """Lists pairs of legend labels whose keys share a marker, face and edge.

    Args:
        fig: The figure; every legend of the figure and of its axes is read.

    Returns:
        The label pairs, empty when no two keys look the same.
    """
    legends = list(fig.legends)
    for sub in fig.subfigs:
        legends.extend(sub.legends)
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            legends.append(legend)
    seen: dict[tuple[str, str, str], str] = {}
    clashes = []
    for legend in legends:
        for handle, text in zip(
            legend.legend_handles, legend.get_texts(), strict=True
        ):
            if not hasattr(handle, "get_marker"):
                continue
            marker = str(handle.get_marker())
            if marker in ("None", ""):
                continue
            key = (
                marker,
                str(handle.get_markerfacecolor()),
                str(handle.get_markeredgecolor()),
            )
            if key in seen:
                clashes.append((seen[key], text.get_text()))
            seen[key] = text.get_text()
    return clashes
