"""CIEDE2000 colour difference for the palette tests (no dependency)."""

import math


def lab(color: str) -> tuple[float, float, float]:
    """Converts a hex colour to CIELAB under D65.

    Args:
        color: `#rrggbb`.

    Returns:
        L, a and b.
    """
    rgb = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb
    ]
    x = 0.4124 * lin[0] + 0.3576 * lin[1] + 0.1805 * lin[2]
    y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    z = 0.0193 * lin[0] + 0.1192 * lin[1] + 0.9505 * lin[2]

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116

    fx, fy, fz = f(x / 0.95047), f(y / 1.0), f(z / 1.08883)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def de2000(c1: str, c2: str) -> float:
    """Computes the CIEDE2000 difference of two hex colours.

    Args:
        c1: First colour, `#rrggbb`.
        c2: Second colour.

    Returns:
        The colour difference.
    """
    l1, a1, b1 = lab(c1)
    l2, a2, b2 = lab(c2)
    chroma1, chroma2 = math.hypot(a1, b1), math.hypot(a2, b2)
    mean_c = (chroma1 + chroma2) / 2
    g = 0.5 * (1 - math.sqrt(mean_c**7 / (mean_c**7 + 25**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    d_l, d_c = l2 - l1, c2p - c1p
    dh = h2p - h1p
    if c1p * c2p == 0:
        dh = 0
    elif dh > 180:
        dh -= 360
    elif dh < -180:
        dh += 360
    d_h = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dh / 2))
    mean_l, mean_cp = (l1 + l2) / 2, (c1p + c2p) / 2
    if c1p * c2p == 0:
        mean_h = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        mean_h = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        mean_h = (h1p + h2p + 360) / 2
    else:
        mean_h = (h1p + h2p - 360) / 2
    t = (
        1
        - 0.17 * math.cos(math.radians(mean_h - 30))
        + 0.24 * math.cos(math.radians(2 * mean_h))
        + 0.32 * math.cos(math.radians(3 * mean_h + 6))
        - 0.20 * math.cos(math.radians(4 * mean_h - 63))
    )
    d_theta = 30 * math.exp(-(((mean_h - 275) / 25) ** 2))
    r_c = 2 * math.sqrt(mean_cp**7 / (mean_cp**7 + 25**7))
    s_l = 1 + 0.015 * (mean_l - 50) ** 2 / math.sqrt(20 + (mean_l - 50) ** 2)
    s_c = 1 + 0.045 * mean_cp
    s_h = 1 + 0.015 * mean_cp * t
    r_t = -math.sin(math.radians(2 * d_theta)) * r_c
    return math.sqrt(
        (d_l / s_l) ** 2
        + (d_c / s_c) ** 2
        + (d_h / s_h) ** 2
        + r_t * (d_c / s_c) * (d_h / s_h)
    )
