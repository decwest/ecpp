from __future__ import annotations

import math


def logistic(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def gate_abs(z_abs: float, z_on: float, z_off: float, endpoint_value: float = 0.01) -> float:
    if z_on < 0.0:
        raise ValueError("z_on must be non-negative")
    if z_off <= z_on:
        raise ValueError("z_off must be greater than z_on")
    if not 0.0 < endpoint_value < 0.5:
        raise ValueError("endpoint_value must be between 0 and 0.5")
    center = 0.5 * (z_on + z_off)
    slope = 2.0 * math.log((1.0 - endpoint_value) / endpoint_value) / (z_off - z_on)
    return logistic(-slope * (z_abs - center))


def gate_abs_by_mode(
    z_abs: float,
    z_on: float,
    z_off: float,
    endpoint_value: float,
    mode: str,
) -> float:
    if mode == "always_on":
        return 1.0
    if mode == "off":
        return 0.0
    if mode != "sigmoid":
        raise ValueError("gate mode must be sigmoid, always_on, or off")
    return gate_abs(z_abs, z_on, z_off, endpoint_value)
