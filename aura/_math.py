"""Lightweight numerical helpers that avoid external dependencies."""

from __future__ import annotations

import math
import random
import zlib
from typing import Callable, Iterable, Iterator, List, Sequence, Tuple

Vector = List[float]


def _rng_for(*parts: object) -> random.Random:
    data = "|".join(str(part) for part in parts).encode("utf8")
    seed = zlib.crc32(data) & 0xFFFFFFFF
    return random.Random(seed)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def apply_linear(weights: List[List[float]], bias: List[float], vector: Sequence[float]) -> Vector:
    return [dot(row, vector) + bias_value for row, bias_value in zip(weights, bias)]


class LinearLayer:
    """Deterministic affine transform built on the helper random source."""

    def __init__(self, in_dim: int, out_dim: int, *, seed: object, bias: bool = False) -> None:
        rng = _rng_for("linear", seed, in_dim, out_dim)
        scale = 1.0 / max(1, in_dim)
        self.weights = [
            [rng.uniform(-scale, scale) for _ in range(in_dim)]
            for _ in range(out_dim)
        ]
        if bias:
            self.bias = [rng.uniform(-scale, scale) for _ in range(out_dim)]
        else:
            self.bias = [0.0 for _ in range(out_dim)]

    def __call__(self, vector: Sequence[float]) -> Vector:
        return apply_linear(self.weights, self.bias, vector)

    def copy_from_flat(self, values: Sequence[float]) -> None:
        iterator: Iterator[float] = iter(values)
        for row in self.weights:
            for index in range(len(row)):
                row[index] = float(next(iterator))

    def state(self) -> Tuple[List[List[float]], List[float]]:
        return self.weights, self.bias


class SequentialLayer:
    def __init__(self, *layers: Callable[[Vector], Vector]) -> None:
        self.layers = list(layers)

    def __call__(self, vector: Sequence[float]) -> Vector:
        result = list(vector)
        for layer in self.layers:
            result = layer(result)
        return result


def relu(vector: Sequence[float]) -> Vector:
    return [max(0.0, value) for value in vector]


def gelu(vector: Sequence[float]) -> Vector:
    return [0.5 * value * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (value + 0.044715 * value ** 3))) for value in vector]


def layer_norm(vector: Sequence[float], *, epsilon: float = 1e-5) -> Vector:
    mean = sum(vector) / len(vector) if vector else 0.0
    variance = sum((value - mean) ** 2 for value in vector) / len(vector) if vector else 0.0
    denominator = math.sqrt(variance + epsilon)
    return [(value - mean) / denominator for value in vector]


def softmax(vector: Sequence[float]) -> Vector:
    if not vector:
        return []
    max_value = max(vector)
    exps = [math.exp(value - max_value) for value in vector]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def tanh(vector: Sequence[float]) -> Vector:
    return [math.tanh(value) for value in vector]


def sin(vector: Sequence[float]) -> Vector:
    return [math.sin(value) for value in vector]


def combine(weighted_vectors: Sequence[Tuple[float, Sequence[float]]]) -> Vector:
    if not weighted_vectors:
        return []
    length = len(weighted_vectors[0][1])
    result = [0.0 for _ in range(length)]
    for weight, vector in weighted_vectors:
        for index in range(length):
            result[index] += weight * vector[index]
    return result


def add(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x + y for x, y in zip(a, b)]


def scale(vector: Sequence[float], factor: float) -> Vector:
    return [factor * value for value in vector]


def concat(a: Sequence[float], b: Sequence[float]) -> Vector:
    return list(a) + list(b)


def mean_and_std(vector: Sequence[float]) -> Tuple[float, float]:
    if not vector:
        return 0.0, 0.0
    mean = sum(vector) / len(vector)
    variance = sum((value - mean) ** 2 for value in vector) / len(vector)
    return mean, math.sqrt(variance)
