"""Verify six-vertex inequalities in exact integer arithmetic."""

from itertools import permutations
from math import lcm
from pathlib import Path
import json

from verify import check_positive_definite
from verify_gain6 import denominator, flag_data, pairs

M = 10**8
NORM = 720
TRIANGLE_WEIGHTS = ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0),
                    (0, 1, 0), (1, 1, 0), (0, 0, 1))
DEGREE_WEIGHTS = ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0),
                  (0, 1, 0), (1, 1, 0), (2, 1, 0), (0, 2, 0),
                  (0, 0, 1), (1, 0, 1))
ENDPOINTS = {42_000_000: 18_711_000, 48_000_000: 20_835_000,
             60_000_000: 25_144_000}


def increment(power, i):
    return tuple(x + (i == j) for j, x in enumerate(power))


def monomial_edges(power):
    edges = []
    j = 1
    for size, count in enumerate(power, start=1):
        for _ in range(count):
            edges.extend((0, k) for k in range(j, j + size))
            edges.extend((j + u, j + v) for u, v in pairs(size))
            j += size
    assert j <= 6
    return edges


def count_monomials(masks, powers):
    """For each induced graph of Q, sum the H-product over all 6! bijections."""
    edges = pairs(6)
    orders = tuple(permutations(range(6)))
    as_mask = {power: [] for power in powers}
    edge_index = {e: i for i, e in enumerate(edges)}
    for power in powers:
        product_edges = monomial_edges(power)
        for order in orders:
            required = sum(1 << edge_index[tuple(sorted((order[i], order[j])))]
                           for i, j in product_edges)
            as_mask[power].append(required)
    return {power: [sum(not (mask & required) for required in forms) for mask in masks]
            for power, forms in as_mask.items()}


def verify_moments(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    masks, objective, products, matrices, specs = flag_data()
    assert len(masks) == 156 and len(products[0]) == len(products[1]) == 11
    powers = (set(TRIANGLE_WEIGHTS) | set(DEGREE_WEIGHTS)
              | {increment(p, 1) for p in TRIANGLE_WEIGHTS}
              | {increment(p, 0) for p in DEGREE_WEIGHTS})
    counts = count_monomials(masks, powers)

    for number, row in enumerate(rows, 1):
        lo, hi, U, R, beta, alpha = (row[k] for k in ("lo", "hi", "U", "R", "slope", "intercept"))
        assert all(type(v) is int for v in (lo, hi, U, R, beta, alpha))
        assert 25_000_000 <= lo < hi <= M and hi in ENDPOINTS
        assert 0 <= U <= M and 0 <= R <= M and beta >= 0
        assert len(row["tri"]) == len(TRIANGLE_WEIGHTS)
        assert len(row["deg"]) == len(DEGREE_WEIGHTS)
        assert all(type(x) is int and x >= 0 for x in row["tri"] + row["deg"] + row["linear"])
        assert len(row["linear"]) == 22
        scale = row["scale"]
        assert type(scale) is int and scale >= 1
        assert row.get("penalty", 0) == 0
        assert len(row["psd"]) == len(specs)
        for block, (_, _, dim) in zip(row["psd"], specs, strict=True):
            assert len(block) == dim
            check_positive_definite(block)
        common = lcm(denominator, scale * M, M * M, 15 * M)
        terms = [(weight, counts[p], counts[increment(p, i)], anchor)
                 for family, weights, i, anchor in (
                     (TRIANGLE_WEIGHTS, row["tri"], 1, U), (DEGREE_WEIGHTS, row["deg"], 0, R))
                 for weight, p in zip(weights, family, strict=True) if weight]
        for j, mask in enumerate(masks):
            residual = objective[j] * (common // denominator) - (
                15 * alpha + beta * mask.bit_count()) * (NORM * common // (15 * M))
            residual += sum(weight * (M * numerator[j] - anchor * base[j])
                            for weight, base, numerator, anchor in terms) * (common // (M * M))
            for k, (p0, p1) in enumerate(zip(*products, strict=True)):
                residual -= (row["linear"][k] * (M * p1[j] - lo * p0[j])
                             + row["linear"][k + 11] * (hi * p0[j] - M * p1[j])) * (common // (scale * M))
            for block, matrix in zip(row["psd"], matrices, strict=True):
                residual -= sum(x * y for xs, ys in zip(block, matrix[j], strict=True)
                                for x, y in zip(xs, ys, strict=True) if y) * (common // scale)
            assert residual >= 0, (number, mask, residual)
    print(f"Verified {len(rows)} rooted inequalities on all {len(masks)} graphs.", flush=True)
    return rows
