"""Verify a root-degree density witness using exact arithmetic."""

from collections import defaultdict
from fractions import Fraction as Q
from itertools import combinations, permutations
from math import lcm
from pathlib import Path
import argparse
import json

from verify import check_positive_definite, flag_counts
from local_constraints import coefficients as extra_coefficients

ROOT = Path(__file__).resolve().parent
SIZE = 5
COEFFICIENT = Q("0.030913355784")
VERTICES = tuple(range(SIZE))
ORDERS = tuple(permutations(VERTICES, 2))
PAIRS = tuple((i, j) for j in VERTICES for i in range(j))
TRIPLES = tuple(combinations(VERTICES, 3))


def flags(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        index, size, edges, colors = line.split(";")
        assert int(size) == SIZE
        matrix = [[0] * SIZE for _ in VERTICES]
        for (x, y), entry in zip(PAIRS, map(int, edges.split(",")), strict=True):
            matrix[x][y] = matrix[y][x] = entry
        colors = tuple(map(int, colors.split(",")))
        allowed = not any(colors[x] == colors[y] == 1 and matrix[x][y] == 2
                          for x, y in PAIRS)
        yield int(index), colors, matrix, allowed


def local(vertices, colors, matrix):
    reached = {v for v in vertices if colors[v] == 0}
    for _ in vertices:
        reached |= {v for v in vertices if any(matrix[v][w] for w in reached)}
    return len(reached) == len(vertices)


def mass_numerator(colors, matrix):
    return sum(
        (2 if colors[x] == 1 else 1)
        for x, y in ORDERS
        if matrix[x][y] == 1 and colors[y] == 0
        and all(matrix[x][z] for z in VERTICES if z != x)
    )


def strong_counts(colors, matrix):
    """Rows conditioned on an incident marked edge and a third vertex."""
    counts = {}
    for x, y in ORDERS:
        if matrix[x][y] != 1 or colors[y] != 0:
            continue
        other = [w for w in VERTICES if w not in (x, y)]
        extension = 2 * int(all(matrix[x][w] for w in other))
        for u, v in combinations(other, 2):
            w = next(i for i in other if i not in (u, v))
            e, f = matrix[x][w], matrix[y][w]
            if colors[w] == 1 and not (e or f):
                continue
            if colors[x] == colors[y]:
                e, f = sorted((e, f))
            key = (colors[x], colors[y], colors[w], e, f)
            strong = matrix[u][v] == 1 and any(
                matrix[i][j] for i in (x, y) for j in (u, v)
            )
            p, q = counts.get(key, (0, 0))
            counts[key] = p + 2 * strong, q + extension
    return counts


def product_counts(colors, matrix):
    """Product of marked-edge mass and a local three-vertex flag."""
    counts = {}
    for vertices in TRIPLES:
        u, v = [i for i in VERTICES if i not in vertices]
        marked = matrix[u][v] == 1 and (colors[u] == 0 or colors[v] == 0)
        extension = sum(bool(matrix[w][u] and matrix[w][v]) for w in vertices)
        if (not (marked or extension)) or not local(vertices, colors, matrix):
            continue
        key = min(
            (colors[x], colors[y], colors[z], matrix[x][y], matrix[x][z], matrix[y][z])
            for x, y, z in permutations(vertices)
        )
        p, q = counts.get(key, (0, 0))
        counts[key] = p + marked, q + extension
    return counts


def common_counts(colors, matrix):
    """Rows for the common-neighbor bound, conditioned on four vertices."""
    counts = {}
    for x, y in ORDERS:
        if colors[y] != 0 or matrix[x][y] != 1:
            continue
        for w in VERTICES:
            if w in (x, y) or not matrix[x][w]:
                continue
            a, b = [i for i in VERTICES if i not in (x, y, w)]
            if not local((x, y, a, b), colors, matrix):
                continue
            u = (colors[a], matrix[x][a], matrix[y][a])
            v = (colors[b], matrix[x][b], matrix[y][b])
            key = (colors[x], colors[y], *min(u, v), *max(u, v), matrix[a][b])
            p, q = counts.get(key, (0, 0))
            counts[key] = p + 1, q + bool(matrix[y][w])
    return counts


def tree_count(colors, graph):
    n=len(colors)
    matrix=[[-int(bool(graph[i][j])) for j in range(n)] for i in range(n)]
    for i in range(n):
        matrix[i][i]=sum(bool(x) for x in graph[i])+int(colors[i]==0)
    previous=1
    for k in range(n-1):
        pivot=matrix[k][k]
        assert pivot>0
        for i in range(k+1,n):
            for j in range(k+1,n):
                value=pivot*matrix[i][j]-matrix[i][k]*matrix[k][j]
                matrix[i][j], remainder=divmod(value,previous)
                assert remainder==0
        previous=pivot
    assert matrix[-1][-1]>0
    return matrix[-1][-1]


def verify_mass(path):
    witness = json.loads(path.read_text(encoding="utf-8"))
    tau = Q(witness["threshold"])
    gamma = Q(witness["slope"])
    signed = Q(witness.get("signed", str(COEFFICIENT)))
    assert Q(witness.get("root_lower", witness["threshold"])) == tau
    assert Q(witness.get("root_upper", "1")) == 1
    assert not witness.get("all_edges", False)
    claimed = Q(witness["intercept"])
    scale = witness["scale"]
    additional = witness.get('additional_rows', False)
    assert not witness.get('unsigned',False)
    assert 0 < tau < 1 and gamma >= 0 and signed >= 0 and type(scale) is int and scale > 0
    assert all(type(x) is int and x>=0 for x in witness['linear'])

    all_flags = tuple(flags(ROOT / "flags.txt"))
    allowed = {index for index, _, _, keep in all_flags if keep}
    assert len(all_flags) == 17950 and len(allowed) == 9530
    objective = flag_counts()

    degree_rows = {}
    mass_rows = {}
    common_rows = {}
    additional_rows = {}
    local_data = {}
    for index, colors, matrix, keep in all_flags:
        degree = strong_counts(colors, matrix)
        for key in degree:
            degree_rows.setdefault(key, len(degree_rows))
        if not keep:
            continue
        product = product_counts(colors, matrix)
        for key in product:
            mass_rows.setdefault(key, len(mass_rows))
        common = common_counts(colors, matrix)
        for key in common:
            common_rows.setdefault(key, len(common_rows))
        extra = extra_coefficients(colors, matrix) if additional else {}
        for key in extra:
            additional_rows.setdefault(key, len(additional_rows))
        local_data[index] = mass_numerator(colors, matrix), degree, product, common, extra

    scale_den = lcm(60, 5000, 80 * gamma.denominator, 30 * tau.denominator,
                    40, tau.denominator)
    if (scale_den * scale) % (5000 * signed.denominator):
        scale_den *= signed.denominator
    den = scale * scale_den

    entries = iter(witness["linear"])
    positivity = witness["psd"]
    lhs = {index: 0 for index in allowed}
    constant = 0

    with (ROOT / "flags.sdpa").open(encoding="utf-8") as file:
        source = (line.split() for line in file if line.strip() and not line.startswith("*"))
        count = int(next(source)[0])
        blocks_count = int(next(source)[0])
        dimensions = list(map(int, next(source)))
        printed = next(source)
        assert count == len(all_flags) == len(printed)
        assert blocks_count == len(dimensions) and dimensions[:2] == [-count, -1]
        assert len(positivity) == sum(d > 0 for d in dimensions)
        blocks = [None, None]
        positive_blocks = iter(positivity)
        for dimension in dimensions[2:]:
            if dimension < 0:
                block = [next(entries) for _ in range(-dimension)]
                assert min(block) >= 0
            else:
                block = next(positive_blocks)
                assert len(block) == dimension
                check_positive_definite(block)
            blocks.append(block)
        assert next(positive_blocks, None) is None

        degree_multiplier = [next(entries) for _ in degree_rows]
        lo, hi = next(entries), next(entries)
        mass_multiplier = [(next(entries), next(entries)) for _ in mass_rows]
        common_multiplier = [next(entries) for _ in common_rows]
        additional_multiplier = [next(entries) for _ in additional_rows]
        assert next(entries, None) is None
        assert min(degree_multiplier) >= 0 and min(common_multiplier) >= 0
        assert min(x for pair in mass_multiplier for x in pair) >= 0 and min(lo, hi) >= 0
        assert all(x >= 0 for x in additional_multiplier)

        constant += int(scale_den * (hi - tau * lo))
        obj60 = den // 60
        degree_scale = den * signed / 5000
        assert degree_scale.denominator == 1
        degree_scale = degree_scale.numerator
        for index, (mass, degree, products, common, extra) in local_data.items():
            obj, archived_degree = objective[index]
            total = obj60 * obj + degree_scale * archived_degree
            value = scale_den * (hi - lo + 16 * scale * gamma) * Q(mass, 80)
            assert value.denominator == 1
            total -= value.numerator
            for key, (p, q) in degree.items():
                multiplier = degree_multiplier[degree_rows[key]]
                total += multiplier * (scale_den * p - int(scale_den * 4 * tau) * q)
            for key, (p, q) in products.items():
                low, high = mass_multiplier[mass_rows[key]]
                total += low * (scale_den // 40 * p - int(scale_den * tau / 30) * q)
                total += high * (scale_den // 30 * q - scale_den // 40 * p)
            for key, (p, q) in common.items():
                multiplier = common_multiplier[common_rows[key]]
                total += multiplier * (int(scale_den * (2 - 2 * tau)) * p - scale_den * q)
            for key,(p,q) in extra.items():
                coefficient=scale_den*p-int(scale_den*tau)*q
                total+=additional_multiplier[additional_rows[key]]*coefficient
            lhs[index] = total

        for flag, block, i, j, coefficient in source:
            flag, block, i, j = map(int, (flag, block, i, j))
            if block <= 2 or (flag and flag - 1 not in allowed):
                continue
            data = blocks[block - 1]
            if dimensions[block - 1] < 0:
                assert i == j
                multiplier = data[i-1]
            else:
                multiplier = data[i-1][j-1] * (2 if i != j else 1)
            term = int(coefficient) * multiplier * scale_den
            if flag:
                lhs[flag-1] += term
            else:
                constant -= term

    if witness.get('tree_residual', False):
        trees = {i: tree_count(colors, graph)
                 for i, colors, graph, keep in all_flags if keep}
        residual = max(Q(max(0, value), den*trees[i]) for i, value in lhs.items())
    else:
        residual = Q(max(0, max(lhs.values())), den)
    bound = gamma + (Q(constant, den) + 4802 * residual) / 16
    assert bound < claimed, (path.name, bound, claimed)
    print(f"Density certificate verified: {path.name}", flush=True)
    return tau, gamma, claimed


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Run without Python's -O option.")
    parser = argparse.ArgumentParser()
    parser.add_argument("witness", type=Path, help="path to a density witness JSON file")
    verify_mass(parser.parse_args().witness)
