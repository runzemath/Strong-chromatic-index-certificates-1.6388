"""Verify the certificate and the numerical inequalities using integers and rationals."""

from fractions import Fraction
from itertools import combinations, permutations
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
SCALE = 10**12
DENOM = 15_000
ROUNDING = Fraction(1, SCALE)


def check_positive_definite(block):
    """The pivots in fraction-free elimination are leading principal minors."""
    n = len(block)
    assert all(len(row) == n for row in block)
    assert all(type(x) is int for row in block for x in row)
    assert all(block[i][j] == block[j][i] for i in range(n) for j in range(i))
    matrix = [row[:] for row in block]
    previous = 1
    for k in range(n):
        pivot = matrix[k][k]
        assert pivot > 0
        for i in range(k + 1, n):
            for j in range(i, n):
                numerator = pivot * matrix[i][j] - matrix[k][i] * matrix[k][j]
                matrix[i][j], remainder = divmod(numerator, previous)
                assert remainder == 0
        previous = pivot


def flag_counts():
    """Count the objective and minimum-degree terms on every size-5 flag."""
    vertices = range(5)
    pairs = list(permutations(vertices, 2))
    flag_edges = [(i, j) for j in vertices for i in range(j)]
    rows = []
    for line in (ROOT / 'flags.txt').read_text(encoding="utf-8").splitlines():
        if line.startswith('#'):
            continue
        index, order, edge_string, colors_string = line.split(';')
        assert int(index) == len(rows) and order == '5'
        edges = [[0] * 5 for _ in vertices]
        for (i, j), value in zip(flag_edges, map(int, edge_string.split(',')), strict=True):
            edges[i][j] = edges[j][i] = value
        colors = list(map(int, colors_string.split(',')))
        objective = 0
        strong_degree = 0
        extension = 0
        for x, y in pairs:
            if colors[y] != 0 or edges[x][y] != 1:
                continue
            cross = colors[x] == 1
            other = [w for w in vertices if w != x and w != y]
            if cross and all(edges[x][w] for w in other):
                extension += 6
            for u, v in combinations(other, 2):
                w = next(i for i in other if i != u and i != v)
                if not edges[x][w] or edges[u][v] != 1:
                    continue
                if not any(edges[i][j] for i in (x, y) for j in (u, v)):
                    continue
                if cross:
                    strong_degree += 1
                if colors[u] == 0 or colors[v] == 0:
                    objective += 2 if cross else 1
        min_row = 10_000 * strong_degree - 17_297 * extension
        rows.append((objective, min_row))
    return rows


def verify_sdp():
    """Check the size-5 residual coefficientwise with its exact denominators."""
    counts = flag_counts()
    witness = json.loads((ROOT / 'witness.json').read_text(encoding="utf-8"))
    diagonal = iter(witness['diagonal'])
    positive = witness['positive']

    with (ROOT / 'flags.sdpa').open(encoding="utf-8") as source:
        lines = (line.split() for line in source if line.strip() and not line.startswith('*'))
        number_flags = int(next(lines)[0])
        number_blocks = int(next(lines)[0])
        sizes = list(map(int, next(lines)))
        objectives = next(lines)
        assert number_flags == len(counts) == len(objectives) == 17_950
        assert number_blocks == len(sizes) and sizes[:2] == [-number_flags, -1]
        blocks = []
        for index, size in enumerate(sizes, start=1):
            if size < 0:
                block = [next(diagonal) for _ in range(-size)]
                assert all(type(x) is int and x>=0 for x in block)
            else:
                block = positive[str(index)]
                assert len(block) == size
                check_positive_definite(block)
            blocks.append(block)
        assert next(diagonal, None) is None
        assert len(positive) == 39

        for printed, (coefficient, _) in zip(objectives, counts, strict=True):
            assert abs(Fraction(printed) + Fraction(coefficient, 60)) < ROUNDING
        residual = [value * (DENOM // 60) * SCALE for value, _ in counts]
        degree_coefficients = [0] * number_flags
        constant = 0
        for flag, block, i, j, coefficient in lines:
            flag, block, i, j = map(int, (flag, block, i, j))
            block_index, row, column = block - 1, i - 1, j - 1
            if sizes[block_index] < 0:
                assert row == column
                multiplier = blocks[block_index][row]
            else:
                multiplier = blocks[block_index][row][column]
                if row != column:
                    multiplier *= 2
            if block == 2:
                value = Fraction(coefficient)
                numerator = round(5000 * value)
                assert abs(value - Fraction(numerator, 5000)) < ROUNDING
                assert flag > 0
                degree_coefficients[flag - 1] += numerator
                term = numerator * (DENOM // 5000) * multiplier
            else:
                term = int(coefficient) * DENOM * multiplier
            if flag:
                residual[flag - 1] += term
            else:
                constant -= term
        assert degree_coefficients == [value for _, value in counts]

    residual_bound = Fraction(max(0, max(residual)), DENOM * SCALE)
    constant_bound = Fraction(constant, DENOM * SCALE)
    assert Fraction(blocks[1][0], SCALE) == Fraction('0.030913355784')
    assert constant_bound + 4_802 * residual_bound < Fraction('10.6434')
    print(f'{len(positive)} positive-definite blocks verified')
    print(f'Constant term = {constant_bound}')
    print(f'Maximum residual coefficient = {residual_bound}')



if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Run without Python's -O option.")
    verify_sdp()
