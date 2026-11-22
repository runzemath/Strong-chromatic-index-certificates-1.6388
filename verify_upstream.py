"""Reconstruct the five-vertex basis and the archived flag coefficients."""

from collections import Counter, defaultdict
from itertools import combinations, permutations, product
from math import gcd, prod
from pathlib import Path

from verify_mass import flags, local


ROOT = Path(__file__).resolve().parent
VERTICES = tuple(range(5))
PAIRS = tuple(combinations(VERTICES, 2))
ORDERS = tuple(permutations(VERTICES))
TRIPLES = tuple(combinations(VERTICES, 3))


def encoding(colors, matrix, order):
    return (tuple(colors[v] for v in order)
            + tuple(matrix[u][v] for u, v in combinations(order, 2)))


def representative(colors, matrix, vertices):
    """Return a canonical labeled type and all its labelings on these vertices."""
    best = None
    labelings = []
    for order in permutations(vertices):
        key = encoding(colors, matrix, order)
        if best is None or key < best:
            best, labelings = key, [order]
        elif key == best:
            labelings.append(order)
    return best, labelings


def labeled_local_count():
    """Sum color assignments over underlying simple graphs on five vertices."""
    total = 0
    for present in product((0, 1), repeat=len(PAIRS)):
        components = [{v} for v in VERTICES]
        for (u, v), edge in zip(PAIRS, present, strict=True):
            if edge and components[u] is not components[v]:
                joined = components[u] | components[v]
                for w in joined:
                    components[w] = joined
        black_choices = prod(2**len(group) - 1
                             for i, group in enumerate(components) if min(group) == i)
        total += black_choices * 2**sum(present)
    return total


def primitive(coefficients):
    """Divide a sparse integer vector by the positive gcd of its entries."""
    divisor = gcd(*coefficients.values())
    if not divisor:
        return ()
    return tuple(sorted((i, value // divisor)
                        for i, value in coefficients.items() if value))


def record_square(block, flag, first, second):
    # The stored coefficient is a matrix entry, before the factor two in the trace.
    pair = tuple(sorted((first, second)))
    block[pair][flag] += 2 if first == second else 1


def reconstruct():
    """Return primitive linear rows and unnormalized square blocks."""
    seen = set()
    labeled = restricted = 0
    rows = defaultdict(Counter)
    squares = defaultdict(lambda: defaultdict(Counter))
    normalization = Counter({0: 10})
    for index, colors, matrix, keep in flags(ROOT / "flags.txt"):
        assert index == len(seen)
        assert len(colors) == 5 and all(c in (0, 1) for c in colors)
        assert all(matrix[u][v] in (0, 1, 2) for u, v in PAIRS)
        assert local(VERTICES, colors, matrix)
        encodings = {encoding(colors, matrix, order) for order in ORDERS}
        key = min(encodings)
        assert key not in seen
        seen.add(key)
        labeled += len(encodings)
        restricted += keep
        flag = index + 1

        for v in VERTICES:
            others = tuple(u for u in VERTICES if u != v)
            if local(others, colors, matrix):
                kind, labelings = representative(colors, matrix, others)
                adjacent = [sum(bool(matrix[v][order[i]]) for order in labelings)
                            for i in range(4)]
                for i, count in enumerate(adjacent):
                    rows[kind, "black", i][flag] += (
                        2 * count - len(labelings) * (colors[v] == 0)
                    )
                    for j in range(4):
                        if i != j:
                            rows[kind, i, j][flag] += count - adjacent[j]

            if colors[v] == 0:
                if all(matrix[v][u] for u in others):
                    normalization[flag] -= 1
                # Every split into two unlabeled pairs is counted once here.
                for u in others[1:]:
                    pair = (others[0], u)
                    rest = tuple(w for w in others if w not in pair)
                    first = (v, *pair)
                    second = (v, *rest)
                    if local(first, colors, matrix) and local(second, colors, matrix):
                        left = min(encoding(colors, matrix, (v, *order))
                                   for order in (pair, pair[::-1]))
                        right = min(encoding(colors, matrix, (v, *order))
                                    for order in (rest, rest[::-1]))
                        record_square(squares[(0,)], flag, left, right)

        for vertices in TRIPLES:
            if not local(vertices, colors, matrix):
                continue
            u, v = (w for w in VERTICES if w not in vertices)
            if not (local((*vertices, u), colors, matrix)
                    and local((*vertices, v), colors, matrix)):
                continue
            kind, labelings = representative(colors, matrix, vertices)
            for order in labelings:
                left = (colors[u], *(matrix[u][w] for w in order))
                right = (colors[v], *(matrix[v][w] for w in order))
                record_square(squares[kind], flag, left, right)

    assert (len(seen), restricted) == (17950, 9530)
    assert labeled == labeled_local_count()
    linear_rows = {primitive(row) for row in rows.values()}
    linear_rows.add(primitive(normalization))
    return linear_rows, squares


def square_cells(block, indices):
    """Normalize a whole block by one positive integer, preserving its basis."""
    divisor = 0
    for column in block.values():
        divisor = gcd(divisor, *column.values())
    assert divisor > 0
    return {
        tuple(sorted((indices[a], indices[b]))):
            tuple(sorted((flag, value // divisor) for flag, value in column.items()))
        for (a, b), column in block.items()
    }


def fingerprints(size, cells):
    """Permutation-invariant signatures used to find candidate basis bijections."""
    diagonal = [cells.get((i, i), ()) for i in range(size)]
    return [
        (diagonal[i], tuple(sorted(
            (diagonal[j], cells.get(tuple(sorted((i, j))), ()))
            for j in range(size) if i != j
        )))
        for i in range(size)
    ]


def same_square(size, left, right, left_signatures, right_signatures):
    """Check equality of all coefficients under one simultaneous permutation."""
    candidates = [
        [j for j in range(size) if left_signatures[i] == right_signatures[j]]
        for i in range(size)
    ]
    order = sorted(range(size), key=lambda i: len(candidates[i]))
    image = {}
    used = set()

    def extend(k):
        if k == size:
            return True
        i = order[k]
        for j in candidates[i]:
            if j in used:
                continue
            if any(left.get(tuple(sorted((i, a))), ())
                   != right.get(tuple(sorted((j, b))), ())
                   for a, b in image.items()):
                continue
            image[i] = j
            used.add(j)
            if extend(k + 1):
                return True
            used.remove(j)
            del image[i]
        return False

    return extend(0)


def read_sdpa():
    """Read the stored blocks in the sign convention of Appendix A."""
    with (ROOT / "flags.sdpa").open(encoding="utf-8") as source:
        lines = (line.split() for line in source
                 if line.strip() and not line.startswith("*"))
        count = int(next(lines)[0])
        blocks = int(next(lines)[0])
        sizes = tuple(map(int, next(lines)))
        objective = next(lines)
        assert count == len(objective) == 17950
        assert sizes[:2] == (-count, -1) and len(sizes) == blocks
        assert Counter(s for s in sizes if s > 0) == {55: 1, 53: 38}
        nonnegative = set()
        linear = {}
        square = defaultdict(lambda: defaultdict(dict))
        for f, b, i, j, value in lines:
            f, b, i, j = map(int, (f, b, i, j))
            assert 0 <= f <= count and 1 <= b <= blocks
            size = sizes[b - 1]
            assert 1 <= i <= j <= abs(size)
            if b == 1:
                assert f == i == j and int(value) == 1 and f not in nonnegative
                nonnegative.add(f)
            elif b > 2 and size < 0:
                assert i == j
                row = linear.setdefault((b, i), {})
                assert f not in row
                row[f] = -int(value) if f == 0 else int(value)
            elif b > 2:
                assert f > 0 and int(value) > 0
                column = square[b][i - 1, j - 1]
                assert f not in column
                column[f] = int(value)
        assert nonnegative == set(range(1, count + 1))
        return sizes, linear, square


def verify_upstream():
    linear_rows, expected_squares = reconstruct()
    sizes, linear, squares = read_sdpa()
    nonzero = 0
    for row in linear.values():
        normalized = primitive(row)
        if normalized:
            assert normalized in linear_rows, "An archived linear row has no matching flag expression"
            nonzero += 1
    assert nonzero == 2490

    available = defaultdict(list)
    for expected in expected_squares.values():
        basis = sorted({f for pair in expected for f in pair})
        size = len(basis)
        cells = square_cells(expected, {f: i for i, f in enumerate(basis)})
        signatures = fingerprints(size, cells)
        available[size, tuple(sorted(signatures))].append((cells, signatures))

    for b, stored in squares.items():
        size = sizes[b - 1]
        cells = square_cells(stored, {i: i for i in range(size)})
        signatures = fingerprints(size, cells)
        choices = available[size, tuple(sorted(signatures))]
        match = next((i for i, (other, other_signatures) in enumerate(choices)
                      if same_square(size, cells, other, signatures, other_signatures)), None)
        assert match is not None, f"Archived square block {b} has no matching flag product"
        choices.pop(match)
    assert len(squares) == 39 and not any(available.values())
    print("Five-vertex basis, 39 square blocks, and 2490 nonzero linear rows verified.",
          flush=True)


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Run without Python optimization.")
    verify_upstream()
