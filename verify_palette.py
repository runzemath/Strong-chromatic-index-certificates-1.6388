"""Check the interval bounds and integer recurrence for Appendix C."""

from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from functools import lru_cache
from math import isqrt
from multiprocessing import get_context
from pathlib import Path
from typing import NamedTuple

from verify_moments import ENDPOINTS


M = 100_000_000
M2 = M * M
M3 = M2 * M
SCALE = 1_000_000_000_000
BASE = 66_521_250
REFERENCE = 86_485_000
K_LO, K_HI = 185_480_000, 185_481_000
TRIANGLE_MAX, STAR_MAX = 61_700_000, 61_800_000
TRIANGLE_MULTIPLIER = 10_000_000
TRIANGLE_SLOPE = 35_100_000
MOMENTS = (
    (18_000_000, 8_000_000, 6_410_000, 28_100_000, 6_140_000, 29_000_000),
    (24_000_000, 2_500_000, 6_865_000, 26_800_000, 6_321_000, 28_500_000),
)
TAIL = (
    (77_000_000, 30_026_000, 0),
    (70_000_000, 8_851_000, 27_500_000),
    (62_000_000, 6_681_000, 30_600_000),
    (53_000_000, 5_379_000, 32_700_000),
    (43_000_000, 2_835_000, 37_500_000),
)
SHARPER_TAIL = (
    (77_000_000, 30_151_000, 0),
    (70_000_000, 8_976_000, 27_500_000),
    (66_000_000, 7_156_000, 30_100_000),
    (60_000_000, 6_364_000, 31_300_000),
    (54_000_000, 5_644_000, 32_500_000),
    (50_000_000, 3_700_000, 36_100_000),
    (43_000_000, 2_750_000, 38_000_000),
)


def ceil_div(numerator, denominator):
    return (numerator + denominator - 1) // denominator


class Schedule(NamedTuple):
    q: int
    cutoff: int
    claim: int
    cuts: tuple
    density: tuple
    bonuses: tuple


class Envelope:
    """Locate minima of x*r + y over the points (x, y) in one family."""

    def __init__(self, lines):
        by_slope = {}
        for x, y, q in lines:
            slope = Fraction(x, q)
            if slope in by_slope:
                _, previous_y, previous_q = by_slope[slope]
                assert y * previous_q == previous_y * q
            by_slope[slope] = (x, y, q)
        self.lines = tuple(line for _, line in sorted(by_slope.items()))
        breaks = []
        for (x, y, q), (xx, yy, qq) in zip(self.lines, self.lines[1:]):
            breaks.append(Fraction(yy * q - y * qq, xx * q - x * qq))
        assert all(a <= b for a, b in zip(breaks, breaks[1:]))
        self.integral = all(b.denominator == 1 for b in breaks)
        self.breaks = tuple(int(b) if self.integral else (b.numerator, b.denominator) for b in breaks)

    def minimum_index(self, numerator, denominator=1):
        if self.integral:
            return bisect_right(self.breaks, -ceil_div(numerator, denominator))
        lo, hi = 0, len(self.breaks)
        while lo < hi:
            mid = (lo + hi) // 2
            p, q = self.breaks[mid]
            if p * denominator <= -numerator * q:
                lo = mid + 1
            else:
                hi = mid
        return lo


def reference_lines(weights, cuts):
    # Supporting lines in Appendix C.1. Store x = -p/q and
    # y = (a+p*M)/q as the triples (-p, a+p*M, q).
    hull = []
    for w, d in zip(weights, cuts):
        if hull and hull[-1][0] == w:
            continue
        while len(hull) > 1:
            w0, d0 = hull[-2]
            w1, d1 = hull[-1]
            if (d1 - d0) * (w - w1) < (d - d1) * (w1 - w0):
                break
            hull.pop()
        hull.append((w, d))
    slopes = [(0, 1)]
    slopes.extend((d1 - d0, w1 - w0)
                  for (w0, d0), (w1, d1) in zip(hull, hull[1:]))
    cross_x, cross_q = 0, 1
    for w, d in zip(weights, cuts):
        if d < REFERENCE and w < M and (REFERENCE - d) * cross_q > cross_x * (M - w):
            cross_x, cross_q = REFERENCE - d, M - w
    if cross_x:
        slopes.append((cross_x, cross_q))
    points = tuple((M - w, REFERENCE - d) for w, d in zip(weights, cuts))
    return [(x, max(q * b - x * v for v, b in points), q) for x, q in slopes]


@lru_cache(maxsize=48_000)
def degree_data(dl, dh, density):
    pairs = {(BASE, BASE)}
    for intercept, slope in density:
        pairs.add((intercept - slope + ceil_div(slope * dl, M),
                   intercept - slope + ceil_div(slope * dh, M)))
    # A pair greater at both endpoints can never lower the budget.
    frontier = []
    smallest_hi = M + 1
    for lo, hi in sorted(pairs):
        if hi < smallest_hi:
            frontier.append((lo, hi))
            smallest_hi = hi
    return ceil_div(dh * dh, M), tuple(frontier)


def linear_gain(s, pieces):
    for start, intercept, slope in pieces:
        if s >= start:
            return intercept + slope * s // M
    return 0


class BasePhase:
    def __init__(self, schedule, weights):
        self.smallest = weights[0]
        self.width = M - self.smallest
        self.smallest_squared = ceil_div(self.smallest**2, M)
        self.threshold = schedule.cuts[0]
        self.density = schedule.density
        self.bonuses = schedule.bonuses
        self.bonus_starts = tuple(start for start, _ in self.bonuses)
        lines = reference_lines(weights, schedule.cuts)
        self.nonnegative = Envelope(line for line in lines if line[1] >= 0)
        self.negative = Envelope(line for line in lines if line[1] < 0)
        slope = ceil_div(K_HI * (REFERENCE - self.threshold), M)
        cap, cap_slope = M, M
        for intercept, gradient in self.density:
            candidate = intercept - gradient + ceil_div((gradient + slope) * self.threshold, M)
            if candidate < cap:
                cap, cap_slope = candidate, gradient + slope
        self.triangle_caps = [(cap, cap_slope)]
        beta = max(ceil_div(M * (w - self.smallest), d - self.threshold)
                   for w, d in zip(weights[1:], schedule.cuts[1:]))
        factor = ceil_div((M + self.smallest) * beta, M)
        if factor * M2 <= K_LO * self.smallest**2:
            cap, slope = self.triangle_caps[0]
            cap = (ceil_div(self.smallest**2 * cap, M2)
                   + ceil_div(factor * (M - REFERENCE) * self.threshold * (M - self.threshold), M3))
            self.triangle_caps.append((cap, ceil_div(self.smallest**2 * slope, M2)))
        self.boxes = 0

    def size_bound(self, dh, uh):
        mean_hi = self.smallest + (self.width * uh + M - 1) // M
        return (dh * mean_hi + M - 1) // M

    def profile_density(self, dl, dh, ul, uh):
        budget = self.budget(dl, dh, ul)
        high_mass = (dh * uh + M - 1) // M
        hh = min(budget, (high_mass * high_mass + M - 1) // M)
        um = max(ul, min(M // 2, uh))
        mixed = (2 * dh * dh * um * (M - um) + M3 - 1) // M3
        cross = min(budget - hh, mixed)
        reference = (hh * M + cross * self.smallest
                     + (budget - hh - cross) * self.smallest_squared + M - 1) // M
        return budget, reference

    def degree_sum_cap(self, dl, dh):
        def gain(d):
            return ceil_div(K_HI * ((REFERENCE - self.threshold) * d
                                   + (M - REFERENCE) * (M - d)), M2)
        lo, hi = gain(dl), gain(dh)
        _, density = degree_data(dl, dh, self.density)
        return min(max(lo + a, hi + b) for a, b in density)

    def triangle_ok(self, dl, dh, ul, uh, cap, slope):
        limit = cap + slope * (dl - self.threshold) // M
        size = self.size_bound(dh, uh)
        if size * size <= limit * M:
            return True
        return self.profile_density(dl, dh, ul, uh)[1] <= limit

    def hit_ok(self, dl, dh, ul, uh, cap):
        t = self.size_bound(dh, uh)
        if t <= cap:
            return True
        budget, c_ref = self.profile_density(dl, dh, ul, uh)
        c = min(c_ref, (t * t + M - 1) // M)
        s0 = max(0, t * t // M - c)
        if t - s0 * (3 * M - isqrt(M * s0) - 1) // (6 * M) <= cap:
            return True
        x = M - t
        exponential = (M + x + x*x // (2*M) + x**3 // (6*M2)
                       + x**4 // (24*M3))
        required = ceil_div((M - cap) * M, exponential)
        s = M - c
        assert s >= self.bonus_starts[0]
        gain = 1_800_000 + 38_500_000 * min(s, 44_600_000) // M
        i = bisect_right(self.bonus_starts, s) - 1
        start, value = self.bonuses[i]
        if i and s - start < 10_000:
            previous = self.bonuses[i - 1][1]
            gain += previous + (value - previous) * (s - start) // 10_000
        else:
            gain += value
        if (gain >= required or s * (3*M - isqrt(M*s) - 1) // (6*M) >= required
                or linear_gain(s, TAIL) >= required or linear_gain(s, SHARPER_TAIL) >= required):
            return True

        s_ref = M - c_ref
        if (25_000_000 <= s_ref <= 48_000_000 and required <= 20_835_000
                and self.earlier_moment_ok(dl, dh, t, budget, c_ref, required)):
            return True
        return self.mixed_moment_ok(dl, dh, uh, t, budget, c_ref, required)

    def earlier_moment_ok(self, dl, dh, t, budget, c_ref, required):
        s_ref = M - c_ref
        difference = max(0, self.degree_sum_cap(dl, dh) - budget)
        smallest_sq = self.smallest**2
        for bound, slope in self.triangle_caps:
            if s_ref >= 25_500_000:
                gap, increase = max(0, TRIANGLE_MAX - bound), max(0, bound - TRIANGLE_MAX)
                derivative = ceil_div(TRIANGLE_MULTIPLIER * gap * max(0, 4*c_ref - M), M2)
                if ((TRIANGLE_SLOPE - derivative) * K_LO * smallest_sq
                        >= TRIANGLE_MULTIPLIER * slope * c_ref * M):
                    correction = (ceil_div(TRIANGLE_MULTIPLIER * slope * c_ref * difference, M2*K_LO)
                                  + ceil_div(TRIANGLE_MULTIPLIER * increase, M))
                    bonus = TRIANGLE_MULTIPLIER * gap * c_ref * (2*c_ref - M) // M3
                    if 3_775_000 + TRIANGLE_SLOPE * s_ref // M - correction + bonus >= required:
                        return True
            if s_ref > 42_000_000 or required > 18_711_000:
                continue
            dgap, increase = max(0, STAR_MAX - bound), max(0, bound - STAR_MAX)
            root = isqrt(M * c_ref) + 1
            c_cube = c_ref**3 // M2
            for j, (star, clique, b_lo, a_lo, b_hi, a_hi) in enumerate(MOMENTS):
                breakpoint = 30_000_000 if j == 0 else 32_000_000
                line = b_lo + a_lo * s_ref // M if s_ref <= breakpoint else b_hi + a_hi * s_ref // M
                remaining_gap, slope_min = dgap, a_lo
                if j == 1 and dgap:
                    low_line = (7_600_000 + 26_800_000 * s_ref // M if s_ref < 32_000_000
                                else 7_760_000 + 26_300_000 * s_ref // M)
                    assert low_line >= line
                    portion = min(6_000_000, dgap)
                    line += (low_line - line) * portion // 6_000_000
                    remaining_gap -= portion
                    slope_min = 26_300_000
                factor = ceil_div(star * t**3, M3) + ceil_div(clique * c_ref * root, M2)
                derivative = ceil_div(3 * star * remaining_gap * c_ref * c_ref, M3)
                if (slope_min - derivative) * K_LO * smallest_sq < slope * factor * M2:
                    continue
                correction = (ceil_div(slope * factor * difference, M*K_LO)
                              + ceil_div((star + clique) * increase, M))
                addition = star * remaining_gap * c_cube // M2
                if line + addition - correction >= required:
                    return True
        return False

    def check_interval(self, lower, upper, check, *bound):
        stack = [(lower, upper, 0, M if self.width else 0)]
        width = self.width
        boxes = 0
        while stack:
            dl, dh, ul, uh = stack.pop()
            boxes += 1
            if check(dl, dh, ul, uh, *bound):
                continue
            if max(dh - dl, uh - ul) <= 1:
                raise ValueError(f"Unverified interval bound {bound} at {(dl, dh, ul, uh)}")
            if dh - dl > 1 and (uh - ul <= 1 or dh - dl >= (uh - ul) * width // M):
                d = (dl + dh) // 2
                stack.append((dl, d, ul, uh))
                stack.append((d, dh, ul, uh))
            else:
                u = (ul + uh) // 2
                assert ul < u < uh
                stack.append((dl, dh, ul, u))
                stack.append((dl, dh, u, uh))
        self.boxes += boxes


HERE = Path(__file__).resolve().parent
OBS_DEGREE = (0, 1, 2, 3, 4, 2, 3, 4, 4, 3, 4)
TRI_INDEX = (0, 1, 2, 3, 5, 6, 9)


class Moment(NamedTuple):
    lo: int
    hi: int
    U: int
    R: int
    slope: int
    intercept: int
    triangle: tuple
    degree: tuple


def read_moments(rows):
    return tuple(Moment(row["lo"], row["hi"], row["U"], row["R"], row["slope"], row["intercept"],
                        tuple((i, v) for i, v in zip(TRI_INDEX, row["tri"], strict=True) if v),
                        tuple((i, v) for i, v in enumerate(row["deg"]) if v))
                 for row in rows)


def mul_hi(a, b):
    return ceil_div(a * b, M)


def mul_lo(a, b):
    return a * b // M


def observe(t, c):
    """Pointwise upper bounds, and lower bounds on averages with derivative bounds."""
    r2 = mul_hi(t, t)
    r3 = mul_hi(r2, t)
    r4 = mul_hi(r3, t)
    a = min(c, r2)
    b = mul_hi(a, isqrt(M * c) + 1)
    upper = (M, t, r2, r3, r4, a, mul_hi(t, a), mul_hi(r2, a), mul_hi(a, a), b, mul_hi(t, b))
    tp = (M, t, r2, r3)
    average = (M, *(min(upper[i], mul_hi(c, tp[OBS_DEGREE[i]-1]))
                    for i in range(1, len(OBS_DEGREE))))
    d2 = c * c // t
    d3 = c * d2 // t
    d4 = c * d3 // t
    tri_low = max(0, 2 * d2 - mul_hi(t, c))
    tri_slope = max(0, ceil_div(4 * M * c, t) - t)
    sqrt_low = isqrt(M2 * tri_low // t)
    sqrt_hi = isqrt(M2 * (tri_low + 3) // t) + 1
    ra = mul_lo(tri_low, sqrt_low)
    aa = tri_low * tri_low // t
    ra_derivative = ceil_div(3 * sqrt_hi * tri_slope, 2 * M)
    aa_derivative = ceil_div(2 * (tri_low + 3) * tri_slope, t)
    ratio = ceil_div(M * c, t)
    ratio2 = mul_hi(ratio, ratio)
    ratio3 = mul_hi(ratio2, ratio)
    low = (M, c, d2, d3, d4, tri_low, ra, aa, aa, 0, 0)
    derivative = (0, M, 2 * ratio, 3 * ratio2, 4 * ratio3, tri_slope,
                  ra_derivative, aa_derivative, aa_derivative, 0, 0)
    return upper, average, low, derivative


def weighted(weights, upper, average, lower, derivative):
    return (sum(mul_hi(a, upper[i]) for i, a in weights),
            sum(mul_hi(a, average[i]) for i, a in weights),
            sum(mul_lo(a, lower[i]) for i, a in weights),
            sum(mul_hi(a, derivative[i]) for i, a in weights))


class Phase(BasePhase):
    def __init__(self, schedule, weights, rows, moments):
        super().__init__(schedule, weights)
        self.rows = rows
        self.moments = moments
        self.extra_cache = {}
        self.excess_intercept = 0
        if self.width:
            for w, hi in zip(weights, schedule.cuts[1:] + (M,), strict=True):
                bound = mul_hi(w, hi - self.threshold)
                bound -= (M-self.threshold) * (w-self.smallest) // self.width
                self.excess_intercept = max(self.excess_intercept, bound)

    def budget(self, dl, dh, ul):
        key = (dl, dh, ul)
        known = self.extra_cache.get(key)
        if known is not None:
            return known
        value, density_bounds = degree_data(dl, dh, self.density)
        z = self.width - self.width * ul // M
        endpoints = []
        pos, neg = self.nonnegative, self.negative
        if pos.lines:
            x, y, q = pos.lines[pos.minimum_index(z)]
            factor = x*z + y
            endpoints.append((dl * factor, dh * factor, q * M2))
        if neg.lines:
            lo = neg.minimum_index(dl * z, 2 * dl - M)
            hi = neg.minimum_index(dh * z, 2 * dh - M)
            for x, y, q in neg.lines[lo:hi+1]:
                endpoints.append((dl * x * z + (2 * dl - M)*y,
                                  dh * x * z + (2 * dh - M)*y, q * M2))
        for raw_lo, raw_hi, den in endpoints:
            low = ceil_div((K_HI if raw_lo >= 0 else K_LO) * raw_lo, den)
            high = ceil_div((K_HI if raw_hi >= 0 else K_LO) * raw_hi, den)
            for bound_lo, bound_hi in density_bounds:
                value = min(value, max(bound_lo + low, bound_hi + high))
        for _, _, b, g, klo, khi in self.rows:
            bound_lo = b - g + mul_hi(g, dl)
            bound_hi = b - g + mul_hi(g, dh)
            if khi == 0:
                value = min(value, bound_hi)
                continue
            for raw_lo, raw_hi, den in endpoints:
                low = ceil_div((khi if raw_lo >= 0 else klo) * raw_lo, den)
                high = ceil_div((khi if raw_hi >= 0 else klo) * raw_hi, den)
                value = min(value, max(bound_lo + low, bound_hi + high))
        assert value >= 0
        self.extra_cache[key] = value
        return value

    def budgets(self, dl, dh, budget):
        yield K_LO, max(0, self.degree_sum_cap(dl, dh) - budget)
        for lo, hi, b, g, klo, khi in self.rows:
            if klo and lo <= dl and dh <= hi:
                values = (b - g + mul_hi(g, d) +
                          ceil_div(khi * ((REFERENCE-self.threshold)*d + (M-REFERENCE)*(M-d)), M2)
                          for d in (dl, dh))
                yield klo, max(0, max(values) - budget)

    def mu_charge(self, penalty, budget_slope, budgets, zero_bound):
        options = [(0, mul_hi(penalty, zero_bound))]
        for k, diff in budgets:
            options.append((ceil_div(penalty * M3, k * self.smallest**2),
                            ceil_div(penalty * diff, k)))
        feasible = [(a, b) for a, b in options if a <= budget_slope]
        cap = min(b for _, b in feasible)
        for a, b in options:
            if a > budget_slope and b < cap:
                for c, d in feasible:
                    cap = min(cap, d + ceil_div((budget_slope - c) * (b - d), a - c))
        return cap

    def mixed_moment_ok(self, dl, dh, uh, t, budget, c, required):
        s = M - c
        if not 25_000_000 <= s <= 60_000_000:
            return False
        candidates = [row for row in self.moments if row.lo <= s <= row.hi and required <= ENDPOINTS[row.hi]]
        if not candidates:
            return False
        bounds = observe(t, c)
        budgets = tuple(self.budgets(dl, dh, budget))
        zero_bound = mul_hi(M - self.threshold, t)
        if self.width:
            zero_bound = min(zero_bound,
                             mul_hi(dh, self.excess_intercept + mul_hi(M - self.threshold, uh)))
        for row in candidates:
            tri_sup, tri_avg, tri_lo, tri_derivative = weighted(row.triangle, *bounds)
            deg_sup, deg_avg, deg_lo, deg_derivative = weighted(row.degree, *bounds)
            gd, hd = max(0, row.R - self.threshold), max(0, self.threshold - row.R)
            line = row.intercept + mul_lo(row.slope, s) + mul_lo(gd, deg_lo) - mul_hi(hd, deg_avg)
            for U, nu in self.triangle_caps:
                gt, ht = max(0, row.U - U), max(0, U - row.U)
                value = line + mul_lo(gt, tri_lo) - mul_hi(ht, tri_avg)
                if value < required:
                    continue
                budget_slope = row.slope - mul_hi(gd, deg_derivative) - mul_hi(gt, tri_derivative)
                if budget_slope < 0:
                    continue
                penalty = deg_sup + mul_hi(nu, tri_sup)
                charge = self.mu_charge(penalty, budget_slope, budgets, zero_bound)
                if value - charge >= required:
                    return True
        return False


def read_bound(line, n):
    count, *values = map(int, line.split())
    assert count >= 2 and count <= 20
    assert len(values) == 2 * count + n
    triangle = tuple(zip(values[:2 * count:2], values[1:2 * count:2], strict=True))
    caps = tuple(values[2 * count:])
    assert all(0 < x <= M for x in caps)
    assert all(0 < U <= M and 0 <= v <= M for U, v in triangle)
    return triangle, caps


def scheduled_step(q, scale):
    below_40, below_10, below_02 = 185, 85, 45
    if scale < SCALE // 50:
        return min(q, below_02)
    if scale < SCALE // 10:
        return min(q, below_10)
    if scale < 2 * SCALE // 5:
        return min(q, below_40)
    return q


def prepare_checks(schedule, rows, moments, bounds):
    weights = (M,) * len(schedule.cuts)
    scale, accumulated = SCALE, 0
    checks = []
    records = iter(bounds)
    while scale > schedule.cutoff:
        scans, caps = read_bound(next(records), len(weights))
        assert all(a <= b for a, b in zip(caps, caps[1:]))
        assert all(0 < w <= M for w in weights)
        assert weights[-1] == M and all(a <= b for a, b in zip(weights, weights[1:]))
        checks.append((len(checks) + 1, schedule, weights, scans, caps, rows, moments))
        q, peak = scheduled_step(schedule.q, scale), caps[-1]
        accumulated += ceil_div(scale * peak, M * q)
        updated = []
        for w, cap in zip(weights, caps, strict=True):
            assert 0 < cap <= peak < 2 * cap
            r = peak - cap
            denominator = 2 * q*q * cap*cap
            numerator = denominator - 2*q*cap*r - r*(cap-r)
            updated.append(ceil_div(w * numerator, denominator))
        weights = tuple(updated)
        next_scale = ceil_div(scale * (q-1), q)
        assert next_scale < scale
        scale = next_scale
    assert weights[-1] == M and all(0 < w <= M for w in weights)
    assert all(a <= b for a, b in zip(weights, weights[1:]))
    scans, caps = read_bound(next(records), len(weights))
    assert next(records, None) is None
    checks.append((len(checks) + 1, schedule, weights, scans, caps, rows, moments))
    return checks, accumulated + ceil_div(scale * max(caps), M)


def check_phase(job):
    number, schedule, weights, scans, caps, rows, moments = job
    try:
        phase = Phase(schedule, weights, rows, moments)
        analytical = tuple(phase.triangle_caps)
        assert scans[:len(analytical)] == analytical, (scans, analytical)
        numerical = scans[len(analytical):]
        assert tuple(slope for _, slope in numerical) == (
            74_000_000, 98_000_000, 48_000_000, 55_000_000, 60_000_000,
            65_000_000, 70_000_000, 80_000_000, 88_000_000)
        for cap, slope in numerical:
            phase.check_interval(schedule.cuts[0], M, phase.triangle_ok, cap, slope)
            phase.triangle_caps.append((cap, slope))
        for lo, hi, cap in zip(schedule.cuts, schedule.cuts[1:] + (M,), caps, strict=True):
            phase.check_interval(lo, hi, phase.hit_ok, cap)
    except (AssertionError, ValueError) as error:
        raise ValueError(f"Palette phase {number}: {error}") from error
    return phase.boxes


def verify_palette(parameters, schedule, rows, moments, jobs):
    lines = (HERE / "palette_bounds.txt").read_text(encoding="ascii").splitlines()
    checks, total = prepare_checks(schedule, tuple(map(tuple, rows)), read_moments(moments),
                                   lines)
    assert len(checks) == parameters["phases"] + 1
    assert total == parameters["cap"] < schedule.claim
    assert schedule.claim * M >= schedule.cuts[0] * SCALE
    print(f"Checking {len(checks) - 1} phases and the completion step.", flush=True)

    def report(results):
        boxes = 0
        for phase, count in enumerate(results, start=1):
            boxes += count
            if phase % 50 == 0 or phase == len(checks):
                print(f"Verified {phase}/{len(checks)} phase and completion bounds.", flush=True)
        return boxes

    if jobs == 1:
        boxes = report(map(check_phase, checks))
    else:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn")) as pool:
            boxes = report(pool.map(check_phase, checks))
    print(f"Palette cap {total} / {SCALE}, phases {len(checks)-1}, boxes {boxes}.", flush=True)
