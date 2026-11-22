"""Verify the local flag certificates and palette bound using exact arithmetic."""
import argparse
from fractions import Fraction as Q
import json
import os
from pathlib import Path
from shutil import which

from replay_palette_cpp import replay_cpp
from verify import verify_sdp
from verify_gain6 import verify_occupancy
from verify_mass import COEFFICIENT, verify_mass
from verify_moments import verify_moments
from verify_offer_extensions import verify_extensions
from verify_palette import M, SCALE, Schedule, ceil_div, verify_palette
from verify_upstream import verify_upstream

HERE = Path(__file__).resolve().parent


def integer(x):
    assert x.denominator == 1
    return int(x)


def witness_file(name):
    path = Path(name)
    assert path.name == name and len(path.parts) == 1 and name not in (".", "..")
    return HERE / name


def read_inputs(check_local, check_ordinary):
    parameters = json.loads((HERE / "parameters.json").read_text(encoding="ascii"))
    tau = Q(parameters["threshold"])
    pruning = Q(parameters["pruning"])
    assert tau == Q("0.8175") and pruning == Q("0.8194")
    assert 2 * pruning == Q(parameters["bound"]) == Q("1.6388")
    assert parameters["cap"] == 819_392_866_996 and parameters["phases"] == 487
    if check_local:
        verify_upstream()
        verify_sdp()

    density = []
    for name in parameters["signed_witnesses"]:
        path = witness_file(name)
        witness = json.loads(path.read_text(encoding="ascii"))
        theta, gamma, b = (Q(witness[key]) for key in ("threshold", "slope", "intercept"))
        assert theta == tau and 0 <= gamma <= 1 and 0 < b <= 1
        assert Q(witness.get("signed", str(COEFFICIENT))) == COEFFICIENT
        if check_local:
            assert verify_mass(path) == (theta, gamma, b)
        density.append((integer(M * b), integer(M * gamma)))

    extra_density = []
    for name in parameters["density_witnesses"]:
        path = witness_file(name)
        witness = json.loads(path.read_text(encoding="ascii"))
        theta, gamma, b, lo, hi, signed = (Q(witness[key]) for key in (
            "threshold", "slope", "intercept", "root_lower", "root_upper", "signed"))
        assert theta == lo == tau and hi == 1
        assert 0 <= gamma <= 1 and 0 < b <= 1 and 0 <= signed <= Q(1, 30)
        assert witness["all_edges"] is False
        if check_local:
            assert verify_mass(path) == (theta, gamma, b)
        k = signed * 60 * M
        extra_density.append((integer(M * lo), integer(M * hi), integer(M * b), integer(M * gamma),
                              k.numerator // k.denominator, ceil_div(k.numerator, k.denominator)))

    if check_ordinary:
        verify_occupancy()
        verify_extensions(sharper=True)
    path = witness_file(parameters["moment_witnesses"])
    moments = verify_moments(path) if check_ordinary else json.loads(path.read_text(encoding="ascii"))
    assert len(moments) == 33

    head, *body = (HERE / "schedule.txt").read_text(encoding="ascii").splitlines()
    n, q, stop, tolerance, claim, count, rows, method = map(int, head.split())
    assert (n, q, stop, tolerance, count, rows, method) == (176, 110, 2_200_000_000, 5800, 18, 5, 10)
    assert Q(claim, SCALE) == pruning and parameters["cap"] < claim
    assert [tuple(map(int, row.split())) for row in body[:rows]] == density
    cuts = tuple(map(int, body[rows].split()))
    assert len(cuts) == n and Q(cuts[0], M) == tau and cuts[-1] < M
    assert all(a < b for a, b in zip(cuts, cuts[1:]))
    assert Q("1.8548") < 60 * COEFFICIENT < Q("1.85481")
    bonuses = tuple(tuple(map(int, row.split())) for row in body[rows + 1:])
    cases = json.loads((HERE/"gain6_witness.json").read_text(encoding="ascii"))["cases"][:-1]
    expected = [(25_000_000, 284_000), (25_500_000, 311_000)]
    expected.extend((integer(M * Q(case["lo"])), integer(M * Q(case["bonus"]))) for case in cases)
    assert bonuses == tuple(expected) and len(bonuses) == count
    schedule = Schedule(q, stop, claim, cuts, tuple(density), bonuses)
    return parameters, schedule, tuple(extra_density), moments


def main():
    if not __debug__:
        raise SystemExit("Run without Python optimization to enable the certificate checks.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appendix", choices=("A", "B", "C"), help="check only the indicated appendix")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="number of palette workers")
    parser.add_argument("--cpp", action="store_true", help="also regenerate the palette bounds with the C++ search")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 61:
        parser.error("--jobs must be between 1 and 61")
    compiler = None
    if args.cpp:
        if args.appendix in ("A", "B"):
            parser.error("--cpp requires a check that includes Appendix C")
        compiler = which("g++")
        if compiler is None:
            parser.error("--cpp requires GNU g++ with C++20, OpenMP, and signed 128-bit integer support on PATH")
    parameters, schedule, extra_density, moments = read_inputs(
        check_local=args.appendix in (None, "A"), check_ordinary=args.appendix in (None, "B"))
    if args.appendix in (None, "C"):
        if compiler is not None:
            replay_cpp(compiler, parameters, extra_density, moments, args.jobs)
        verify_palette(parameters, schedule, extra_density, moments, args.jobs)
    print("All requested checks passed.", flush=True)


if __name__ == "__main__":
    main()
