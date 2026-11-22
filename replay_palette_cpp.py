"""Regenerate and compare the recorded bounds using the C++ palette search."""

import os
from pathlib import Path
import re
from shutil import copyfile
import subprocess
from tempfile import TemporaryDirectory

from verify_palette import SCALE

HERE = Path(__file__).resolve().parent


def parameter_header(density, moments):
    lines = [
        "struct ExtraDensity { I lo, hi, b, g, klo, khi; bool all; };",
        f"constexpr std::array<ExtraDensity, {len(density)}> EXTRA_DENSITY{{{{",
    ]
    for row in density:
        values = ", ".join(map(str, row))
        lines.append(f"    {{{values}, false}},")
    lines.extend([
        "}};",
        "struct ExtraMoment { I lo, hi, U, R, slope, intercept; "
        "std::array<I, 7> tri; std::array<I, 11> deg; };",
        f"constexpr std::array<ExtraMoment, {len(moments)}> EXTRA_MOMENTS{{{{",
    ])
    for row in moments:
        values = ", ".join(str(row[key]) for key in ("lo", "hi", "U", "R", "slope", "intercept"))
        triangle = ", ".join(map(str, row["tri"]))
        degree = ", ".join(map(str, row["deg"]))
        lines.append(f"    {{{values}, {{{triangle}}}, {{{degree}}}}},")
    lines.append("}};")
    return "\n".join(lines) + "\n"


def replay_cpp(compiler, parameters, density, moments, jobs):
    print("Replaying the C++ palette search.", flush=True)
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        source = directory / "palette.cpp"
        binary = directory / ("palette.exe" if os.name == "nt" else "palette")
        generated = directory / "palette_bounds.txt"
        copyfile(HERE / "palette.cpp", source)
        (directory / "parameters.hpp").write_text(parameter_header(density, moments), encoding="ascii")
        command = [compiler, "-std=c++20", "-O3", "-fopenmp", str(source), "-o", str(binary)]
        environment = dict(os.environ, OMP_NUM_THREADS=str(jobs))
        try:
            subprocess.run(command, check=True)
            with (HERE / "schedule.txt").open("rb") as schedule:
                result = subprocess.run([str(binary), str(generated)], stdin=schedule,
                                        stdout=subprocess.PIPE, encoding="ascii",
                                        check=True, env=environment)
        except (OSError, subprocess.CalledProcessError) as error:
            raise SystemExit(f"C++ palette replay failed ({error}).") from error

        match = re.fullmatch(r"Palette cap (\d+) / (\d+), phases (\d+), boxes (\d+)\s*", result.stdout)
        if match is None or tuple(map(int, match.groups()[:3])) != (
                parameters["cap"], SCALE, parameters["phases"]):
            raise SystemExit(f"Unexpected C++ palette result ({result.stdout.strip()}).")

        recorded = (HERE / "palette_bounds.txt").read_text(encoding="ascii").splitlines()
        regenerated = generated.read_text(encoding="ascii").splitlines()
        if len(recorded) != len(regenerated):
            raise SystemExit("The number of regenerated palette rows differs from the recorded count.")
        for index, (expected, actual) in enumerate(zip(recorded, regenerated), start=1):
            if expected != actual:
                raise SystemExit(f"The regenerated palette bounds differ at row {index}.")
    print(result.stdout, end="", flush=True)
    print(f"The C++ replay reproduced all {len(recorded)} recorded palette rows.", flush=True)
