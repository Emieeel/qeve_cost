#!/usr/bin/env python3

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import openfermion as of


def _load_text_operator(path: Path):
    """Load an OpenFermion operator from the plain-text format in this repo.

    These files look like:
      FermionOperator:\n
      <coeff> [0^ 1 2^ 3] +\n
    or:
      QubitOperator:\n
      (<coeff>+0j) [X0 Y1 Z2] +\n
    This intentionally avoids `openfermion.load_operator`, which treats paths as
    package test-data paths.
    """

    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty operator file: {path}")

    header = lines[0].strip()
    if header.startswith("FermionOperator"):
        kind = "fermion"
        op = of.FermionOperator()
    elif header.startswith("QubitOperator"):
        kind = "qubit"
        op = of.QubitOperator()
    else:
        raise ValueError(f"Unrecognized operator header in {path}: {header!r}")

    for raw in lines[1:]:
        line = raw.strip()
        if not line:
            continue
        if line.endswith("+"):
            line = line[:-1].strip()

        brk = line.find("[")
        if brk < 0:
            raise ValueError(f"Bad operator line (missing '[') in {path}: {raw!r}")

        coeff_str = line[:brk].strip()
        rest = line[brk:].strip()
        end = rest.find("]")
        if not rest.startswith("[") or end < 0:
            raise ValueError(f"Bad operator line (bad brackets) in {path}: {raw!r}")

        term_content = rest[1:end].strip()  # e.g. "X0 Y1" or "0^ 1"
        coeff = complex(coeff_str)

        if kind == "fermion":
            term = term_content if term_content else ()
            op += of.FermionOperator(term, coeff)
        else:
            term = term_content if term_content else ()
            op += of.QubitOperator(term, coeff)

    return op


def _compress_qubit_operator(op: of.QubitOperator, abs_tol: float) -> of.QubitOperator:
    if not isinstance(op, of.QubitOperator):
        raise TypeError(f"Expected QubitOperator, got {type(op)}")
    # Make a copy so we don't mutate callers' objects.
    out = of.QubitOperator()
    out.terms = dict(op.terms)
    out.compress(abs_tol=abs_tol)
    return out


def _term_key(term: Tuple[Tuple[int, str], ...]) -> str:
    if len(term) == 0:
        return "[]"
    return "[" + " ".join(f"{p}{a}" for (p, a) in term) + "]"


def _compare_qubit_operators(
    got: of.QubitOperator,
    expected: of.QubitOperator,
    *,
    atol: float,
    rtol: float,
    show: int,
) -> Tuple[bool, str]:
    got_terms: Dict[Tuple[Tuple[int, str], ...], complex] = got.terms
    exp_terms: Dict[Tuple[Tuple[int, str], ...], complex] = expected.terms

    all_terms = set(got_terms.keys()) | set(exp_terms.keys())

    worst: List[Tuple[float, Tuple[Tuple[int, str], ...], complex, complex]] = []
    max_err = 0.0
    bad = 0
    for term in all_terms:
        a = got_terms.get(term, 0.0)
        b = exp_terms.get(term, 0.0)
        err = abs(a - b)
        tol = atol + rtol * max(abs(a), abs(b))
        if err > tol:
            bad += 1
        if err > max_err:
            max_err = err
        worst.append((err, term, a, b))

    worst.sort(key=lambda x: x[0], reverse=True)

    summary_lines = [
        f"terms(got)={len(got_terms)} terms(expected)={len(exp_terms)}",
        f"max_abs_error={max_err:.3e}",
        f"mismatched_terms={bad}",
    ]

    if show > 0:
        summary_lines.append(f"top_{show}_diffs:")
        for err, term, a, b in worst[:show]:
            summary_lines.append(
                f"  {err:.3e}  {_term_key(term)}  got={a!r}  expected={b!r}"
            )

    passed = bad == 0
    return passed, "\n".join(summary_lines)


def _map_fermion_to_qubit(op: of.FermionOperator, mapping: str) -> of.QubitOperator:
    if mapping == "jordan_wigner":
        return of.jordan_wigner(op)
    if mapping == "bravyi_kitaev":
        return of.bravyi_kitaev(op)
    raise ValueError(f"Unknown mapping: {mapping}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a TC FermionOperator (sq_tc_*.data) maps to the provided "
            "QubitOperator (qubit_tc_*.data) under a chosen mapping."
        )
    )
    parser.add_argument(
        "--fermion",
        type=Path,
        required=True,
        help="Path to sq_tc_*.data (OpenFermion FermionOperator file)",
    )
    parser.add_argument(
        "--qubit",
        type=Path,
        required=True,
        help="Path to qubit_tc_*.data (OpenFermion QubitOperator file)",
    )
    parser.add_argument(
        "--mapping",
        choices=["auto", "jordan_wigner", "bravyi_kitaev"],
        default="auto",
        help="Which fermion→qubit mapping to test.",
    )
    parser.add_argument(
        "--compress",
        type=float,
        default=1e-12,
        help="Absolute tolerance for dropping tiny coefficients before compare.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=2e-8,
        help="Absolute tolerance for coefficient comparison.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="Relative tolerance for coefficient comparison.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=10,
        help="Show N largest coefficient diffs.",
    )

    args = parser.parse_args()

    fermion = _load_text_operator(args.fermion)
    expected_qubit = _compress_qubit_operator(_load_text_operator(args.qubit), args.compress)

    if not isinstance(fermion, of.FermionOperator):
        raise TypeError(
            f"{args.fermion} did not load as FermionOperator (got {type(fermion)})"
        )

    fermion.compress(abs_tol=args.compress)

    mappings: Iterable[str]
    if args.mapping == "auto":
        mappings = ["jordan_wigner", "bravyi_kitaev"]
    else:
        mappings = [args.mapping]

    best = None
    for mapping in mappings:
        got_qubit = _compress_qubit_operator(_map_fermion_to_qubit(fermion, mapping), args.compress)
        passed, summary = _compare_qubit_operators(
            got_qubit,
            expected_qubit,
            atol=args.atol,
            rtol=args.rtol,
            show=args.show,
        )
        header = f"=== mapping={mapping} ==="
        print(header)
        print(summary)
        print()

        # Track best by max error (lower is better) in auto mode.
        max_err_line = next((l for l in summary.splitlines() if l.startswith("max_abs_error=")), None)
        max_err = math.inf
        if max_err_line is not None:
            try:
                max_err = float(max_err_line.split("=", 1)[1])
            except ValueError:
                pass
        candidate = (passed, max_err, mapping)
        if best is None or candidate[1] < best[1]:
            best = candidate

        if passed and args.mapping != "auto":
            return 0

    if args.mapping == "auto" and best is not None:
        passed, _, mapping = best
        if passed:
            print(f"PASS (best mapping: {mapping})")
            return 0
        print(f"FAIL (best mapping tried: {mapping})")
        return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
