"""
Magic Square 5x5 Solver (Improved)
=================================

This module implements a deterministic, resumable solver for a 5x5 additive–
multiplied magic square. Every row, column, and both diagonals must share the
same sum S and the same product P. The solver iterates P values and tries to
assemble five candidate rows that satisfy all column/diagonal constraints.

README / Usage
--------------
The module is runnable as a script or importable as a library.

Colab example:
    from google.colab import drive
    drive.mount('/content/drive')

    !python magic_square_improved.py \
      --work-dir /content/drive/MyDrive/magic_square_5x5 \
      --start-P 120 \
      --cell-max 200 \
      --time-per-P 1800 \
      --save-interval 600 \
      --gen-mode expdist \
      --workers 4 \
      --verbose

    !python magic_square_improved.py \
      --work-dir /content/drive/MyDrive/magic_square_5x5 \
      --state-file magic_state.json

Local example:
    python magic_square_improved.py --start-P 120 --cell-max 50 --time-per-P 60
    python -m magic_square_improved --state-file magic_state.json

The solver is deterministic given the same inputs and caps. State is saved
atomically to allow resuming after timeouts or interruptions.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import logging
import math
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

DEFAULT_WORK_DIR = "/content/drive/MyDrive/magic_square_5x5"
DEFAULT_STATE_FILE = "magic_state.json"

MAX_FACTOR_TUPLES = 50_000
MAX_COMBOS_PER_SUM = 10_000
MAX_PERMS_PER_COMBO = 2_000

LOGGER = logging.getLogger(__name__)

Row = Tuple[int, int, int, int, int]
Grid = List[Row]


@dataclass(frozen=True)
class SolverConfig:
    start_P: int
    max_P: int
    cell_max: int
    time_per_P: float
    save_interval: float
    state_file: Path
    work_dir: Path
    gen_mode: str
    allow_duplicates_in_row: bool
    allow_duplicates_global: bool
    aggressive_pruning: bool
    workers: int
    max_factor_tuples: int
    max_combos_per_sum: int
    max_perms_per_combo: int
    verbose: bool


@dataclass
class SolverState:
    current_P: int
    sum_index: int
    row_indices: List[int]
    placed_rows: List[Row]
    last_save_time: float

    def to_json(self) -> Dict[str, object]:
        return {
            "current_P": self.current_P,
            "sum_index": self.sum_index,
            "row_indices": self.row_indices,
            "placed_rows": self.placed_rows,
        }

    @classmethod
    def from_json(cls, payload: Dict[str, object], now: float) -> "SolverState":
        return cls(
            current_P=int(payload["current_P"]),
            sum_index=int(payload["sum_index"]),
            row_indices=list(payload["row_indices"]),
            placed_rows=[tuple(row) for row in payload["placed_rows"]],
            last_save_time=now,
        )


class MagicSolverImproved:
    """Improved solver with deterministic search and resumable state."""

    def __init__(self, config: SolverConfig) -> None:
        self.config = config
        self.current_P = config.start_P

    @staticmethod
    def factorize(n: int) -> Dict[int, int]:
        """Return the prime factorization of n as a {prime: exponent} dict."""
        if n <= 0:
            raise ValueError("n must be positive")
        factors: Dict[int, int] = {}
        remainder = n
        count = 0
        while remainder % 2 == 0:
            count += 1
            remainder //= 2
        if count:
            factors[2] = count
        p = 3
        while p * p <= remainder:
            count = 0
            while remainder % p == 0:
                count += 1
                remainder //= p
            if count:
                factors[p] = count
            p += 2
        if remainder > 1:
            factors[remainder] = 1
        return factors

    @staticmethod
    def divisors_limited(n: int, cell_max: int) -> List[int]:
        """Return sorted divisors of n that are within [1, cell_max]."""
        if n <= 0:
            return []
        divisors: Set[int] = set()
        limit = int(math.isqrt(n))
        for i in range(1, limit + 1):
            if n % i == 0:
                if i <= cell_max:
                    divisors.add(i)
                other = n // i
                if other <= cell_max:
                    divisors.add(other)
        return sorted(divisors)

    def _generate_expdist_rows(self, P: int) -> Iterator[Row]:
        factors = self.factorize(P)
        primes = sorted(factors.keys())
        exponents = [factors[p] for p in primes]
        cap = self.config.max_factor_tuples
        count = 0

        def distribute_exp(exp: int, slots: int) -> List[Tuple[int, ...]]:
            if slots == 1:
                return [(exp,)]
            result: List[Tuple[int, ...]] = []
            for i in range(exp + 1):
                for rest in distribute_exp(exp - i, slots - 1):
                    result.append((i,) + rest)
            return result

        exp_distributions: List[List[Tuple[int, ...]]] = [
            distribute_exp(exp, 5) for exp in exponents
        ]

        for exp_tuple_set in itertools.product(*exp_distributions):
            values = [1] * 5
            for prime, exp_tuple in zip(primes, exp_tuple_set):
                for idx, exp in enumerate(exp_tuple):
                    values[idx] *= prime**exp
                    if values[idx] > self.config.cell_max:
                        break
                else:
                    continue
                break
            else:
                row = tuple(values)
                if (not self.config.allow_duplicates_in_row) and len(set(row)) < 5:
                    continue
                yield row
                count += 1
                if count >= cap:
                    return

    def _generate_divcomb_rows(self, P: int) -> Iterator[Row]:
        divisors = self.divisors_limited(P, self.config.cell_max)
        if not divisors:
            return iter(())
        allow_dup = self.config.allow_duplicates_in_row
        if allow_dup:
            combos = itertools.combinations_with_replacement(divisors, 5)
        else:
            combos = itertools.combinations(divisors, 5)
        max_perms = self.config.max_perms_per_combo
        max_combos = self.config.max_factor_tuples
        combo_count = 0

        for combo in combos:
            if math.prod(combo) != P:
                continue
            combo_count += 1
            if combo_count > max_combos:
                return
            perms = sorted(set(itertools.permutations(combo)))
            for perm_idx, perm in enumerate(perms):
                if perm_idx >= max_perms:
                    break
                row = tuple(perm)
                yield row

    def generate_rows(self, P: int) -> List[Row]:
        """Generate candidate rows deterministically for product P."""
        if P <= 0:
            return []
        if self.config.gen_mode == "expdist":
            generator = self._generate_expdist_rows(P)
        else:
            generator = self._generate_divcomb_rows(P)
        rows = list(generator)
        rows.sort()
        return rows

    def group_rows_by_sum(self, rows: Sequence[Row]) -> Dict[int, List[Row]]:
        grouped: Dict[int, List[Row]] = defaultdict(list)
        for row in rows:
            grouped[sum(row)].append(row)
        for total in grouped:
            grouped[total].sort()
            if len(grouped[total]) > self.config.max_combos_per_sum:
                grouped[total] = grouped[total][: self.config.max_combos_per_sum]
        return dict(grouped)

    def _feasible_products(
        self,
        current_prod: int,
        rows_left: int,
        used_global: Set[int],
    ) -> bool:
        """Conservative feasibility check for remaining product factors."""
        if current_prod == 0 or self.config.cell_max <= 0:
            return False
        if self.current_P % current_prod != 0:
            return False
        if rows_left == 0:
            return current_prod == self.current_P
        req = self.current_P // current_prod
        divisors = self.divisors_limited(req, self.config.cell_max)
        if self.config.allow_duplicates_global:
            return bool(divisors)
        available = [d for d in divisors if d not in used_global]
        return len(available) >= rows_left

    def _feasible_sums(
        self,
        current_sum: int,
        cells_left: int,
        target_sum: int,
    ) -> bool:
        """Aggressive-only sum feasibility based on min/max remaining values."""
        if not self.config.aggressive_pruning:
            return current_sum <= target_sum
        min_possible = current_sum + cells_left * 1
        max_possible = current_sum + cells_left * self.config.cell_max
        return min_possible <= target_sum <= max_possible

    def _attempt_sum(
        self,
        target_sum: int,
        rows: Sequence[Row],
        state: SolverState,
        start_time: float,
    ) -> Optional[Grid]:
        depth = len(state.placed_rows)
        row_indices = state.row_indices or [0] * 5
        placed = list(state.placed_rows)
        used_global: Set[int] = set()
        if not self.config.allow_duplicates_global:
            for row in placed:
                used_global.update(row)

        col_sums = [0] * 5
        col_prods = [1] * 5
        diag_sum = 0
        diag_prod = 1
        anti_sum = 0
        anti_prod = 1

        for r_index, row in enumerate(placed):
            for c_index, value in enumerate(row):
                col_sums[c_index] += value
                col_prods[c_index] *= value
                if r_index == c_index:
                    diag_sum += value
                    diag_prod *= value
                if r_index + c_index == 4:
                    anti_sum += value
                    anti_prod *= value

        while depth >= 0:
            if time.time() - start_time > self.config.time_per_P:
                state.row_indices = row_indices
                state.placed_rows = placed
                return None

            if depth == 5:
                if (
                    all(value == target_sum for value in col_sums)
                    and all(value == self.current_P for value in col_prods)
                    and diag_sum == target_sum
                    and anti_sum == target_sum
                    and diag_prod == self.current_P
                    and anti_prod == self.current_P
                ):
                    return placed
                depth -= 1
                if depth < 0:
                    break
                last_row = placed.pop()
                for i in range(5):
                    col_sums[i] -= last_row[i]
                    col_prods[i] //= last_row[i]
                diag_sum -= last_row[depth]
                diag_prod //= last_row[depth]
                anti_sum -= last_row[4 - depth]
                anti_prod //= last_row[4 - depth]
                if not self.config.allow_duplicates_global:
                    for value in last_row:
                        used_global.remove(value)
                row_indices[depth] = 0
                continue

            start_idx = row_indices[depth]
            progressed = False
            for idx in range(start_idx, len(rows)):
                row = rows[idx]
                if not self.config.allow_duplicates_global:
                    if any(value in used_global for value in row):
                        continue
                # sum check per column
                next_col_sums = [col_sums[i] + row[i] for i in range(5)]
                if any(s > target_sum for s in next_col_sums):
                    continue

                # diag sums
                next_diag_sum = diag_sum + row[depth]
                next_anti_sum = anti_sum + row[4 - depth]
                if next_diag_sum > target_sum or next_anti_sum > target_sum:
                    continue

                rows_left = 4 - depth
                if not all(
                    self._feasible_sums(next_col_sums[i], rows_left, target_sum)
                    for i in range(5)
                ):
                    continue
                if not self._feasible_sums(next_diag_sum, rows_left, target_sum):
                    continue
                if not self._feasible_sums(next_anti_sum, rows_left, target_sum):
                    continue

                # product feasibility
                next_col_prods = [col_prods[i] * row[i] for i in range(5)]
                next_diag_prod = diag_prod * row[depth]
                next_anti_prod = anti_prod * row[4 - depth]

                if not all(
                    self._feasible_products(next_col_prods[i], rows_left, used_global)
                    for i in range(5)
                ):
                    continue
                if not self._feasible_products(next_diag_prod, rows_left, used_global):
                    continue
                if not self._feasible_products(next_anti_prod, rows_left, used_global):
                    continue

                # accept row
                row_indices[depth] = idx + 1
                placed.append(row)
                for i in range(5):
                    col_sums[i] = next_col_sums[i]
                    col_prods[i] = next_col_prods[i]
                diag_sum = next_diag_sum
                anti_sum = next_anti_sum
                diag_prod = next_diag_prod
                anti_prod = next_anti_prod
                if not self.config.allow_duplicates_global:
                    used_global.update(row)

                depth += 1
                if depth < 5 and len(row_indices) <= depth:
                    row_indices.append(0)
                progressed = True
                break

            if progressed:
                continue

            # backtrack
            if depth == 0:
                break
            row_indices[depth] = 0
            depth -= 1
            last_row = placed.pop()
            for i in range(5):
                col_sums[i] -= last_row[i]
                col_prods[i] //= last_row[i]
            diag_sum -= last_row[depth]
            diag_prod //= last_row[depth]
            anti_sum -= last_row[4 - depth]
            anti_prod //= last_row[4 - depth]
            if not self.config.allow_duplicates_global:
                for value in last_row:
                    used_global.remove(value)

        state.row_indices = row_indices
        state.placed_rows = placed
        return None

    def _ensure_work_dir(self) -> None:
        self.config.work_dir.mkdir(parents=True, exist_ok=True)

    def save_state(self, state: SolverState) -> None:
        self._ensure_work_dir()
        config_payload = dataclasses.asdict(self.config)
        for key, value in config_payload.items():
            if isinstance(value, Path):
                config_payload[key] = str(value)
        payload = {"config": config_payload, "state": state.to_json()}
        tmp_fd, tmp_path = tempfile.mkstemp(dir=self.config.work_dir)
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_path, self.config.state_file)
        state.last_save_time = time.time()
        LOGGER.info("State saved to %s", self.config.state_file)

    @classmethod
    def load_state(cls, config: SolverConfig) -> Optional[SolverState]:
        if not config.state_file.exists():
            return None
        with config.state_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        state_payload = payload.get("state")
        if not state_payload:
            return None
        return SolverState.from_json(state_payload, time.time())

    def solve(self) -> Optional[Grid]:
        state = self.load_state(self.config) or SolverState(
            current_P=self.config.start_P,
            sum_index=0,
            row_indices=[0] * 5,
            placed_rows=[],
            last_save_time=time.time(),
        )
        for P in range(state.current_P, self.config.max_P + 1):
            if P <= 0:
                continue
            start_time = time.time()
            self.current_P = P
            LOGGER.info("Starting P=%s", P)
            rows = self.generate_rows(P)
            if not rows:
                LOGGER.info("No candidate rows for P=%s", P)
                state.sum_index = 0
                continue

            if self.config.workers > 1:
                with Pool(processes=self.config.workers) as pool:
                    row_sums = list(pool.imap(sum, rows))
                rows = [row for _, row in sorted(zip(row_sums, rows))]

            grouped = self.group_rows_by_sum(rows)
            sums = sorted(grouped.keys())
            for sum_idx in range(state.sum_index, len(sums)):
                target_sum = sums[sum_idx]
                LOGGER.info("Trying P=%s with S=%s", P, target_sum)
                state.sum_index = sum_idx
                state.row_indices = [0] * 5
                state.placed_rows = []
                result = self._attempt_sum(target_sum, grouped[target_sum], state, start_time)
                if result:
                    LOGGER.info("Solution found for P=%s S=%s", P, target_sum)
                    return result
                if time.time() - start_time > self.config.time_per_P:
                    self.save_state(state)
                    return None
                if time.time() - state.last_save_time > self.config.save_interval:
                    self.save_state(state)

            state.current_P = P + 1
            state.sum_index = 0
            state.row_indices = [0] * 5
            state.placed_rows = []
            if time.time() - state.last_save_time > self.config.save_interval:
                self.save_state(state)

        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Improved 5x5 magic square solver")
    parser.add_argument("--start-P", type=int, default=120)
    parser.add_argument("--max-P", type=int, default=120)
    parser.add_argument("--cell-max", type=int, default=200)
    parser.add_argument("--time-per-P", type=float, default=1800)
    parser.add_argument("--save-interval", type=float, default=600)
    parser.add_argument("--state-file", type=str, default=DEFAULT_STATE_FILE)
    parser.add_argument("--work-dir", type=str, default=DEFAULT_WORK_DIR)
    parser.add_argument("--gen-mode", choices=["expdist", "divcomb"], default="expdist")
    parser.add_argument("--allow-duplicates-in-row", action="store_true")
    parser.add_argument("--allow-duplicates-global", action="store_true")
    parser.add_argument("--max-factor-tuples", type=int, default=MAX_FACTOR_TUPLES)
    parser.add_argument("--max-combos-per-sum", type=int, default=MAX_COMBOS_PER_SUM)
    parser.add_argument("--max-perms-per-combo", type=int, default=MAX_PERMS_PER_COMBO)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--aggressive-pruning", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> SolverConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    work_dir = Path(args.work_dir)
    state_file = Path(args.state_file)
    if not state_file.is_absolute():
        state_file = work_dir / state_file
    return SolverConfig(
        start_P=args.start_P,
        max_P=args.max_P,
        cell_max=args.cell_max,
        time_per_P=args.time_per_P,
        save_interval=args.save_interval,
        state_file=state_file,
        work_dir=work_dir,
        gen_mode=args.gen_mode,
        allow_duplicates_in_row=args.allow_duplicates_in_row,
        allow_duplicates_global=args.allow_duplicates_global,
        aggressive_pruning=args.aggressive_pruning,
        workers=max(1, args.workers),
        max_factor_tuples=args.max_factor_tuples,
        max_combos_per_sum=args.max_combos_per_sum,
        max_perms_per_combo=args.max_perms_per_combo,
        verbose=args.verbose,
    )


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = parse_args(argv)
    configure_logging(config.verbose)
    solver = MagicSolverImproved(config)
    result = solver.solve()
    if result:
        print("Solution found:")
        for row in result:
            print(" ".join(str(v) for v in row))
    else:
        print("No solution found in the specified range.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
