import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from magic_square_improved import MagicSolverImproved, SolverConfig


def make_config(tmp_path: Path, **kwargs) -> SolverConfig:
    defaults = dict(
        start_P=1,
        max_P=1,
        cell_max=5,
        time_per_P=0.5,
        save_interval=0.1,
        state_file=tmp_path / "state.json",
        work_dir=tmp_path,
        gen_mode="expdist",
        allow_duplicates_in_row=True,
        allow_duplicates_global=True,
        aggressive_pruning=False,
        workers=1,
        max_factor_tuples=1000,
        max_combos_per_sum=1000,
        max_perms_per_combo=1000,
        verbose=False,
    )
    defaults.update(kwargs)
    return SolverConfig(**defaults)


def test_factorize_basic():
    factors = MagicSolverImproved.factorize(360)
    assert factors == {2: 3, 3: 2, 5: 1}


def test_divisors_limited_basic():
    divs = MagicSolverImproved.divisors_limited(12, 6)
    assert divs == [1, 2, 3, 4, 6]


def test_generate_rows_expdist_small():
    config = make_config(Path("."), cell_max=3, allow_duplicates_in_row=True)
    solver = MagicSolverImproved(config)
    rows = solver.generate_rows(2)
    assert (1, 1, 1, 1, 2) in rows


def test_generate_rows_divcomb_small():
    config = make_config(Path("."), cell_max=3, gen_mode="divcomb", allow_duplicates_in_row=True)
    solver = MagicSolverImproved(config)
    rows = solver.generate_rows(3)
    assert (1, 1, 1, 1, 3) in rows


def test_duplicate_flags_affect_row_generation(tmp_path: Path):
    config = make_config(tmp_path, cell_max=3, allow_duplicates_in_row=False)
    solver = MagicSolverImproved(config)
    rows = solver.generate_rows(2)
    assert all(len(set(row)) == 5 for row in rows)


def test_integration_small_solution(tmp_path: Path):
    config = make_config(
        tmp_path,
        start_P=1,
        max_P=1,
        cell_max=1,
        allow_duplicates_in_row=True,
        allow_duplicates_global=True,
    )
    solver = MagicSolverImproved(config)
    result = solver.solve()
    assert result is not None
    assert all(row == (1, 1, 1, 1, 1) for row in result)


def test_state_save_and_resume(tmp_path: Path):
    config = make_config(
        tmp_path,
        start_P=2,
        max_P=2,
        cell_max=5,
        time_per_P=0.0,
        save_interval=0.0,
    )
    solver = MagicSolverImproved(config)
    solver.solve()
    assert config.state_file.exists()
    with config.state_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["state"]["current_P"] == 2

    config2 = make_config(
        tmp_path,
        start_P=2,
        max_P=2,
        cell_max=5,
        time_per_P=0.1,
        save_interval=0.0,
    )
    solver2 = MagicSolverImproved(config2)
    solver2.solve()


def test_duplicate_flags_global(tmp_path: Path):
    config = make_config(
        tmp_path,
        start_P=1,
        max_P=1,
        cell_max=1,
        allow_duplicates_in_row=True,
        allow_duplicates_global=False,
    )
    solver = MagicSolverImproved(config)
    result = solver.solve()
    assert result is None
