"""CLI de promocion: ``poetry run fraud-promote-winner`` o ``python -m promotion``."""

from __future__ import annotations

import argparse
from pathlib import Path

from compat import install_legacy_pickle_aliases
from promotion.pipeline import promote_winner


def main() -> None:
    install_legacy_pickle_aliases()
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--config", type=Path, default=Path("configs/promotion/v1.yml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument("--promoted-root", type=Path, default=Path("artifacts/promoted"))
    parser.add_argument("--override-winner-run-id", type=str, default=None)
    parser.add_argument("--force-test-rerun", action="store_true")
    args = parser.parse_args()

    promote_winner(
        runs_dir=args.runs_dir,
        config_path=args.config,
        data_root=args.data_root,
        configs_root=args.configs_root,
        promoted_root=args.promoted_root,
        override_winner_run_id=args.override_winner_run_id,
        force_test_rerun=args.force_test_rerun,
    )


if __name__ == "__main__":
    main()
