"""Precompute equivariant ST+TCN + TemporalHead target cache inside ignored evaluation/reports."""
from possession.st_tcn_target import compute_and_cache_st_tcn_target


def main():
    return compute_and_cache_st_tcn_target()


if __name__ == "__main__":
    raise SystemExit(main())

