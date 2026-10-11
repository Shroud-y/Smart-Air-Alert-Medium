"""Project-wide constants. All dates are UTC."""

HORIZON_MIN = 30  # forecast horizon H
GRID_STEP_MIN = 5  # step of the training / evaluation grid

# Training period variants compared on validation (README, decision 14).
TRAIN_START_FULL = "2022-03-15"
TRAIN_START_RECENT = "2024-07-01"

# Time split (README, decision 15). Intervals are [start, end).
TRAIN_END = "2026-01-01"
VAL_START, VAL_END = "2026-01-01", "2026-07-01"
TEST_START, TEST_END = "2026-07-01", "2026-09-07"
