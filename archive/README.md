# Archive

Files moved here are **not tests** or have **misleadingly weak assertions**.
They are preserved for reference but should NOT be mistaken for validation.

## Why archived

### `tests/test_phase6_level4.py`
- Claims to validate CAS(10,10) < 5 mHa target
- Actual assertion: `energy < hf_energy` (trivially true for any CI method)
- **No FCI reference, no comparison to Direct-CI baseline**
- Gives false confidence that Phase 6 target is met

### `tests/debug_connection_mismatch.py`
- Debug script (no pytest markers, no assertions)
- Bug it investigated was already fixed (PR C3)

### `tests/bench_get_connections.py`
- Performance benchmark (no assertions, only print statements)

### `tests/bench_precompute.py`
- Performance benchmark (no assertions, only print statements)

## Date archived: 2026-03-09
