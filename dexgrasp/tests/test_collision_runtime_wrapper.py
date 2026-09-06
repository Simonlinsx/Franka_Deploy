from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_collision_wrapper_is_single_threaded_and_background_scheduled():
    script = (ROOT / "scripts/generate_installed_air_audit.sh").read_text(
        encoding="utf-8"
    )

    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        assert f"export {variable}=1" in script
    assert "/usr/bin/chrt --idle 0" in script
    assert "/usr/bin/ionice -c 3" in script
    assert "/usr/bin/nice -n 19" in script
    assert 'exec /usr/bin/chrt --idle 0' in script
