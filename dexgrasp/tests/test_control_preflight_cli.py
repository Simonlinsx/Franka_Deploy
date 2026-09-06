from pathlib import Path

from apps import control_preflight


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = (
    ROOT / "runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz"
)


def test_candidate_override_reports_exact_candidate51_target_offline(capsys):
    result = control_preflight.main(
        [
            "--snapshot",
            str(SNAPSHOT),
            "--selected-index",
            "51",
            "--strict-full-grasp",
        ]
    )

    assert result == 3
    output = capsys.readouterr().out
    assert "selected=51" in output
    assert "Inspire target=[0, 358, 799, 911, 922, 646]" in output
    assert "outside the commissioned q6 range 900..1000" in output
    assert "no exact commissioning evidence" in output


def test_candidate_override_requires_snapshot():
    try:
        control_preflight.main(["--selected-index", "51"])
    except ValueError as exc:
        assert "requires --snapshot" in str(exc)
    else:
        raise AssertionError("candidate override without a snapshot was accepted")
