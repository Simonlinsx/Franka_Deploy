from pathlib import Path

import sim2real.tasks.tabletop_demo as demo


def test_exact_action_demo_does_not_request_checkpoint_policy_io(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan_root = tmp_path / "plans"
    checkpoint_root = tmp_path / "checkpoints"
    plan_root.mkdir()
    checkpoint_root.mkdir()
    (plan_root / "cylinder_posy_low.zip").write_bytes(b"plan")
    (checkpoint_root / "v364_cylinder_+Y_0.02-0.20mps_inference.pt").write_bytes(
        b"checkpoint"
    )
    profile = tmp_path / "profile.json"
    profile.write_text("{}", encoding="utf-8")
    task_config = tmp_path / "tabletop.yaml"
    task_config.write_text("task: tabletop\n", encoding="utf-8")

    captured: list[str] = []

    def fake_replay_main(argv):
        captured.extend(argv)
        return 0

    monkeypatch.setattr(demo, "PLAN_ROOT", plan_root)
    monkeypatch.setattr(demo, "CHECKPOINT_ROOT", checkpoint_root)
    monkeypatch.setattr(
        demo,
        "materialize_task_config",
        lambda _task: (task_config, {}),
    )
    monkeypatch.setattr(demo, "replay_main", fake_replay_main)

    assert (
        demo.main(
            [
                "cylinder-posy-low",
                "--profile",
                str(profile),
                "--run-id",
                "tabletop-exact-action-test",
            ]
        )
        == 0
    )
    assert "--record-policy-io" not in captured
    assert "--record-video" in captured
    assert captured[captured.index("--actions") + 1] == str(
        (plan_root / "cylinder_posy_low.zip").resolve()
    )
    assert captured[captured.index("--object-text") + 1] == (
        "large pink foam cylinder"
    )
