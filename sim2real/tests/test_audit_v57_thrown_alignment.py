from __future__ import annotations

from sim2real.diagnostics.audit_v57_thrown_alignment import build_alignment_audit
from sim2real.tasks.launcher import materialize_task_config


def test_alpha_half_static_alignment_audit_is_accepted_but_not_motion_authority():
    config, _metadata = materialize_task_config("thrown_object")
    report = build_alignment_audit(resolved_config=config)
    assert report["static_alignment_accepted"] is True
    assert report["robot_execution_authorized"] is False
    assert all(report["checks"].values())
    selected = report["v57"]["camera_visibility"]
    assert selected["target_center_vertices_inside"] == 8
    assert selected["target_center_box_fully_visible"] is True
    assert selected["object_support_depth_covered"] is True
    alpha_one = report["alpha_1_0_out_of_scope"]
    assert alpha_one["commissioned"] is False
    assert alpha_one["camera_visibility"]["target_center_vertices_inside"] == 6
