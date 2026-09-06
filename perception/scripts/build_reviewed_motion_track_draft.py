#!/usr/bin/env python3
"""Build a review-only dense bbox/centroid draft and contact sheets.

This tool never opens camera, Franka or RH56 interfaces.  Candidate-derived
records are deliberately marked ``needs_human_review`` and cannot activate the
formal evaluator gate until a person reviews/corrects every contact-sheet tile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_pcd.evaluation.guarded_v2_offline import (  # noqa: E402
    REVIEWED_MOTION_TRACK_SCHEMA,
    REVIEWED_MOTION_TRACK_THRESHOLDS,
    OfflineBenchmarkError,
    _bbox,
    _centroid,
    _inside_source_intervals,
    _load_sparse_ground_truth,
    _load_sparse_ground_truth_manifest,
    _reviewed_source_intervals,
    _sha256_file,
    _video_sampling_contract,
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT / "configs" / "guarded_v2_benchmark_long_vos_baseline.json"
)


def _read_source_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    wanted = set(indices)
    result: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OfflineBenchmarkError(f"cannot open video: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for source_frame in range(frame_count):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise OfflineBenchmarkError(
                    f"video ended at source frame {source_frame}: {path}"
                )
            if source_frame in wanted:
                result[source_frame] = frame
            if len(result) == len(wanted):
                break
    finally:
        capture.release()
    if set(result) != wanted:
        raise OfflineBenchmarkError(
            f"source frames missing: {sorted(wanted.difference(result))}"
        )
    return result


def _draw_tile(
    frame: np.ndarray,
    *,
    bbox_xyxy: list[float],
    centroid_xy: list[float],
    source_frame: int,
    reviewed: bool,
) -> np.ndarray:
    tile = cv2.resize(frame, (424, 240), interpolation=cv2.INTER_AREA)
    scale_x = tile.shape[1] / float(frame.shape[1])
    scale_y = tile.shape[0] / float(frame.shape[0])
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = centroid_xy
    colour = (0, 255, 255) if reviewed else (255, 0, 255)
    cv2.rectangle(
        tile,
        (int(round(x1 * scale_x)), int(round(y1 * scale_y))),
        (int(round(x2 * scale_x)), int(round(y2 * scale_y))),
        colour,
        2,
    )
    cv2.circle(
        tile,
        (int(round(cx * scale_x)), int(round(cy * scale_y))),
        4,
        colour,
        -1,
    )
    status = "REVIEWED" if reviewed else "NEEDS HUMAN REVIEW"
    cv2.putText(
        tile,
        f"source={source_frame} {status}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        colour,
        2,
        cv2.LINE_AA,
    )
    return tile


def _write_contact_sheets(
    frames: dict[int, np.ndarray], records: list[dict], output_dir: Path
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=False)
    rows, columns = 3, 4
    tile_height, tile_width = 240, 424
    paths: list[str] = []
    for offset in range(0, len(records), rows * columns):
        canvas = np.zeros(
            (rows * tile_height, columns * tile_width, 3), dtype=np.uint8
        )
        for local_index, record in enumerate(records[offset : offset + rows * columns]):
            row, column = divmod(local_index, columns)
            source_frame = int(record["source_frame"])
            tile = _draw_tile(
                frames[source_frame],
                bbox_xyxy=record["bbox_xyxy"],
                centroid_xy=record["centroid_xy"],
                source_frame=source_frame,
                reviewed=record["review_status"] in {
                    "human_confirmed",
                    "independent_visual_review",
                },
            )
            y1, x1 = row * tile_height, column * tile_width
            canvas[y1 : y1 + tile_height, x1 : x1 + tile_width] = tile
        path = output_dir / f"contact_sheet_{offset // (rows * columns) + 1:02d}.png"
        if not cv2.imwrite(str(path), canvas):
            raise OfflineBenchmarkError(f"failed to write contact sheet: {path}")
        paths.append(str(path))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a non-authoritative every-frame reviewed-motion draft from "
            "existing sparse human masks plus a guarded_v2 candidate export."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--case", default="fast_green_ball_entry")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-sheet-dir", type=Path, default=None)
    args = parser.parse_args()

    try:
        manifest_path = args.manifest.expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        case = next(
            (
                dict(item)
                for item in manifest.get("cases") or []
                if str(item.get("name", "")) == str(args.case)
            ),
            None,
        )
        if case is None:
            raise OfflineBenchmarkError(f"unknown case={args.case!r}")
        sampling = _video_sampling_contract(case, manifest_path.parent)
        intervals = _reviewed_source_intervals(
            case,
            "reviewed_visible_source_intervals",
            sampled_source_frames=sampling["sampled_source_indices"],
        )
        reviewed_sources = [
            int(source)
            for source in sampling["sampled_source_indices"]
            if _inside_source_intervals(source, intervals)
        ]
        sparse_by_case, _sparse_manifest_path = _load_sparse_ground_truth_manifest(
            manifest.get("sparse_ground_truth_manifest"), manifest_path.parent
        )
        sparse, _sparse_provenance = _load_sparse_ground_truth(
            dict(sparse_by_case.get(str(args.case)) or {}),
            manifest_path.parent,
            sampled_source_frames=set(sampling["sampled_source_indices"]),
            expected_source_video=sampling["video"],
        )
        candidate_masks = (
            args.candidate_root.expanduser().resolve()
            / str(args.case)
            / "masks"
        )
        if not candidate_masks.is_dir():
            raise OfflineBenchmarkError(
                f"candidate masks directory does not exist: {candidate_masks}"
            )
        source_to_position = {
            int(source): position
            for position, source in enumerate(sampling["sampled_source_indices"])
        }
        candidate_proposals: dict[int, np.ndarray] = {}
        for source_frame in reviewed_sources:
            position = source_to_position[source_frame]
            mask_path = candidate_masks / f"{position:06d}.png"
            mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_image is None:
                raise OfflineBenchmarkError(
                    f"cannot read candidate proposal mask: {mask_path}"
                )
            candidate_proposals[source_frame] = mask_image > 127
        records: list[dict] = []
        for source_frame in reviewed_sources:
            sparse_label = sparse.get(source_frame)
            if sparse_label is not None:
                mask = np.asarray(sparse_label["mask"], dtype=bool)
                metadata = dict(sparse_label["metadata"])
                review_status = str(metadata["review_status"])
                label_method = str(metadata["label_method"])
                review_notes = str(metadata["review_notes"])
                proposal_source = "human_sparse_ground_truth"
            else:
                mask = candidate_proposals[source_frame]
                proposal_source = "guarded_v2_candidate"
                if _bbox(mask) is None:
                    nonempty_sources = [
                        source
                        for source, proposal in candidate_proposals.items()
                        if _bbox(proposal) is not None
                    ]
                    if not nonempty_sources:
                        raise OfflineBenchmarkError(
                            "candidate export has no non-empty reviewed-visible "
                            "geometry proposal"
                        )
                    nearest_source = min(
                        nonempty_sources,
                        key=lambda source: abs(
                            source_to_position[source]
                            - source_to_position[source_frame]
                        ),
                    )
                    mask = candidate_proposals[nearest_source]
                    proposal_source = (
                        "nearest_nonempty_guarded_v2_candidate_source_"
                        f"{nearest_source}"
                    )
                review_status = "needs_human_review"
                label_method = "guarded_v2 candidate bbox/centroid proposal"
                review_notes = (
                    "DRAFT ONLY: compare the native RGB frame and contact-sheet "
                    "overlay, then correct geometry and mark human_confirmed."
                )
            bbox = _bbox(mask)
            centroid = _centroid(mask)
            if bbox is None or centroid is None:
                raise OfflineBenchmarkError(
                    f"source_frame={source_frame} has an empty geometry proposal"
                )
            records.append(
                {
                    "source_frame": source_frame,
                    "bbox_xyxy": [int(value) for value in bbox],
                    "centroid_xy": [float(value) for value in centroid],
                    "review_status": review_status,
                    "label_method": label_method,
                    "review_notes": review_notes,
                    "proposal_source": proposal_source,
                }
            )
        payload = {
            "schema": REVIEWED_MOTION_TRACK_SCHEMA,
            "case": str(args.case),
            "source_video_sha256": sampling["sha256"],
            "source_video_fps_hz": sampling["fps_hz"],
            "source_video_frame_count": sampling["frame_count"],
            "source_video_size_wh": sampling["size_wh"],
            "source_start_frame": int(case.get("source_start_frame", 0)),
            "evaluation_fps_hz": sampling["evaluation_fps_hz"],
            "sampled_source_indices": sampling["sampled_source_indices"],
            "reviewed_visible_source_intervals": [list(item) for item in intervals],
            "reviewed_visible_sampled_source_indices": reviewed_sources,
            "thresholds": dict(REVIEWED_MOTION_TRACK_THRESHOLDS),
            "records": records,
            "draft_notice": (
                "This artifact is not formal evidence while any record has "
                "review_status=needs_human_review. Do not add it to the benchmark "
                "manifest until every tile is visually reviewed."
            ),
        }
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        frames = _read_source_frames(sampling["video"], reviewed_sources)
        contact_dir = (
            args.contact_sheet_dir.expanduser().resolve()
            if args.contact_sheet_dir is not None
            else output.with_name(output.stem + "_contact_sheets")
        )
        sheets = _write_contact_sheets(frames, records, contact_dir)
        print(
            json.dumps(
                {
                    "result": "DRAFT_NEEDS_HUMAN_REVIEW",
                    "output": str(output),
                    "output_sha256": _sha256_file(output),
                    "record_count": len(records),
                    "already_human_reviewed": sum(
                        record["review_status"] != "needs_human_review"
                        for record in records
                    ),
                    "needs_human_review": sum(
                        record["review_status"] == "needs_human_review"
                        for record in records
                    ),
                    "contact_sheets": sheets,
                    "hardware_interfaces_opened": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (
        FileExistsError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        OfflineBenchmarkError,
    ) as exc:
        print(f"reviewed motion draft: FAILED: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
