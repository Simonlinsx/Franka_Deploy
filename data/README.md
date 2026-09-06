# Data

This directory is the canonical home for runtime assets, recordings, and
curated non-source inputs. Maintained source code does not live here.

- `checkpoints/`: active model checkpoints, including the AnyDex weights. The
  internal `dexgrasp/weights` link remains for the external AnyDex runtime.
- `perception_corpus/`: recorded RGB-D cases and replay products.
- `runs/`: deployment videos, masks, force logs, telemetry, and audits. The
  `dexgrasp/runs` symlink preserves existing output paths.
- `archives/`: historical bundles, release snapshots, and import metadata that
  are not runtime inputs.

- `test_fixtures/`: small checkpoints, traces, and replay inputs referenced by
  automated tests, including the immutable V94 deployment bundle under
  `test_fixtures/sim2real/`.
- `demos/`: preserved simulation or real-world demonstration evidence used for
  comparison and analysis.
- `captures/`: manually retained camera snapshots.
- `logs/`: local diagnostic logs that are not part of a formal deployment run.
- `test_videos/`: local video staging; commissioned external-video manifests
  continue to pin their own absolute source paths and hashes.

New tools and documentation must use the canonical `data/...` paths. The old
root-level `ckpts` and `object_pcd_testdata` aliases have been retired.
