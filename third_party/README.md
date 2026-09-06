# Third-party checkouts

This directory owns external repositories and their project-local environments:

- `sam2/`: official SAM2 checkout and model checkpoints;
- `long_vos_clean/`: independent YOLO-World/long-VOS experiment workspace;
- `brainco-hand-sdk/`: vendor hand SDK checkout.
- `dex-retargeting/`: optional Inspire/MANO retargeting checkout;
- `wilor-mini/`: optional WiLoR hand-pose checkout and weights;
- `dexgrasp/`: AnyDexGrasp and MinkowskiEngine external checkouts used by the
  maintained `dexgrasp` integration;
- `environments/`: project-specific Python environments that must not be
  treated as maintained source.

Each checkout keeps its own license and Git history. The integration repository
must not vendor or rewrite those histories. Runtime paths should resolve through
the owning launcher or configuration rather than assume a top-level checkout.
Compatibility symlinks remain at `dexgrasp/third_party`,
`dexgrasp/.venv-anydex-official`, and
`examples/inspire_mano_pipeline/.venv` while existing tools migrate.
