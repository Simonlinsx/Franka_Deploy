# Object point-cloud provider internals

`object_pcd_provider.py` remains the stable transactional owner and public
entry point. Its supporting modules are intentionally hardware-inert:

- `config.py` validates provider-local runtime configuration;
- `mask_geometry.py` contains pure binary-mask geometry and registration;
- `state.py` contains evidence, proof, continuity, and pending-commit records.

New camera, tracker, publication, or authority mutation stays in
`ObjectPCDProvider` until it has an explicit transaction boundary. Pure helpers
and immutable records belong in the supporting modules and must remain covered
by the provider replay/provenance tests.
