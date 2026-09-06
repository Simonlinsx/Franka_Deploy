# Observation

This package owns the camera-side data path used by sim-to-real execution:

- `model.py`: policy-resolution RGB-D, point-cloud, history, and proprioception;
- `pointcloud_filters.py`: pure point-cloud and mask candidate geometry;
- `camera_profile.py`: sealed task camera contracts;
- `roi_selector.py`: isolated camera-only object selection;
- `live_preview.py`: read-only live observation and inference preview;
- `visualization.py`: live RGB/mask/point-cloud rendering;
- `capture.py`: camera-only capture of deployment-exact RGB-D and masks.

None of these modules owns Franka or RH56 commands. Device writes remain under
`robot_control` and supervised composition remains under
`sim2real.runtime`.
