A 3d convstack network trained on the FIB-25 validation1 volume.

Use the following inference settings with this checkpoint:

```
  model_name: "convstack_3d.ConvStack3DFFNModel"
  model_args: "{\"depth\": 12, \"fov_size\": [33, 33, 33], \"deltas\": [8, 8, 8]}"
  inference_options {
    init_activation: 0.95
    pad_value: 0.05
    move_threshold: 0.9
  }
```
