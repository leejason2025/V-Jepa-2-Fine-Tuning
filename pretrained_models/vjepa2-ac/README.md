---
license: mit
---
# Model Card for Model ID

This is a hosting of Facebook's VJEPA2-AC with a custom handler to be used with [Huggingface Endpoints](https://endpoints.huggingface.co). View the origional repository here: https://github.com/facebookresearch/vjepa2

- **Developed by:** [Meta]
- **Configured by:** [Skyler Wiernik]
- **Model type:** [Video encoder with action decoder]
- **License:** [MIT]

## Usage

Send a request body like
```
        "inputs": {
            "video": video_frames,
            "current_pose": current_pose_array
        }
```
where video is an array of base64 jpeg frames with the final frame being the goal and current_pose is an array of (x, y, z, rx, ry, rz, gripper).

## Model Card Contact
https://linkedin.com/in/skyler-wiernik