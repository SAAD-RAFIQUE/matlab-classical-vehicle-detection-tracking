# Classical Vehicle Detection and Tracking in MATLAB

This repository contains my MATLAB implementation for detecting and tracking vehicles in road-scene images and videos. I kept the main pipeline classical on purpose: it uses image processing, HOG features, geometric checks, and motion tracking rather than a deep detector.

![Detection overview](assets/detection_overview.png)

The project grew out of experiments with difficult cases such as night scenes, rain, shadows, partial occlusion, and vehicles leaving the frame. The goal is not to present a finished production system. It is a readable baseline where the decisions made by the detector and tracker can be inspected and measured.

## What is in the pipeline?

- HOG descriptors and cosine-similarity checks
- Road and perspective constraints to reject unlikely candidates
- Shadow and morphology-based candidate generation
- IoU-based non-maximum suppression
- Kalman filtering for position and velocity
- Appearance-aware association between detections and tracks
- Camera-motion and ego-motion compensation
- Per-category and per-sequence evaluation

The main script is a classical computer-vision pipeline. It does not use a neural network. `hog_car_detector.m` is a separate HOG + linear SVM experiment included as a comparison point.

## Pipeline at a glance

```text
Image or video
     |
     +--> grayscale enhancement
     +--> HOG and appearance checks
     +--> shadow / morphology proposals
     +--> perspective filtering
     |
     v
IoU-based non-maximum suppression
     |
     v
Kalman prediction and track assignment
     |
     v
camera-motion compensation and annotated output
```

## Repository layout

```text
.
├── assets/                         # Selected output images and videos
├── data/
│   └── README.md                   # Where to place permitted datasets
├── docs/
│   └── architecture.md             # Design and evaluation notes
├── results/
│   ├── README.md
│   └── results.mat                 # Saved result structure from the experiment
├── src/
│   ├── main.m                      # End-to-end detection/tracking script
│   ├── compare_sobel_vs_hog_math.m # Classical-method comparison
│   ├── hog_car_detector.m          # Optional HOG + linear-SVM baseline
│   └── test_pipeline.m             # Small smoke test
└── .gitignore
```

## Requirements

- MATLAB R2023b or newer is recommended
- Image Processing Toolbox
- Computer Vision Toolbox
- Statistics and Machine Learning Toolbox for `hog_car_detector.m`
- Windows MATLAB is useful for the optional memory profiling calls

The raw image and video data is intentionally not included. The repository also does not depend on the original computer-specific paths. See [`data/README.md`](data/README.md) for the expected layout and the `KITTI_ROOT` option.

## Running it

1. Clone the repository and open it in MATLAB.
2. Put the image data in `data/raw_dataset_pictures/` and the video sequences in `data/raw_dataset_videos/`.
3. If KITTI tracking data is available, either put it under `data/kitti/` or set the `KITTI_ROOT` environment variable.
4. Run the main script:

```matlab
run(fullfile('src', 'main.m'))
```

Generated files go to `results/generated/`.

The two additional experiments can be run with:

```matlab
run(fullfile('src', 'compare_sobel_vs_hog_math.m'))
run(fullfile('src', 'hog_car_detector.m'))
```

## Example outputs

The checked-in files are outputs from the original MATLAB workflow and are included so the repository can be inspected without downloading the full dataset:

- [Detection overview](assets/detection_overview.png)
- [Normal scene](assets/1_normal_b1d7b3ac-995f9d8a.png)
- [Night scene](assets/2_night_b1c81faa-c80764c5.png)
- [Rain scene](assets/3_rain_b1cac6a7-04e33135.png)
- [Shadow scene](assets/4_shadow_b1cd1e94-549d0bfe.png)
- [Occlusion scene](assets/5_occlusion_b1c66a42-6f7d68ca.png)
- [Out-of-frame scene](assets/6_out_of_frame_b1d968b9-563405f4.png)
The video links below are lightweight previews of the generated tracking outputs so they are practical to view directly on GitHub:

- [Tracking sequence 0005 preview](assets/tracking_seq_0005_preview.mp4)
- [Tracking sequence 0011 preview](assets/tracking_seq_0011_preview.mp4)
- [Tracking sequence 0020 preview](assets/tracking_seq_0020_preview.mp4)

## Notes and limitations

The detector depends on lighting, camera geometry, segmentation quality, and the assumptions used to generate candidates. Some evaluation paths use annotated boxes for candidate verification, so the numbers should not be interpreted as a blind deployment benchmark. The implementation is best treated as an interpretable research baseline and a starting point for trying stronger learned detectors or better tracking models.

Only selected generated artifacts are included here. Raw datasets are not redistributed, and the repository contains no private credentials or company-internal material. If you use the code for a new experiment, rerun it on data you are allowed to use and record the exact settings.
