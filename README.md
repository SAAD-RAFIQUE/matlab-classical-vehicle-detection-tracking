# Classical Vehicle Detection and Tracking in MATLAB

<p align="center">
  <img src="assets/Detection-Overview.png" alt="Representative detection results across several road-scene conditions" width="920">
</p>

<p align="center">
  <strong>An interpretable MATLAB baseline for vehicle detection, multi-object tracking, and failure-case analysis.</strong><br>
  Classical vision methods, explicit geometry, and motion models — with no neural detector in the main pipeline.
</p>

## At a Glance

| Input | Processing | Output |
| --- | --- | --- |
| Road-scene images and video sequences | Image preparation, candidate generation, HOG and appearance checks, geometry, non-maximum suppression, and Kalman tracking | Annotated frames, track IDs, trajectories, saved results, and evaluation summaries |

This repository contains an intentionally classical computer-vision workflow. The emphasis is on making each decision visible: where candidates come from, why boxes are rejected, how tracks are assigned, and where the method begins to fail.

## Pipeline

<p align="center">
  <img src="assets/Pipeline-Diagram.svg" alt="Pipeline from road-scene frames to vehicle tracks and evaluation" width="1100">
</p>

The main workflow combines:

- shadow- and morphology-based candidate generation;
- HOG descriptors and appearance-similarity checks;
- road and perspective constraints;
- IoU-based non-maximum suppression;
- Kalman filtering for position and velocity;
- appearance-aware association between detections and tracks;
- camera- and ego-motion compensation; and
- per-category and per-sequence evaluation.

The main script is a classical computer-vision pipeline and does not use a neural network. `hog_car_detector.m` is a separate HOG + linear SVM experiment included as a comparison point.

## Results Gallery

The images below are representative outputs from the MATLAB workflow. Together, they cover the conditions that matter when inspecting a detector rather than viewing only one favorable example.

<table>
  <tr>
    <td align="center"><img src="assets/Normal-Scene.png" alt="Normal road scene" width="280"><br><sub>Normal Scene</sub></td>
    <td align="center"><img src="assets/Night-Scene.png" alt="Night road scene" width="280"><br><sub>Night Scene</sub></td>
    <td align="center"><img src="assets/Rain-Scene.png" alt="Rainy road scene" width="280"><br><sub>Rain Scene</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/Shadow-Scene.png" alt="Road scene with shadows" width="280"><br><sub>Shadow Scene</sub></td>
    <td align="center"><img src="assets/Occlusion-Scene.png" alt="Partially occluded vehicles" width="280"><br><sub>Occlusion Scene</sub></td>
    <td align="center"><img src="assets/Out-of-Frame-Scene.png" alt="Vehicle leaving the frame" width="280"><br><sub>Out-of-Frame Scene</sub></td>
  </tr>
</table>

### Tracking Previews

The repository also contains lightweight previews of three generated tracking sequences. They are intentionally downsampled so that they remain practical to open directly from GitHub.

- [Tracking Sequence 0005 Preview](assets/Tracking-Sequence-0005-Preview.mp4)
- [Tracking Sequence 0011 Preview](assets/Tracking-Sequence-0011-Preview.mp4)
- [Tracking Sequence 0020 Preview](assets/Tracking-Sequence-0020-Preview.mp4)

## Repository Map

```text
.
├── assets/
│   ├── Detection-Overview.png       # Contact sheet of representative results
│   ├── Pipeline-Diagram.svg         # Visual overview of the processing stages
│   ├── *-Scene.png                  # Selected detection outputs
│   └── *-Preview.mp4                # Lightweight tracking previews
├── data/
│   └── README.md                    # Expected local dataset layout
├── docs/
│   └── architecture.md              # Design and evaluation notes
├── results/
│   ├── README.md
│   └── results.mat                  # Saved result structure from the experiment
├── src/
│   ├── main.m                       # End-to-end detection and tracking workflow
│   ├── compare_sobel_vs_hog_math.m  # Classical-method comparison
│   ├── hog_car_detector.m           # Optional HOG + linear-SVM baseline
│   └── test_pipeline.m              # Small smoke test
└── .gitignore
```

Technical folder names remain lowercase by design. This keeps MATLAB relative paths, GitHub links, and common repository conventions stable while giving the user-facing assets clear, human-readable names.

## Running the Project

### Requirements

- MATLAB R2023b or newer is recommended.
- Image Processing Toolbox.
- Computer Vision Toolbox.
- Statistics and Machine Learning Toolbox for `hog_car_detector.m`.
- Windows MATLAB is useful for the optional memory-profiling calls.

Raw image and video data are intentionally not included. The repository is not tied to the original computer-specific paths. See [`data/README.md`](data/README.md) for the expected layout and the `KITTI_ROOT` option.

### Steps

1. Clone the repository and open it in MATLAB.
2. Place permitted image data in `data/raw_dataset_pictures/` and video sequences in `data/raw_dataset_videos/`.
3. If KITTI tracking data is available, place it under `data/kitti/` or set the `KITTI_ROOT` environment variable.
4. Run the main workflow:

```matlab
run(fullfile('src', 'main.m'))
```

Generated files are written to `results/generated/`.

The two additional experiments can be run with:

```matlab
run(fullfile('src', 'compare_sobel_vs_hog_math.m'))
run(fullfile('src', 'hog_car_detector.m'))
```

## Scope and Limitations

This is an interpretable research baseline, not a production detector. Performance depends on lighting, camera geometry, segmentation quality, and the assumptions used to generate candidates. Some evaluation paths use annotated boxes for candidate verification, so the reported numbers should not be read as a blind deployment benchmark.

Only selected generated artifacts are included here. Raw datasets are not redistributed, and the repository contains no private credentials or company-internal material. For a new experiment, use data you are allowed to use and record the exact settings.
