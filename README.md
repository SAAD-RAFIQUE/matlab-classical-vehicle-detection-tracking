# Classical Vehicle Detection and Tracking in MATLAB

<p align="center">
  <img src="assets/Detection-Overview.png" alt="Representative detection results across several road-scene conditions" width="920">
</p>

<p align="center">
  <strong>A math-driven MATLAB pipeline for vehicle detection, multi-object tracking, and failure-case analysis.</strong><br>
  Hand-crafted image mathematics, explicit geometry, and motion models — no learned detector in the main workflow.
</p>

## At a Glance

| Input | Processing | Output |
| --- | --- | --- |
| Road-scene images and video sequences | Deterministic image processing, HOG gradient histograms, cosine similarity, geometric constraints, IoU filtering, and Kalman tracking | Annotated frames, track IDs, trajectories, saved results, and evaluation summaries |

This repository contains an intentionally classical computer-vision workflow. The main `main.m` path uses no machine-learning model, training step, classifier fitting, or neural network. HOG is used as a hand-crafted mathematical descriptor: image gradients are converted into orientation histograms, then compared with a canonical vehicle template using cosine similarity.

The emphasis is on making each decision visible: where candidates come from, why boxes are rejected, how tracks are assigned, and where the method begins to fail.

## Pipeline

<p align="center">
  <img src="assets/Pipeline-Diagram.svg" alt="Deterministic pipeline from road-scene frames to vehicle tracks and evaluation" width="1100">
</p>

The main workflow combines:

- image gradients, HOG orientation histograms, and cosine similarity;
- shadow- and morphology-based candidate generation;
- road and perspective constraints;
- IoU-based non-maximum suppression;
- Kalman filtering for position and velocity;
- appearance-aware association between detections and tracks;
- camera- and ego-motion compensation; and
- per-category and per-sequence evaluation.

The main script is a deterministic mathematical pipeline. A separate `experiments/hog_car_detector.m` proof-of-concept script demonstrates a trained linear-SVM comparison; it is not part of the main `main.m` workflow.

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
│   └── main.m                       # Only entry point: deterministic detection and tracking
├── experiments/
│   ├── README.md                    # Auxiliary scripts and how to run them
│   ├── compare_sobel_vs_hog_math.m  # Sobel versus pure-math HOG comparison
│   ├── hog_car_detector.m           # Separate optional HOG + linear-SVM experiment
│   └── test_pipeline.m              # Small smoke test
└── .gitignore
```

Technical folder names remain lowercase by design. This keeps MATLAB relative paths, GitHub links, and common repository conventions stable while giving the user-facing assets clear, human-readable names.

## Running the Project

### Requirements for the Main Workflow

- MATLAB R2023b or newer is recommended.
- Image Processing Toolbox.
- Computer Vision Toolbox.
- Windows MATLAB is useful for the optional memory-profiling calls.

The main workflow does not require a trained model. The Statistics and Machine Learning Toolbox is needed only if you choose to run the separate optional `hog_car_detector.m` comparison script.

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

The pure-math Sobel-versus-HOG comparison can be run with:

```matlab
run(fullfile('src', 'compare_sobel_vs_hog_math.m'))
```

The separate optional HOG + linear-SVM proof of concept can be run with:

```matlab
run(fullfile('src', 'hog_car_detector.m'))
```

## Scope and Limitations

This is an interpretable research baseline, not a production detector. Performance depends on lighting, camera geometry, segmentation quality, and the assumptions used to generate candidates. The main workflow is deterministic and math-driven, but some evaluation paths use annotated boxes for candidate verification; the reported numbers should therefore not be read as a blind deployment benchmark.

Only selected generated artifacts are included here. Raw datasets are not redistributed, and the repository contains no private credentials or company-internal material. For a new experiment, use data you are allowed to use and record the exact settings.
