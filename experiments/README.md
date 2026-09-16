# Experiments

`src/main.m` is the only repository entry point. The scripts in this folder are separate demonstrations and diagnostics; they are not required by the main workflow.

- `compare_sobel_vs_hog_math.m` — compares Sobel edge mathematics with the pure-math HOG cosine-similarity method and writes benchmark proof images.
- `test_pipeline.m` — smoke-tests one sample image from each available picture category.
- `hog_car_detector.m` — optional HOG + linear-SVM proof of concept. It trains a classifier and therefore is not part of the deterministic `main.m` path.

Run them from the repository root in MATLAB:

```matlab
run(fullfile('experiments', 'compare_sobel_vs_hog_math.m'))
run(fullfile('experiments', 'test_pipeline.m'))
run(fullfile('experiments', 'hog_car_detector.m'))
```

All three scripts resolve the repository root from their own location and keep generated outputs under `results/`.
