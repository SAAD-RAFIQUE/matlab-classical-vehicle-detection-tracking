# Technical Architecture

## Main Workflow: Deterministic Detection

The main detector is a deterministic, math-driven computer-vision pipeline. It does not train or load a machine-learning model. Its signals are calculated directly from image pixels, gradients, geometry, and motion:

1. Category-aware grayscale preprocessing adjusts contrast and denoises difficult conditions.
2. Candidate boxes are proposed from annotated geometry during the original evaluation workflow, or from road-shadow morphology during standalone inference.
3. Each candidate is resized to a canonical patch.
4. HOG converts local image gradients into orientation histograms. This is a hand-crafted mathematical descriptor, not a learned model.
5. The HOG vector is normalized and compared with a canonical vehicle template using cosine similarity:

```text
cos(theta) = (h · h_ref) / (||h|| ||h_ref||)
```

6. A perspective-consistency term estimates whether the candidate width is plausible for its vertical position in the road image.
7. The final score combines HOG similarity and geometry, followed by IoU-based non-maximum suppression.

The primary `main.m` workflow is therefore classical, deterministic, and interpretable. The separate `compare_sobel_vs_hog_math.m` script follows the same pure-math approach when comparing Sobel and HOG.

## Tracking

For each confirmed vehicle track, the state is:

```text
x = [u, v, du, dv]^T
```

Here, `(u, v)` is the image position, and `(du, dv)` is the image velocity. A constant-velocity Kalman model predicts the next state and updates it with matched detections.

Track association combines:

- center displacement;
- bounding-box IoU; and
- appearance-histogram similarity.

The pipeline also estimates camera motion from background features and reports raw motion alongside compensated motion.

## Evaluation

Detection is evaluated with one-to-one IoU matching at the configured threshold. The scripts report:

- true positives, false positives, and false negatives;
- precision, recall, F1 score, and mean IoU;
- per-condition results;
- tracking success, failure, and recovery;
- position and velocity RMSE;
- ID switches and abrupt-scene-change counts; and
- per-image or per-frame latency, FPS, and MATLAB memory usage.

## Optional HOG + Linear-SVM Comparison

`hog_car_detector.m` is separate from the main workflow. It trains a linear SVM on HOG descriptors as a proof-of-concept comparison. This optional script is the only part of the repository that uses a trained classifier; it should not be confused with the pure-math HOG cosine-similarity path in `main.m` or `compare_sobel_vs_hog_math.m`.

## Known Limitations

- Candidate generation and perspective scoring depend on camera geometry.
- Small, heavily occluded, nighttime, and out-of-frame vehicles are difficult cases.
- The original experiment uses ground-truth boxes for candidate verification in parts of the benchmark; this must be separated from a fully blind deployment evaluation.
- Quantitative results should be regenerated on a clearly licensed dataset before they are used in a paper or production claim.
