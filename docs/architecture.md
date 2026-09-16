# Technical architecture

## Detection

The main detector combines complementary classical signals:

1. Category-aware grayscale preprocessing adjusts contrast and denoises difficult conditions.
2. Candidate boxes are proposed from annotated geometry during the original evaluation workflow or from road-shadow morphology during standalone inference.
3. Each candidate is resized to a canonical patch and represented by a HOG descriptor.
4. The descriptor is normalized and compared with a canonical vehicle template using cosine similarity:

```text
cos(theta) = (h · h_ref) / (||h|| ||h_ref||)
```

5. A perspective-consistency term estimates whether the candidate width is plausible for its vertical position in the road image.
6. The final score combines appearance and geometry, followed by IoU-based non-maximum suppression.

The primary `main.m` workflow is classical and interpretable. It does not contain a deep neural detector. The optional `hog_car_detector.m` file adds a separate HOG + linear-SVM comparison experiment.

## Tracking

For each confirmed vehicle track, the state is:

```text
x = [u, v, du, dv]^T
```

where `(u, v)` is image position and `(du, dv)` is image velocity. A constant-velocity Kalman model predicts the next state and updates it with matched detections.

Track association combines:

- center displacement,
- bounding-box IoU,
- appearance-histogram similarity.

The pipeline also estimates camera motion from background features and reports raw versus compensated motion.

## Evaluation

Detection is evaluated with one-to-one IoU matching at the configured threshold. The scripts report:

- true positives, false positives, and false negatives,
- precision, recall, F1 score, and mean IoU,
- per-condition results,
- tracking success/failure and recovery,
- position and velocity RMSE,
- ID switches and abrupt-scene-change counts,
- per-image/frame latency, FPS, and MATLAB memory usage.

## Known limitations

- Candidate generation and perspective scoring are camera-geometry dependent.
- Small, heavily occluded, night, or out-of-frame vehicles are difficult cases.
- The original experiment uses ground-truth boxes for candidate verification in parts of the benchmark; this must be separated from a fully blind deployment evaluation.
- Quantitative results should be regenerated on a clearly licensed dataset before being used in a paper or production claim.
