# Data Placement

Raw datasets are intentionally not included in this public repository. The MATLAB entry point reads local, licensed data from the paths below.

## Picture data

```text
data/
└── raw_dataset_pictures/
    ├── 1_normal/
    │   └── *.jpg
    ├── 2_night/
    │   └── *.jpg
    ├── 3_rain/
    │   └── *.jpg
    ├── 4_shadow/
    │   └── *.jpg
    ├── 5_occlusion/
    │   └── *.jpg
    ├── 6_out_of_frame/
    │   └── *.jpg
    ├── pictures_gt.csv
    └── pictures_info.csv
```

The six folder names are technical identifiers used directly by `main.m`. Keep them unchanged. Image filenames must be unique across all six folders because the scripts use the filename to join annotations.

`pictures_gt.csv` must contain these columns:

```text
image,category,x1,y1,x2,y2
```

- `image`: filename, including `.jpg`;
- `category`: one of the six folder names above; and
- `x1,y1,x2,y2`: vehicle bounding-box coordinates in pixels.

`pictures_info.csv` must contain:

```text
image,timeofday
```

`timeofday` is used for the optional shadow statistics; the value `night` is treated as nighttime.

## Video data

```text
data/
└── raw_dataset_videos/
    ├── seq_0005/
    │   ├── frames/
    │   │   └── *.png  (or *.jpg)
    │   └── gt.csv
    ├── seq_0011/
    │   ├── frames/
    │   └── gt.csv
    └── seq_0020/
        ├── frames/
        └── gt.csv
```

Each `gt.csv` must contain at least:

```text
frame,track_id,x1,y1,x2,y2,ignore
```

`frame` is zero-based, matching the KITTI frame numbering used by `main.m`. `ignore=1` marks regions such as `DontCare`; those rows are excluded from evaluation. `occluded` and `truncated` may also be present, but the main workflow does not require them.

## Optional KITTI source

If `gt.csv` and frames are not already prepared, `main.m` can build the three sequences from the official KITTI Tracking files. The expected source tree is:

```text
KITTI_ROOT/
├── data_tracking_image_2/
│   └── training/image_02/0005/*.png
├── data_tracking_label_2/
│   └── training/label_02/0005.txt
└── data_tracking_calib/
    └── training/calib/0005.txt
```

The same layout is repeated for `0011` and `0020`. Set the `KITTI_ROOT` environment variable if the dataset is outside the repository. If it is stored locally, the default fallback is `data/kitti/`. Calibration is optional; when available, `main.m` writes a small local `calib.csv` beside the generated sequence.

Do not commit raw or licensed data unless redistribution is permitted. Keep private paths, credentials, and company-internal material outside the repository.
