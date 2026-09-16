# Data placement

Raw datasets are intentionally not included in this public repository.

To reproduce the pipeline, place licensed data in this structure:

```text
data/
├── raw_dataset_pictures/
│   ├── 1_normal/
│   ├── 2_night/
│   ├── 3_rain/
│   ├── 4_shadow/
│   ├── 5_occlusion/
│   ├── 6_out_of_frame/
│   ├── pictures_gt.csv
│   └── pictures_info.csv
├── raw_dataset_videos/
│   ├── seq_0005/
│   │   ├── frames/
│   │   └── gt.csv
│   ├── seq_0011/
│   └── seq_0020/
└── kitti/                         # Optional KITTI root or set KITTI_ROOT
```

The MATLAB scripts expect the picture categories and annotation columns used by the original experiment. Do not commit data unless its license permits redistribution. Keep local dataset paths, credentials, and internal company material outside the repository.

For KITTI, use the official source and follow its license and citation requirements. When a local KITTI installation is outside the repository, set `KITTI_ROOT` before running `src/main.m`.
