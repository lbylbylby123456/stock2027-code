# ReCaT Submission Code

This anonymized package contains the core implementation used for the ReCaT experiments and the script for generating sensitivity-analysis figures.

## Structure

```text
recat_submission_code/
├── src/
│   └── run_recat.py
├── scripts/
│   └── generate_sensitivity_figures.py
├── configs/
│   └── default_config.yaml
├── data/
│   └── README.md
├── figures/
├── results/
└── requirements.txt
```

## Run ReCaT on one chronological split

Update `DATASET_NAME` and the processed file paths in `src/run_recat.py`, then run:

```bash
python src/run_recat.py
```

The script runs one train/validation/test split. Five-fold walk-forward results can be reproduced by running the same script over the corresponding split files and aggregating the metrics.

## Generate sensitivity figures

```bash
python scripts/generate_sensitivity_figures.py
```

The generated PDFs are written to `figures/`.

## Anonymization

Author names, institutional identifiers, emails, personal paths, repository URLs, and nonessential personal comments have been removed from this package.
