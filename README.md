# UniFaceForge reproduction

This repository contains a runnable PyTorch implementation reconstructed from
the equations in `Paper/main.tex`. Install with `python -m pip install -e .`
and run a check with `python -m pytest -q`.

To reproduce metrics on an accessible local copy of the data:

```text
data/
  fake/   # all manipulated images
  real/   # authentic images
```

Then run `python scripts/train.py data --epochs 1`. The script records the
seed, class mapping, split sizes, confusion matrix, and metrics in the saved
checkpoint. The public Hugging Face files are currently gated and total about
90 GB, and this repository contains neither those files nor a trained
checkpoint; therefore the paper's published numbers cannot be verified here
without authenticated access to the dataset.

After authenticating with `huggingface-cli login`, download the gated files
with `python scripts/download_dataset.py D:/datasets/unifaceforge`.

After training, compute the score-based metrics with
`python scripts/evaluate.py data runs/unifaceforge.pt`; this reports ROC AUC
and EER from the saved predictions rather than copying values from the paper.
