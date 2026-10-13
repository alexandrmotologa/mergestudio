# CLI Reference

The `mergestudio` command line utility exposes all merge algorithms and studio management tools.

## Commands

### `mergestudio merge`
Run a full model merge from a declarative configuration file.

```bash
mergestudio merge --config ./configs/plan.yaml
```

Options:
- `-c, --config PATH`: Path to merge plan YAML or JSON file. [required]

---

### `mergestudio slerp`
Merge two models using Spherical Linear Interpolation.

```bash
mergestudio slerp --model-a ./model_a --model-b ./model_b --alpha 0.5 -o ./out
```

Options:
- `-a, --model-a PATH`: Path to first model directory. [required]
- `-b, --model-b PATH`: Path to second model directory. [required]
- `--alpha FLOAT`: Interpolation factor in [0.0, 1.0]. Default: 0.5.
- `-o, --output PATH`: Output directory for merged files. Default: `./output/slerp_merged`.
- `--per-row / --no-per-row`: Apply SLERP across row vectors rather than globally.
- `--preserve-norm / --no-preserve-norm`: Interpolate vector norms. Default: True.

---

### `mergestudio ties`
Merge two or more fine tuned models into a base anchor model using TIES-Merging.

```bash
mergestudio ties --base ./base_model -d ./donor_1 -d ./donor_2 --density 0.2 -o ./out
```

Options:
- `--base PATH`: Path to base anchor model. [required]
- `-d, --donor PATH`: Path to fine tuned donor model (repeatable). [required]
- `--density FLOAT`: Fraction of top magnitude weights to retain. Default: 0.2.
- `--weight FLOAT`: Global multiplier lambda for the aggregated delta. Default: 1.0.
- `-o, --output PATH`: Output directory. Default: `./output/ties_merged`.

---

### `mergestudio dare`
Merge models using Drop And Rescale.

```bash
mergestudio dare --base ./base_model -d ./donor_1 -p 0.5 --rescale -o ./out
```

Options:
- `--base PATH`: Path to base anchor model. [required]
- `-d, --donor PATH`: Path to fine tuned donor model (repeatable). [required]
- `-p, --drop-rate FLOAT`: Delta drop probability in [0.0, 1.0). Default: 0.5.
- `--rescale / --no-rescale`: Scale surviving weights by 1/(1-p). Default: True.
- `-o, --output PATH`: Output directory. Default: `./output/dare_merged`.

---

### `mergestudio inspect`
Inspect model structure, layers, hidden size, and tensor count.

```bash
mergestudio inspect ./path/to/model
```

---

### `mergestudio studio`
Launch the interactive Web Studio.

```bash
mergestudio studio --host 127.0.0.1 --port 8000
```
