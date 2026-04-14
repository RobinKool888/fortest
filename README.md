# Issue Trend Forecast

Digitise a cumulative-issue curve image, fit multiple growth models,
and generate a numeric + visual forecast.

---

## How it works

| Step | What happens | Output file(s) |
|------|-------------|----------------|
| **1 – Extract** | Green-curve pixels are detected and mapped to `(date, count)` using the two edge calibration points you supply | `extracted_data.csv` |
| **2 – Fit** | Seven growth models are fit; the best R² wins | `fit_comparison.png`, `model_r2_comparison.png` |
| **3 – Forecast** | All models predict future counts; the winner is charted | `predictions.csv`, `forecast.png` |

---

## Option A — Local deployment

### Prerequisites

```bash
pip install numpy scipy matplotlib pillow pandas
```

For Chinese labels in charts, install a CJK font first:

```bash
# Ubuntu / Debian
sudo apt-get install fonts-noto-cjk

# macOS (Homebrew)
brew install --cask font-noto-sans-cjk

# Windows
# Download and install "Noto Sans CJK SC" from https://fonts.google.com/noto
```

> **No CJK font?** The script automatically falls back to English labels —
> no manual configuration needed.

### Run

```bash
python issue_forecast.py \
    --image       "Sport Gloves Vice (Minimal Wear).png" \
    --left-date   2025-10-26 \
    --left-count  9650 \
    --right-date  2026-04-09 \
    --right-count 12200
```

All five output files are written to the same directory as the image.

### Full argument reference

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--image` | No | `Sport Gloves Vice (Minimal Wear).png` (next to script) | Path to the curve PNG |
| `--left-date` | **Yes** | — | Date at the left edge of the X-axis (`YYYY-MM-DD`) |
| `--left-count` | **Yes** | — | Issue count at the left edge |
| `--right-date` | **Yes** | — | Date at the right edge of the X-axis (`YYYY-MM-DD`) |
| `--right-count` | **Yes** | — | Issue count at the right edge |
| `--today` | No | `right-date` | Forecast reference / "today" date |
| `--out-dir` | No | image directory | Directory where outputs are saved |
| `--future-days` | No | `30 60 90 180 365 730` | Space-separated list of forecast horizons (days) |

---

## Option B — GitHub Actions (online, no local install)

### One-time setup

1. **Push your curve image** to the repository root (default name: `Sport Gloves Vice (Minimal Wear).png`).  
   If you use a different path, pass it via the `image_path` input.

2. The workflow file is already at `.github/workflows/forecast.yml` — no
   further configuration is needed.

### Running the workflow

1. Go to the repository on GitHub.
2. Click **Actions** → **Issue Trend Forecast** → **Run workflow**.
3. Fill in the inputs:

   | Input | Required | Example |
   |-------|----------|---------|
   | Left-edge date | **Yes** | `2025-10-26` |
   | Left-edge count | **Yes** | `9650` |
   | Right-edge date | **Yes** | `2026-04-09` |
   | Right-edge count | **Yes** | `12200` |
   | Forecast date (`today`) | No | `2026-04-14` |
   | Forecast horizons | No | `30 60 90 180 365 730` |
   | Image path in repo | No | `Sport Gloves Vice (Minimal Wear).png` |

4. Click **Run workflow**.

### Downloading the results

Once the job finishes (usually under 2 minutes):

1. Click the completed run.
2. Scroll to the **Artifacts** section at the bottom.
3. Click **forecast-outputs** to download a ZIP containing all five files.

```
forecast-outputs.zip
├── extracted_data.csv        ← Step 1: raw digitised curve data
├── fit_comparison.png        ← Step 2: all-model overlay chart
├── model_r2_comparison.png   ← Step 2: R² bar chart
├── predictions.csv           ← Step 3: numeric forecast table
└── forecast.png              ← Step 3: best-model forecast chart
```

Artifacts are retained for **30 days**.

---

## Tips

- **Square boxes instead of Chinese text?**  
  Install a CJK font (see above) or run on GitHub Actions which installs
  `fonts-noto-cjk` automatically.

- **Curve not detected?**  
  The extractor looks for pixels where `G > 80` and `R, B < 50` (pure green).
  Adjust `GREEN_CHANNEL_MIN` / `RED_BLUE_CHANNEL_MAX` at the top of
  `issue_forecast.py` if your chart uses a different shade.

- **Custom forecast horizons:**  
  ```bash
  --future-days 7 14 30 60
  ```
