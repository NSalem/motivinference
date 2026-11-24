# Figures

One script per figure of the paper and supplement. Each one draws the whole figure, schematics included, and saves it
at its printed size, so nothing is rescaled or edited by hand afterwards.

Double-column figures are laid out in millimetres on a 183 mm page (`figstyle.LAYOUT_DOUBLE`) and saved at the
journal's maximum width, 180 mm for NHB (`figstyle.DOUBLE`): `Canvas` scales positions and drawings by
`figstyle.page_scale()` (180/183), while type and line widths keep their point sizes. Coordinates in the scripts are
therefore layout millimetres.

| Script | Figure | Printed size (mm) |
|---|---|---|
| `fig1.py` | 1: design, trials, stimulus distributions, main behavioural results | 180 × 106.2 |
| `fig2.py` | 2: noisy-inference model, model comparison, inference noise, model vs data | 180 × 106.7 |
| `fig3.py` | 3: full model: noise per source, variance partition, model lesioning | 180 × 91.5 |
| `fig4.py` | 4: bias and noise from repeated trials | 180 × 140.7 |
| `fig5.py` | 5: response times | 180 × 126.4 |
| `sfig1.py` … `sfig7.py` | Supplementary Figures 1–7 | 180 wide (S5: 89) |

The inputs of each figure are listed in its docstring and in `PAPER_MAP.md`. Data loading and the statistics drawn in
the figures come from `src/motinf/` (`motinf.exploratory.figure_data` and the analysis modules), never from copies,
so the numbers in a figure cannot drift from the tables.

## Running

From the repository root, in the environment of `environment.yml`:

```bash
python run_pipeline.py --stage figures        # fig1 ... sfig7
python scripts/exploratory/figures/fig3.py                        # one figure
```

Each script writes `plots/exploratory/figures/<name>.svg` (live text), a 300 dpi `.png` preview and, when Arial is
installed, a `.pdf`. It also writes the figure's source data, the plotted values, to
`plots/exploratory/figures/source_data/<name>/`: one CSV per panel (`<name><panel>[_<part>].csv`; one row per participant
and condition where the panel shows participants) and `index.csv`, which describes each file
(`motinf.exploratory.figlib.source_data`). Environment switches:

| Variable | Effect |
|---|---|
| `MOTINF_FIG_OUT` | write `<dir>/figures/<name>.*` instead (for example to compare a render with the committed files) |
| `MOTINF_FONT_DIR` | an extra folder in which to look for `arial.ttf` and its bold / italic files |

**Arial.** It is looked for in `MOTINF_FONT_DIR`, the Windows font folder (native or `/mnt/c/Windows/Fonts` from WSL),
and the macOS font folders. Without it, glyphs are drawn with Liberation Sans, which has the same character widths, so
the layout does not change. The SVGs still name Arial, and no PDF is written, because it would embed Liberation Sans.

## House style (`src/motinf/exploratory/figlib/`)

- **`figstyle`**: page geometry, type, lines, colours, labels and export. `save()` checks every figure and warns about:
  - text outside 5–7 pt (panel letters excepted), and coloured text;
  - text that overlaps other text or runs off the page;
  - a figure wider or taller than the page;
  - glyphs that fall back to a font other than Arial.
- **`dist`**: distribution panels. For each condition:
  - a box (IQR, median, whiskers to 1.5 IQR) with the mean ± 95% CI on it;
  - a half violin holding the participants as dots;
  - stars for p < .05 / .01 / .001, with no "ns";
  - values beyond the axis limits counted at the edge.
- **`canvas`**: a whole-figure canvas in mm, and the task's gratings. **`matrix`**: confusion and correlation matrices.

Conventions:

- **Type** (Arial): 7 pt for axis labels and titles, 6 pt for ticks and annotations, 8 pt bold for panel letters. Text is
  always black; colour is carried by the marks.
- **Spacing** (mm, at print size):
  - panel letters 0.5 mm from the left edge, the first row 1.6 mm from the top, titles 4 mm right of their letter;
  - about 1.2–1.5 mm from a title to its panel, about 2 mm between rows, 1 mm below the lowest element.
- **Colours**:
  - loss `#e41a1c`, neutral `#8c8c8c`, gain `#4daf4a`;
  - noise sources: sensory `#b26f01`, inference `#0a80a9`, selection `#8c00b2`;
  - shaded panel backgrounds: the colour at 15% on white.
- **Labels**:
  - incentives "−1 / 0 / +1" with "Incentive (£)", and "vs £0" on effect panels;
  - task names from `figstyle.TASK_REPRO` / `TASK_CATEG` ("Stimulus Reproduction" / "Sequence Categorization");
  - RT-split panels: "Noise, fast − slow" and "Incentive effect, fast − slow".

New style settings go in `figlib`, not in individual scripts.

Fixed choices: the sensory-noise symbol is σ_ang, as in the text and tables; one set of noise-source colours everywhere;
lowercase panel letters (NHB); Fig. 2d, right, shows the reduction (£0 − incentive, positive when incentives help).
