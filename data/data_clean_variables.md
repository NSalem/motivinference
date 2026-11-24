## Cleaned inference data: `data_clean_inf.csv`

These files live under `data/processed/<exp>/data_clean_inf.csv` (e.g. `data/processed/exp1/`), and contain **one row per categorization (inference) trial** after RT and participant-level cleaning.

### Trial structure and stimulus information

- **seqang_1..12**: sequence of Gabor orientations in radians (original, noiseless angles), one per position. In this and every other `_1..12` column, the positions beyond `seqlen` hold 0.
- **seqllr_1..12**: log-likelihood ratios contributed by each element in the sequence (original, noiseless evidence per item).
- **gaborimg_1..12**: integer IDs of the Gabor image used at each sequence position.
- **gaborpeakori_1..12**: peak orientation of the underlying Gabor kernel (radians).
- **gaborpeakenergy_1..12**: peak energy of the Gabor response at each position.
- **gabormeanenergynoise_1..12**: mean noise energy of the Gabor response at each position.
- **gabormaxenergynoise_1..12**: maximum noise energy across spatial positions for each item.
- **gabornoisepeakssum_1..12**: sum of peaks in the noise-only Gabor energy maps per item.
- **gabornoisepeaksn_1..12**: number of peaks in the noise-only Gabor energy maps per item.

### Noisy sequence (model-relevant) quantities

- **seqang_noisy_1..12**: noisy orientations actually presented on each trial (after adding external noise).
- **seqllr_noisy_1..12**: corresponding noisy log-likelihood ratios for each item.
- **sumllr_noisy**: total noisy log-likelihood ratio over the sequence; the main scalar evidence variable used in modeling and regression.

### Participant and session metadata

- **participant**: participant identifier (e.g. `S001`).
- **date**: session timestamp string.
- **session**: session (1 or 2).
- **incentive**: incentive condition code (`-1`, `0`, `1` for loss / neutral / gain).
- **seqlen**: sequence length (number of items shown: 4, 8 or 12).
- **trialN**: trial number of the participant across both sessions (1–864; session 2 starts at 433), assigned before cleaning, so removed trials leave gaps.

### Ground truth and mapping

- **sumllr**: noiseless total log-likelihood ratio for the sequence (before adding external noise).
- **true_cat**: true category (`-1` / `+1`).
- **kappa**: concentration parameter of the generative von Mises distribution for orientations (0.5 throughout).
- **respmap**: mapping of the categories to the response keys (`-1` / `+1`): `choice = -resp_side × respmap`.

### Behavioural responses (categorization)

- **resp**: key pressed (`left` / `right`).
- **resp_side**: `resp` coded as `-1` (left) / `+1` (right).
- **choice**: signed choice coded as `-1` / `+1` for the two categories.
- **correct**: binary accuracy indicator (`1` if choice matches `true_cat`, else `0`).
- **rt**: reaction time for the categorization response (seconds).

### Summary features

- **optim**: `1` if the choice agrees with the sign of `sumllr_noisy`, i.e. the choice an ideal observer of the presented sequence would make ("choices consistent with the evidence" in the paper).

All trials in `data_clean_inf.csv` have passed trial-level RT cuts and participant-level exclusion defined in `motinf.prereg.cleaning`. The table is suitable for psychometric modeling and variance-partition analyses of **binary choices**. Repeated sequences (for the variance partition) are identified by their noisy sample angles (`seqang_noisy_1..12`).

---

## Cleaned estimation data: `data_clean_est.csv`

These files live under `data/processed/<exp>/data_clean_est.csv` (e.g. `data/processed/exp1/`), and contain **one row per estimation trial** after RT and participant-level cleaning.

### Participant and session metadata

- **participant**: participant identifier (e.g. `S001`).
- **date**: session timestamp string.
- **session**: session (1 or 2).
- **incentive**: incentive condition code (`-1`, `0`, `1` for loss / neutral / gain).
- **trialN**: trial number of the participant across both sessions (1–144; session 2 starts at 73), assigned before cleaning, so removed trials leave gaps.

### Orientation and noise variables

- **estimang**: true orientation to be estimated on that trial (radians, wrapped to [0, π)).
- **estimang_noisy**: noisy version of the true orientation after adding external noise (radians, wrapped to [0, π)).
- **gaborimg_estim**: integer ID of the Gabor image used in the estimation display.
- **peakori_estim**: peak orientation of the underlying Gabor kernel for the estimation stimulus (radians).
- **gabormeanenergynoise_estim**: mean noise energy of the Gabor response for the estimation stimulus.

### Behavioural responses (estimation)

- **resp_estim**: participant’s estimation response (radians, wrapped to [0, π)).
- **dev**: signed angular deviation between response and true orientation (radians, wrapped to [-π/2, π/2]).
- **dev_noisy**: signed angular deviation relative to the noisy true orientation (radians, wrapped to [-π/2, π/2]).
- **absdev**: absolute value of `dev` (radians).
- **absdev_noisy**: absolute value of `dev_noisy` (radians).
- **rt**: reaction time for the estimation response (seconds).

All trials in `data_clean_est.csv` have passed trial-level RT cuts and participant-level QC as implemented in `motinf.prereg.cleaning`. This table is intended for analyses of **continuous estimation errors** and the associated variance partitioning; repeated stimuli are identified by `gaborimg_estim`.

