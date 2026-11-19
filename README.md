# Vahan lead funnel analysis

This repository contains an exploratory analysis of the Vahan case study lead funnel. The work focuses on one question: which lead sources have the highest observed rate of leads reaching First Trip (FT)? A second section examines how fields recorded in the funnel are associated with that outcome.

The analysis uses the supplied case study extract. It is a one-time observational dataset, not a live scoring system. The model results are retrospective and are not evidence that a lead can be ranked at upload time.

## Project files

- `analysis.ipynb` contains the data checks, cohort comparisons, SQL cross-check, exploratory model and recommendations.
- `generate_report.py` regenerates a PDF report from the CSV without relying on saved notebook outputs.
- `cohort_analysis_summary.md` is a short write-up of the cohort results and their limitations.
- `Vahan_Case_Study.csv` is the source data used by the notebook.
- `Vahan_Case_Study_Submission.pdf` is the supplied case study brief.
- `requirements.txt` lists the Python packages used by the notebook.

## Running the analysis

Use Python 3.11 or newer. From the project directory, install the listed packages and launch Jupyter:

```powershell
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open `analysis.ipynb` and run the cells from top to bottom. The notebook expects `Vahan_Case_Study.csv` in the project directory. It creates its SQL comparison in an in-memory SQLite database and does not write analysis results to separate files.

To generate the report PDF from the current CSV, run:

```powershell
python generate_report.py
```

This writes `Vahan_Case_Study_Report.pdf` in the project directory. The source submission PDF is left untouched. Use `--input` and `--output` to select different paths.

## What the analysis does

The notebook checks repeated phone hashes, missing identifiers and funnel inconsistencies before aggregation. For known phone hashes that appear more than once, it retains the row with the most advanced recorded funnel state. Rows without a phone hash are kept individually because they cannot safely be identified as the same lead.

It compares `FT_after_upload` by `lead_source`, excludes cohorts with fewer than 100 uploaded leads, and ranks warm and cold sources separately. The report includes Wilson confidence intervals and pairwise Fisher exact tests to show how uncertain the ranking is. A SQL aggregation provides a second implementation of the cohort calculation.

The model section compares classifiers using five-fold stratified cross-validation. Its features include call, interest and onboarding fields whose timing relative to FT is not established by the supplied extract. The resulting scores and feature rankings are descriptive only; in particular, the reported top-percent capture is not a prospective estimate and should not guide call prioritization.

## Current read of the results

The strongest observed cold-source rates are for `Single Referral > 7 days` and `Khanna- 2W 26th Jul`, at roughly 0.9% each. Their confidence intervals overlap substantially, so the data does not support a reliable first-versus-second ordering. `OLX` ranks below them, but its estimate is based on four FT events, three of which have no logged connection.

Among warm sources, the observed rates are close and the available event counts do not distinguish the leaders. Read the full [cohort summary](cohort_analysis_summary.md) alongside the notebook for the detailed figures and caveats.

## Project notes

### 9 August 2026

The first pass established the cohort question and the notebook workflow. The lead-source comparison was made the main result, with FT rate used in place of intermediate call activity as the outcome.

The data checks then surfaced repeated phone hashes across sources and FT rows without a logged connection. The cohort summary was added to keep those issues and the uncertainty visible next to the ranking.

The notebook was expanded with a SQL cross-check and an exploratory model section. Later review found that its funnel fields could not be confirmed as available before the outcome. The model metrics are retained as retrospective analysis, not a lead-prioritization result.

### 19 November 2026

The audit kept rows with missing phone hashes separate during deduplication, aligned the SQL version with that rule, and added the notebook's SciPy and JupyterLab dependencies. The model interpretation and recommendations now state the limits of its retrospective features.

The project now includes a PDF report generator. It recalculates data checks, cohort rates, confidence intervals and pairwise tests from the CSV at run time, then draws a cohort interval chart and a log-scale funnel chart. The report does not copy the old submission's static numbers.

