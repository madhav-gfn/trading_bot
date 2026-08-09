# Summary: Which 3 cohorts convert best, and on what metric

**Metric:** FT rate — the share of leads that reach **First Trip** (`FT_after_upload`), measured against total
leads uploaded. This was chosen over intermediate funnel metrics (connect rate, interest rate) because it
reflects the actual business outcome, not just telecalling activity.

**Aggregation level:** `lead_source` (cohort/campaign), the natural grain for "which cohort is best." Cohorts
under 100 leads were excluded as too small to draw a conclusion from.

**Cohorts split warm vs. cold**, because they aren't playing the same game: "warm" cohorts
(`PreOb-Ob Fees Paid...`, `AI Connected...`) are already pre-qualified leads, while "cold" cohorts
(`Single Referral`, `Khanna...`, `OLX...`) are fresh acquisition traffic. Ranking them together would just
reward funnel stage, not source quality.

## Data-quality fixes applied before ranking

- **Deduped leads by `candidate_phone`**: 1,077 phone numbers (12% of rows) appeared under more than one
  `lead_source`, and one lead was double-credited with an FT event. Kept one row per unique lead (their
  most-advanced funnel state) so no cohort's count is inflated by a lead shared with another source. This
  alone removed what had looked like a 3rd-place cold cohort (`Khanna AI`) — it turned out to be an artifact
  of the double-count.
- **Flagged 22 internally inconsistent rows** (`FT_after_upload=1` but `Connected=0`) rather than silently
  trusting them — these are shown per-cohort as `ft_anomalies` so a reader can see how much of a cohort's win
  depends on them.
- **Reported 95% Wilson confidence intervals** alongside every FT rate, since counts are single digits per
  cohort (14, 14, 7, 7, 5...) — a bare ranking of these would overstate precision.

## Result

**Top cold cohorts:**

| Rank | Cohort | Leads | FT events | FT rate | 95% CI |
|---|---|---|---|---|---|
| 1 | Single Referral > 7 days - 24th Jul | 1,500 | 14 | 0.93% | 0.56–1.56% |
| 2 | Khanna - 2W 26th Jul | 1,546 | 14 | 0.91% | 0.54–1.51% |
| 3 | OLX - Ashwin - 2W - 17 Jul | 5,182 | 4 | 0.08% | 0.03–0.20% |

**Top warm cohorts:**

| Rank | Cohort | Leads | FT events | FT rate | 95% CI |
|---|---|---|---|---|---|
| 1 | PreOb-Ob Fees Paid 29th Jul (set 1) | 1,479 | 7 | 0.47% | 0.23–0.97% |
| 2 | PreOb-Ob Fees Paid 29th Jul (set 2) | 1,550 | 7 | 0.45% | 0.22–0.93% |
| 3 | AI Connected but not Connected by TC - Set 1 | 1,480 | 5 | 0.34% | 0.14–0.79% |

## Caveats that belong in the write-up, not just the appendix

- **#1 and #2 cold cohorts are statistically tied** — their confidence intervals overlap heavily. Treat them
  as co-leaders, not a clean 1-2.
- **#3 cold cohort (`OLX`) is weak evidence** — only 4 FT events, and 3 of those are the flagged anomaly (FT
  recorded with no logged connect). It's a real cohort, but the least trustworthy of the six shown.
- **All three warm cohorts have overlapping CIs** — directionally `PreOb set 1` leads, but not distinguishable
  from the other two at this sample size.
- **Bottom line:** only two cohorts (`Single Referral`, `Khanna- 2W 26th Jul`) show a genuinely reliable
  FT-rate signal above the noise floor; everything past that is a weaker claim and should be presented as
  such.
