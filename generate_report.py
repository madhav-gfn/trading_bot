"""Generate a PDF cohort report from the Vahan case study CSV."""

from __future__ import annotations

import argparse
import math
import re
import unicodedata
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact


PAGE_WIDTH, PAGE_HEIGHT = 595, 842
LEFT, RIGHT = 54, 541
INK = (0.09, 0.13, 0.17)
MUTED = (0.35, 0.41, 0.46)
BLUE = (0.21, 0.42, 0.60)
ORANGE = (0.76, 0.40, 0.12)
GREEN = (0.21, 0.51, 0.39)
RED = (0.73, 0.27, 0.26)
PURPLE = (0.43, 0.35, 0.62)
TEAL = (0.16, 0.52, 0.57)
PALE = (0.95, 0.96, 0.97)
GRID = (0.84, 0.87, 0.89)
MIN_LEADS = 100
WARM_KEYWORDS = ("PreOb", "AI Connected")


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.96
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def load_and_clean(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_csv(path)
    required = {
        "candidate_phone", "lead_source", "Uploaded Leads", "Attempted",
        "Connected", "Interested", "FT_after_upload",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    repeated = raw.loc[raw["candidate_phone"].notna(), "candidate_phone"].value_counts()
    repeated = repeated[repeated > 1]
    repeated_rows = raw[raw["candidate_phone"].isin(repeated.index)]
    priority = ["FT_after_upload", "Interested", "Connected", "Attempted"]
    ordered = raw.sort_values(priority, ascending=False).copy()
    ordered["_dedupe_key"] = ordered["candidate_phone"].astype("object")
    missing_phone = ordered["candidate_phone"].isna()
    ordered.loc[missing_phone, "_dedupe_key"] = [
        f"__missing_phone_row_{index}" for index in ordered.index[missing_phone]
    ]
    clean = ordered.drop_duplicates("_dedupe_key", keep="first").drop(columns="_dedupe_key").copy()
    clean["ft_anomaly"] = (clean["FT_after_upload"] == 1) & (clean["Connected"] == 0)

    return clean, {
        "raw_rows": len(raw),
        "unique_rows": len(clean),
        "repeated_phone_hashes": len(repeated),
        "repeated_phone_rows": len(repeated_rows),
        "missing_phone_rows": int(raw["candidate_phone"].isna().sum()),
        "raw_ft": int(raw["FT_after_upload"].sum()),
        "clean_ft": int(clean["FT_after_upload"].sum()),
        "ft_anomalies": int(((raw["FT_after_upload"] == 1) & (raw["Connected"] == 0)).sum()),
    }


def cohort_table(clean: pd.DataFrame) -> pd.DataFrame:
    cohort = clean.groupby("lead_source", dropna=False).agg(
        leads=("Uploaded Leads", "sum"),
        attempted=("Attempted", "sum"),
        connected=("Connected", "sum"),
        interested=("Interested", "sum"),
        ft=("FT_after_upload", "sum"),
        anomalies=("ft_anomaly", "sum"),
    ).reset_index()
    cohort["segment"] = cohort["lead_source"].map(
        lambda source: "warm" if any(key in str(source) for key in WARM_KEYWORDS) else "cold"
    )
    cohort["ft_rate"] = cohort["ft"] / cohort["leads"]
    intervals = cohort.apply(lambda row: wilson_interval(int(row["ft"]), int(row["leads"])), axis=1)
    cohort["ci_low"] = [interval[0] for interval in intervals]
    cohort["ci_high"] = [interval[1] for interval in intervals]
    cohort["attempt_rate"] = cohort["attempted"] / cohort["leads"]
    return cohort[cohort["leads"] >= MIN_LEADS].sort_values(
        ["segment", "ft_rate", "lead_source"], ascending=[True, False, True]
    )


def ascii_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class PdfPage:
    def __init__(self, number: int):
        self.number = number
        self.commands: list[str] = []

    @staticmethod
    def _color(rgb: tuple[float, float, float]) -> str:
        return " ".join(f"{channel:.3f}" for channel in rgb)

    def text(self, x: float, top: float, value: object, size: float = 10, bold: bool = False,
             color: tuple[float, float, float] = INK) -> None:
        font = "F2" if bold else "F1"
        y = PAGE_HEIGHT - top - size
        self.commands.append(
            f"BT /{font} {size:.1f} Tf {self._color(color)} rg 1 0 0 1 {x:.1f} {y:.1f} Tm ({ascii_text(value)}) Tj ET"
        )

    def line(self, x1: float, top1: float, x2: float, top2: float,
             color: tuple[float, float, float] = GRID, width: float = 0.7) -> None:
        self.commands.append(
            f"{self._color(color)} RG {width:.1f} w {x1:.1f} {PAGE_HEIGHT-top1:.1f} m {x2:.1f} {PAGE_HEIGHT-top2:.1f} l S"
        )

    def rect(self, x: float, top: float, width: float, height: float,
             fill: tuple[float, float, float] | None = None,
             stroke: tuple[float, float, float] | None = None) -> None:
        y = PAGE_HEIGHT - top - height
        if fill:
            self.commands.append(f"{self._color(fill)} rg {x:.1f} {y:.1f} {width:.1f} {height:.1f} re f")
        if stroke:
            self.commands.append(f"{self._color(stroke)} RG 0.7 w {x:.1f} {y:.1f} {width:.1f} {height:.1f} re S")

    def paragraph(self, x: float, top: float, text: str, width: float,
                  size: float = 10, leading: float = 14,
                  color: tuple[float, float, float] = INK, bold: bool = False) -> float:
        max_chars = max(10, int(width / (size * 0.51)))
        lines: list[str] = []
        for paragraph in text.split("\n"):
            words = paragraph.split()
            line = ""
            for word in words:
                candidate = f"{line} {word}".strip()
                if len(candidate) > max_chars and line:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        for i, line in enumerate(lines):
            self.text(x, top + i * leading, line, size=size, bold=bold, color=color)
        return top + len(lines) * leading

    def footer(self) -> None:
        self.line(LEFT, 806, RIGHT, 806)
        self.text(LEFT, 815, "Vahan lead funnel analysis", size=8, color=MUTED)
        self.text(RIGHT - 18, 815, f"{self.number}", size=8, color=MUTED)


def write_pdf(output: Path, pages: list[PdfPage]) -> None:
    objects: list[bytes] = []
    page_ids = []
    page_tree_id = 3
    font_regular_id, font_bold_id = 4, 5
    next_id = 6
    for page in pages:
        content = "\n".join(page.commands).encode("latin-1")
        stream_id = next_id
        page_id = next_id + 1
        next_id += 2
        objects.append(f"{stream_id} 0 obj\n<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream\nendobj\n")
        objects.append(
            f"{page_id} 0 obj\n<< /Type /Page /Parent {page_tree_id} 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_regular_id} 0 R /F2 {font_bold_id} 0 R >> >> /Contents {stream_id} 0 R >>\nendobj\n".encode()
        )
        page_ids.append(page_id)
    page_tree = f"{page_tree_id} 0 obj\n<< /Type /Pages /Kids [{' '.join(f'{i} 0 R' for i in page_ids)}] /Count {len(page_ids)} >>\nendobj\n".encode()
    catalog_id = 1
    catalog = f"{catalog_id} 0 obj\n<< /Type /Catalog /Pages {page_tree_id} 0 R >>\nendobj\n".encode()
    info_id = 2
    info = f"{info_id} 0 obj\n<< /Title (Vahan Lead-Source Cohort Performance) /Author (Vahan lead funnel analysis) >>\nendobj\n".encode()
    font_regular = f"{font_regular_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode()
    font_bold = f"{font_bold_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n".encode()
    all_objects = [catalog, info, page_tree, font_regular, font_bold, *objects]
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for item in all_objects:
        offsets.append(len(pdf))
        pdf.extend(item)
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Size {len(offsets)} /Root {catalog_id} 0 R /Info {info_id} 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(pdf)


def add_page_header(page: PdfPage, title: str, subtitle: str | None = None) -> float:
    page.text(LEFT, 36, title, size=22, bold=True)
    if subtitle:
        page.paragraph(LEFT, 68, subtitle, RIGHT - LEFT, size=9, leading=13, color=MUTED)
        top = 96
    else:
        top = 75
    page.line(LEFT, top, RIGHT, top, color=BLUE, width=1.5)
    return top + 18


def add_table(page: PdfPage, top: float, headers: list[str], rows: list[list[str]], widths: list[float],
              row_height: float = 31, font_size: float = 8) -> float:
    height = row_height
    x = LEFT
    page.rect(LEFT, top, sum(widths), height, fill=INK)
    for i, (header, width) in enumerate(zip(headers, widths)):
        page.paragraph(x + 6, top + 7, header, width - 12, size=font_size, leading=font_size + 2, color=(1, 1, 1), bold=True)
        x += width
    y = top + height
    for row_index, row in enumerate(rows):
        x = LEFT
        page.rect(LEFT, y, sum(widths), row_height, fill=PALE if row_index % 2 else (1, 1, 1))
        page.line(LEFT, y + row_height, RIGHT, y + row_height)
        for value, width in zip(row, widths):
            page.paragraph(x + 6, y + 6, value, width - 12, size=font_size, leading=font_size + 1.5)
            x += width
        y += row_height
    return y


def add_rate_chart(page: PdfPage, top: float, cohorts: pd.DataFrame) -> float:
    """Draw a dot-and-interval chart of FT rates with Wilson confidence intervals."""
    page.text(LEFT, top, "FT rate by cohort", size=12, bold=True)
    page.text(LEFT, top + 17, "Points show the observed rate; lines show the 95% Wilson interval.", size=8, color=MUTED)
    plot_left, plot_right = 265, 500
    plot_top = top + 35
    entries = []
    for segment in ("cold", "warm"):
        entries.extend((segment, row) for _, row in cohorts[cohorts["segment"] == segment].head(3).iterrows())
    max_rate = 0.016
    scale = (plot_right - plot_left) / max_rate
    for tick in (0, 0.004, 0.008, 0.012, 0.016):
        x = plot_left + tick * scale
        page.line(x, plot_top - 3, x, plot_top + len(entries) * 16 - 2, color=GRID, width=0.45)
        page.text(x - 9, plot_top + len(entries) * 16 + 1, f"{tick:.1%}", size=7, color=MUTED)
    for index, (segment, row) in enumerate(entries):
        y = plot_top + index * 16 + 5
        label = str(row["lead_source"])
        label = label if len(label) < 28 else label[:25] + "..."
        page.text(LEFT, y - 5, label, size=7.3, color=INK)
        color = BLUE if segment == "cold" else ORANGE
        low = plot_left + min(float(row["ci_low"]), max_rate) * scale
        high = plot_left + min(float(row["ci_high"]), max_rate) * scale
        value = plot_left + min(float(row["ft_rate"]), max_rate) * scale
        page.line(low, y, high, y, color=color, width=1.8)
        page.line(low, y - 3, low, y + 3, color=color, width=1)
        page.line(high, y - 3, high, y + 3, color=color, width=1)
        page.rect(value - 3, y - 3, 6, 6, fill=color)
    axis_y = plot_top + len(entries) * 16 + 9
    page.line(plot_left, axis_y - 4, plot_right, axis_y - 4, color=INK, width=0.6)
    page.rect(LEFT, axis_y + 3, 7, 7, fill=BLUE)
    page.text(LEFT + 12, axis_y + 1, "Cold", size=7.5, color=MUTED)
    page.rect(LEFT + 58, axis_y + 3, 7, 7, fill=ORANGE)
    page.text(LEFT + 70, axis_y + 1, "Warm", size=7.5, color=MUTED)
    return axis_y + 13


def add_funnel_chart(page: PdfPage, top: float, cohorts: pd.DataFrame) -> float:
    """Draw top cold and warm cohorts across funnel stages on a logarithmic scale."""
    page.text(LEFT, top, "Funnel drop-off across the leading cohorts", size=12, bold=True)
    page.text(LEFT, top + 16, "Lead counts use a logarithmic scale so each funnel stage remains visible.", size=8, color=MUTED)
    leaders = pd.concat([
        cohorts[cohorts["segment"] == "cold"].head(3),
        cohorts[cohorts["segment"] == "warm"].head(3),
    ], ignore_index=True)
    colors_by_series = [BLUE, ORANGE, GREEN, RED, PURPLE, TEAL]
    plot_left, plot_right = 103, 365
    plot_top, plot_bottom = top + 31, top + 102
    max_log = 4.0
    stages = ["Leads", "Attempted", "Connected", "Interested", "FT"]
    xs = [plot_left + i * (plot_right - plot_left) / (len(stages) - 1) for i in range(len(stages))]
    for tick, label in [(0, "1"), (1, "10"), (2, "100"), (3, "1k"), (4, "10k")]:
        y = plot_bottom - tick / max_log * (plot_bottom - plot_top)
        page.line(plot_left, y, plot_right, y, color=GRID, width=0.5)
        page.text(plot_left - 25, y - 3, label, size=6.5, color=MUTED)
    for x, label in zip(xs, stages):
        page.line(x, plot_top, x, plot_bottom, color=GRID, width=0.4)
        page.text(x - (12 if len(label) < 7 else 22), plot_bottom + 5, label, size=7, color=MUTED)
    for index, (_, row) in enumerate(leaders.iterrows()):
        values = [int(row["leads"]), int(row["attempted"]), int(row["connected"]), int(row["interested"]), int(row["ft"])]
        points = []
        for x, value in zip(xs, values):
            log_value = min(max_log, max(0.0, math.log10(max(value, 1))))
            y = plot_bottom - log_value / max_log * (plot_bottom - plot_top)
            points.append((x, y))
        color = colors_by_series[index]
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            page.line(x1, y1, x2, y2, color=color, width=1.5)
        for x, y in points:
            page.rect(x - 2.2, y - 2.2, 4.4, 4.4, fill=color)
        label = str(row["lead_source"])
        label = label if len(label) <= 24 else label[:21] + "..."
        legend_y = plot_top + index * 12
        page.line(382, legend_y + 4, 394, legend_y + 4, color=color, width=2)
        page.text(399, legend_y, label, size=6.5, color=INK)
    return plot_bottom + 18


def make_report(input_csv: Path, output_pdf: Path) -> dict[str, int]:
    clean, stats = load_and_clean(input_csv)
    cohorts = cohort_table(clean)
    pages = [PdfPage(i) for i in range(1, 5)]

    # Page 1: executive summary and audit counts.
    page = pages[0]
    y = add_page_header(page, "Lead-source cohort performance", "Vahan sourcing and telecalling funnel | First Trip conversion analysis")
    y = page.paragraph(LEFT, y, "Executive summary", RIGHT - LEFT, size=14, leading=19, bold=True) + 7
    y = page.paragraph(LEFT, y,
        "This report compares lead sources by the share of uploaded leads that reached First Trip (FT). "
        "Warm and cold sources are ranked separately because they enter the funnel at different stages. "
        "Wilson confidence intervals and Fisher exact tests keep the small event counts in view.",
        RIGHT - LEFT, size=10, leading=15) + 10
    page.rect(LEFT, y, RIGHT - LEFT, 61, fill=PALE, stroke=GRID)
    page.rect(LEFT, y, 4, 61, fill=BLUE)
    page.paragraph(LEFT + 14, y + 12,
        "The two leading cold cohorts are statistically tied in this sample. Treat Single Referral > 7 days "
        "and Khanna - 2W 26th Jul as co-leaders, not a firm first and second.",
        RIGHT - LEFT - 28, size=10.5, leading=15, bold=True)
    y += 78
    page.text(LEFT, y, "Data integrity", size=13, bold=True)
    y += 23
    checks = [
        ["Source rows", f"{stats['raw_rows']:,}"],
        ["Rows after lead-level deduplication", f"{stats['unique_rows']:,}"],
        ["Repeated phone hashes", f"{stats['repeated_phone_hashes']:,} hashes across {stats['repeated_phone_rows']:,} rows"],
        ["Missing phone hashes", f"{stats['missing_phone_rows']:,}; kept separately"],
        ["FT events", f"{stats['raw_ft']:,} raw; {stats['clean_ft']:,} after deduplication"],
        ["FT with no logged connection", f"{stats['ft_anomalies']:,} raw records flagged"],
    ]
    y = add_table(page, y, ["Check", "Result"], checks, [175, RIGHT - LEFT - 175], row_height=29, font_size=8.5)
    page.paragraph(LEFT, y + 10,
        "Known phone hashes keep their most advanced recorded funnel state. Missing hashes cannot identify a "
        "person, so each row is retained. Inconsistent FT rows remain in cohort rates and are flagged.",
        RIGHT - LEFT, size=8.5, leading=12, color=MUTED)

    # Page 2: current ranking and confidence intervals.
    page = pages[1]
    y = add_page_header(page, "Cohort performance", f"FT rate = FT events / uploaded leads after deduplication | Cohorts below {MIN_LEADS} leads omitted")
    for segment, label in [("cold", "Top cold cohorts"), ("warm", "Top warm cohorts")]:
        page.text(LEFT, y, label, size=13, bold=True)
        y += 22
        subset = cohorts[cohorts["segment"] == segment].head(3)
        rows = []
        for rank, (_, row) in enumerate(subset.iterrows(), 1):
            rows.append([
                str(rank), str(row["lead_source"]), f"{int(row['leads']):,}", str(int(row["ft"])),
                f"{row['ft_rate']:.2%}", f"{row['ci_low']:.2%} - {row['ci_high']:.2%}",
            ])
        y = add_table(page, y, ["#", "Cohort", "Leads", "FT", "Rate", "95% Wilson CI"], rows,
                      [22, 205, 48, 32, 55, 125], row_height=42, font_size=7.7) + 19
    page.text(LEFT, y, "How to read this table", size=12, bold=True)
    y += 19
    y = page.paragraph(LEFT, y,
        "The first two cold cohorts have close observed rates and overlapping confidence intervals. The third "
        "cold cohort has very few FT events, so a small change in the records could materially change its rate. "
        "Warm cohorts are already pre-qualified and should not be ranked directly against cold acquisition sources.",
        RIGHT - LEFT, size=9.5, leading=14) + 12
    add_rate_chart(page, y, cohorts)

    # Page 3: pairwise significance and funnel readings.
    page = pages[2]
    y = add_page_header(page, "Pairwise comparisons", "Two-sided Fisher exact tests on FT vs. non-FT among the displayed top three in each segment")
    for segment, label in [("cold", "Cold cohorts"), ("warm", "Warm cohorts")]:
        page.text(LEFT, y, label, size=13, bold=True)
        y += 22
        subset = cohorts[cohorts["segment"] == segment].head(3).reset_index(drop=True)
        rows = []
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                a, b = subset.iloc[i], subset.iloc[j]
                table = [[int(a["ft"]), int(a["leads"] - a["ft"])], [int(b["ft"]), int(b["leads"] - b["ft"])]]
                p = float(fisher_exact(table).pvalue)
                rows.append([
                    f"{a['lead_source']} / {b['lead_source']}",
                    "<0.001" if p < 0.001 else f"{p:.3f}",
                    "No clear difference" if p >= 0.05 else "Different in this sample",
                ])
        y = add_table(page, y, ["Comparison", "p-value", "Reading"], rows,
                      [310, 70, RIGHT - LEFT - 380], row_height=41, font_size=7.7) + 18
    page.text(LEFT, y, "Operational read", size=13, bold=True)
    y += 22
    for message in [
        "Treat Single Referral > 7 days and Khanna - 2W 26th Jul as co-leading cold sources; the sample does not support a reliable ordering.",
        "The OLX cohort ranks third, but its estimate is based on four FT events. Check the flagged records before making a budget decision.",
        "Warm-source rates are directionally ranked only. The current sample does not separate the leading warm cohorts with confidence.",
    ]:
        y = page.paragraph(LEFT, y, "- " + message, RIGHT - LEFT, size=9, leading=13) + 7
    y = page.paragraph(LEFT, y + 2,
        "Pairwise p-values are unadjusted. A non-significant result is not proof that the cohort rates are equal.",
        RIGHT - LEFT, size=8.5, leading=12, color=MUTED) + 12
    add_funnel_chart(page, y, cohorts)

    # Page 4: repeatable method and model caveat.
    page = pages[3]
    y = add_page_header(page, "Method and limitations", "Generated directly from the CSV; no saved notebook output is used")
    sections = [
        ("Cohort method", "The generator deduplicates known phone hashes by keeping the row with the most advanced recorded funnel state. Missing phone hashes remain separate records. It aggregates by lead_source, splits warm sources by the PreOb and AI Connected name patterns, excludes cohorts below 100 leads, and calculates Wilson intervals."),
        ("Statistical comparison", "Fisher exact tests compare the FT and non-FT counts for each pair among the displayed top three in each segment. Tests are unadjusted and the small positive counts limit the conclusions."),
        ("Model interpretation", "The notebook's model includes call, interest and onboarding fields, but their timing relative to FT is not established in the supplied extract. Random cross-validation metrics and feature rankings are retrospective descriptions, not prospective estimates of who can be identified at upload. This report does not repeat them as operational performance claims."),
        ("Next analysis step", "Before lead scoring is considered, establish event timestamps and the exact decision point, then evaluate features available at that point on a later time period held out from training."),
        ("Source and definitions", f"Input file: {input_csv.name}. First Trip is represented by FT_after_upload. Minimum cohort size: {MIN_LEADS}. Confidence intervals: 95% Wilson score intervals. Pairwise tests: two-sided Fisher exact tests."),
    ]
    for heading, body in sections:
        page.text(LEFT, y, heading, size=12, bold=True)
        y += 19
        y = page.paragraph(LEFT, y, body, RIGHT - LEFT, size=9.5, leading=14) + 15
    page.text(LEFT, y, "Data quality follow-up", size=12, bold=True)
    y += 19
    page.paragraph(LEFT, y,
        "Investigate repeated source attribution and FT records without a logged connection in the upstream "
        "pipeline. Keep these checks in future refreshes so changes in the export are visible.",
        RIGHT - LEFT, size=9.5, leading=14)

    for pdf_page in pages:
        pdf_page.footer()
    write_pdf(output_pdf, pages)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Vahan cohort analysis PDF report.")
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("Vahan_Case_Study.csv"), help="Input CSV path")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("Vahan_Case_Study_Report.pdf"), help="Output PDF path")
    args = parser.parse_args()
    stats = make_report(args.input.resolve(), args.output.resolve())
    print(f"Report written to {args.output.resolve()}")
    print(f"Rows: {stats['raw_rows']:,} raw, {stats['unique_rows']:,} after deduplication; FT events: {stats['clean_ft']:,}")


if __name__ == "__main__":
    main()
