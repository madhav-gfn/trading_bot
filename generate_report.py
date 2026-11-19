"""Generate a PDF cohort report from the Vahan case study CSV."""

from __future__ import annotations

import argparse
import math
import re
import unicodedata
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def analyze_models(clean: pd.DataFrame) -> dict[str, object]:
    """Reproduce the notebook's exploratory model comparisons from the cleaned CSV."""
    model_data = clean[~clean["ft_anomaly"]].copy()
    model_data["upload_date_parsed"] = pd.to_datetime(model_data["upload_date"], format="%d-%m-%Y")
    model_data["upload_dow"] = model_data["upload_date_parsed"].dt.day_name()
    model_data["attempt_per_lead_filled"] = model_data["Attempt per Lead"].fillna(0)
    categorical = ["lead_source", "upload_dow"]
    numeric = [
        "Attempted", "Connected", "Interested", "OB_after_upload",
        "OB_after_first_attempt", "tag_filled", "attempt_per_lead_filled",
    ]
    X = model_data[categorical + numeric]
    y = model_data["FT_after_upload"].astype(int)
    scaled = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numeric),
    ])
    raw = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ], remainder="passthrough")
    candidates = {
        "Logistic Regression": Pipeline([
            ("pre", scaled),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("pre", raw),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=42)),
        ]),
        "Gradient Boosting": Pipeline([
            ("pre", raw),
            ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)),
        ]),
        "Dummy baseline": Pipeline([
            ("pre", raw),
            ("clf", DummyClassifier(strategy="stratified", random_state=42)),
        ]),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: dict[str, dict[str, object]] = {}
    for name, pipeline in candidates.items():
        probabilities = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
        order = probabilities.argsort()[::-1]
        top_count = max(1, int(len(y) * 0.01))
        captured = int(y.to_numpy()[order[:top_count]].sum())
        results[name] = {
            "roc_auc": float(roc_auc_score(y, probabilities)),
            "pr_auc": float(average_precision_score(y, probabilities)),
            "top1_capture": captured / int(y.sum()),
            "captured": captured,
            "probabilities": probabilities,
        }

    logistic = candidates["Logistic Regression"]
    cv_predictions = cross_val_predict(logistic, X, y, cv=cv, method="predict")
    matrix = confusion_matrix(y, cv_predictions, labels=[0, 1])
    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    logistic.fit(train_x, train_y)
    holdout_predictions = logistic.predict(test_x)
    holdout = {
        "n": len(test_y),
        "positives": int(test_y.sum()),
        "precision": float(precision_score(test_y, holdout_predictions, zero_division=0)),
        "recall": float(recall_score(test_y, holdout_predictions, zero_division=0)),
        "f1": float(2 * precision_score(test_y, holdout_predictions, zero_division=0) * recall_score(test_y, holdout_predictions, zero_division=0) /
                     max(precision_score(test_y, holdout_predictions, zero_division=0) + recall_score(test_y, holdout_predictions, zero_division=0), 1e-12)),
    }

    logistic.fit(X, y)
    logistic_names = logistic.named_steps["pre"].get_feature_names_out()
    coefficients = logistic.named_steps["clf"].coef_[0]
    logistic_features = sorted(
        [(str(name).split("__", 1)[-1], float(value)) for name, value in zip(logistic_names, coefficients)],
        key=lambda item: abs(item[1]), reverse=True,
    )[:8]

    forest = candidates["Random Forest"]
    forest.fit(X, y)
    forest_names = forest.named_steps["pre"].get_feature_names_out()
    importances = forest.named_steps["clf"].feature_importances_
    grouped_importance: dict[str, float] = {}
    for feature_name, importance in zip(forest_names, importances):
        clean_name = str(feature_name).split("__", 1)[-1]
        base_feature = next((feature for feature in categorical + numeric if clean_name == feature or clean_name.startswith(feature + "_")), clean_name)
        grouped_importance[base_feature] = grouped_importance.get(base_feature, 0.0) + float(importance)
    forest_features = sorted(grouped_importance.items(), key=lambda item: item[1], reverse=True)[:8]

    probabilities = results["Logistic Regression"]["probabilities"]
    fpr, tpr, _ = roc_curve(y, probabilities)
    pr_precision, pr_recall, _ = precision_recall_curve(y, probabilities)
    order = probabilities.argsort()[::-1]
    ordered_y = y.to_numpy()[order]
    gains = []
    for fraction in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0):
        count = max(1, int(len(ordered_y) * fraction))
        gains.append((fraction, int(ordered_y[:count].sum()) / int(y.sum())))

    return {
        "n": len(y), "positives": int(y.sum()), "positive_rate": float(y.mean()),
        "results": results, "matrix": matrix, "holdout": holdout,
        "precision": pr_precision, "recall": pr_recall, "fpr": fpr, "tpr": tpr,
        "gains": gains, "logistic_features": logistic_features, "forest_features": forest_features,
    }


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

    def text(self, x: float, top: float, value: object, size: float = 10, bold: bool = False, mono: bool = False,
             color: tuple[float, float, float] = INK) -> None:
        font = "F3" if mono else ("F2" if bold else "F1")
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
    font_regular_id, font_bold_id, font_mono_id = 4, 5, 6
    next_id = 7
    for page in pages:
        content = "\n".join(page.commands).encode("latin-1")
        stream_id = next_id
        page_id = next_id + 1
        next_id += 2
        objects.append(f"{stream_id} 0 obj\n<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream\nendobj\n")
        objects.append(
            f"{page_id} 0 obj\n<< /Type /Page /Parent {page_tree_id} 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_regular_id} 0 R /F2 {font_bold_id} 0 R /F3 {font_mono_id} 0 R >> >> /Contents {stream_id} 0 R >>\nendobj\n".encode()
        )
        page_ids.append(page_id)
    page_tree = f"{page_tree_id} 0 obj\n<< /Type /Pages /Kids [{' '.join(f'{i} 0 R' for i in page_ids)}] /Count {len(page_ids)} >>\nendobj\n".encode()
    catalog_id = 1
    catalog = f"{catalog_id} 0 obj\n<< /Type /Catalog /Pages {page_tree_id} 0 R >>\nendobj\n".encode()
    info_id = 2
    info = f"{info_id} 0 obj\n<< /Title (Vahan Lead-Source Cohort Performance) /Author (Vahan lead funnel analysis) >>\nendobj\n".encode()
    font_regular = f"{font_regular_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n".encode()
    font_bold = f"{font_bold_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>\nendobj\n".encode()
    font_mono = f"{font_mono_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>\nendobj\n".encode()
    all_objects = [catalog, info, page_tree, font_regular, font_bold, font_mono, *objects]
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


def add_curve_chart(page: PdfPage, x: float, top: float, width: float, height: float,
                    title: str, x_values: object, y_values: object,
                    x_label: str, y_label: str, baseline: float | None = None) -> None:
    page.text(x, top, title, size=10, bold=True)
    left, right = x + 31, x + width - 8
    plot_top, plot_bottom = top + 20, top + height - 24
    for tick in (0, 0.5, 1):
        gy = plot_bottom - tick * (plot_bottom - plot_top)
        gx = left + tick * (right - left)
        page.line(left, gy, right, gy, color=GRID, width=0.45)
        page.line(gx, plot_top, gx, plot_bottom, color=GRID, width=0.45)
        page.text(left - 18, gy - 3, f"{tick:.1f}", size=6, color=MUTED)
        page.text(gx - 6, plot_bottom + 3, f"{tick:.1f}", size=6, color=MUTED)
    if baseline is not None:
        by = plot_bottom - baseline * (plot_bottom - plot_top)
        page.line(left, by, right, by, color=ORANGE, width=0.8)
    xs = np.asarray(x_values, dtype=float)
    ys = np.asarray(y_values, dtype=float)
    if len(xs) > 180:
        indices = np.linspace(0, len(xs) - 1, 180).astype(int)
        xs, ys = xs[indices], ys[indices]
    points = [
        (left + float(a) * (right - left), plot_bottom - float(b) * (plot_bottom - plot_top))
        for a, b in zip(xs, ys)
    ]
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        page.line(x1, y1, x2, y2, color=BLUE, width=1.3)
    page.text(left + 16, plot_bottom + 13, x_label, size=6.5, color=MUTED)
    page.text(x + 1, plot_top + 3, y_label, size=6.5, color=MUTED)


def add_gain_chart(page: PdfPage, top: float, gains: list[tuple[float, float]]) -> None:
    page.text(LEFT, top, "Cumulative capture when leads are ranked by model score", size=11, bold=True)
    page.text(LEFT, top + 15, "Retrospective out-of-fold estimate; not an intake-time forecast.", size=8, color=MUTED)
    left, right = LEFT + 42, RIGHT - 8
    plot_top, plot_bottom = top + 29, top + 112
    for tick in (0, 0.5, 1):
        gx = left + tick * (right - left)
        gy = plot_bottom - tick * (plot_bottom - plot_top)
        page.line(left, gy, right, gy, color=GRID, width=0.5)
        page.line(gx, plot_top, gx, plot_bottom, color=GRID, width=0.5)
        page.text(gx - 7, plot_bottom + 3, f"{tick:.0%}", size=7, color=MUTED)
        page.text(left - 30, gy - 3, f"{tick:.0%}", size=7, color=MUTED)
    page.line(left, plot_bottom, right, plot_top, color=ORANGE, width=0.8)
    points = [(left + fraction * (right - left), plot_bottom - capture * (plot_bottom - plot_top)) for fraction, capture in gains]
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        page.line(x1, y1, x2, y2, color=BLUE, width=1.8)
    for x, y in points:
        page.rect(x - 2.2, y - 2.2, 4.4, 4.4, fill=BLUE)


def add_feature_chart(page: PdfPage, x: float, top: float, width: float,
                      title: str, features: list[tuple[str, float]], signed: bool = False) -> None:
    page.text(x, top, title, size=10, bold=True)
    entries = features[:7]
    max_value = max((abs(value) for _, value in entries), default=1.0) or 1.0
    bar_x = x + 112
    bar_width = width - 143
    for index, (name, value) in enumerate(entries):
        y = top + 24 + index * 19
        label = name if len(name) < 19 else name[:16] + "..."
        page.text(x, y, label, size=7, color=INK)
        length = abs(value) / max_value * bar_width
        color = (BLUE if value >= 0 else RED) if signed else GREEN
        page.rect(bar_x, y + 2, length, 7, fill=color)
        page.text(bar_x + bar_width + 5, y, f"{value:+.2f}" if signed else f"{value:.2f}", size=6.7, color=MUTED)


def make_report(input_csv: Path, output_pdf: Path) -> dict[str, int]:
    clean, stats = load_and_clean(input_csv)
    cohorts = cohort_table(clean)
    models = analyze_models(clean)
    pages = [PdfPage(i) for i in range(1, 9)]

    # Page 1: executive summary and audit counts.
    page = pages[0]
    y = add_page_header(page, "Lead-source cohort performance", "Vahan sourcing and telecalling funnel | First Trip conversion analysis")
    y = page.paragraph(LEFT, y, "Executive summary", RIGHT - LEFT, size=14, leading=19, bold=True) + 7
    y = page.paragraph(LEFT, y,
        "This report answers three questions: which lead-source cohorts have the strongest observed FT rate, "
        "how the SQL aggregation reproduces that ranking, and what an exploratory model associates with FT. "
        "Warm and cold sources are ranked separately. Confidence intervals, tests and event counts keep the "
        "small sample in view.",
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
    page.paragraph(LEFT, 692,
        f"Model sample: {models['positives']} FT events in {models['n']:,} records ({models['positive_rate']:.2%}). "
        "Model metrics are retrospective: several inputs may only be known after upload or near the FT outcome.",
        RIGHT - LEFT, size=9, leading=13, color=MUTED)

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

    # Page 4: SQL equivalent and grouped output.
    page = pages[3]
    y = add_page_header(page, "SQL aggregation", "The same cohort question expressed as a window-function query")
    y = page.paragraph(LEFT, y,
        "The query treats the raw CSV as a leads table, keeps each known phone hash's most advanced funnel state, "
        "preserves rows with missing identifiers, assigns warm or cold status, and ranks sources within each segment.",
        RIGHT - LEFT, size=9, leading=13) + 11
    code_lines = [
        "WITH ranked AS (",
        "  SELECT *, ROW_NUMBER() OVER (",
        "    PARTITION BY COALESCE(candidate_phone, 'missing_' || rowid)",
        "    ORDER BY FT_after_upload DESC, Interested DESC,",
        "             Connected DESC, Attempted DESC) AS rn",
        "  FROM leads",
        "), one_lead AS (SELECT * FROM ranked WHERE rn = 1),",
        "cohort AS (",
        "  SELECT lead_source, SUM(\"Uploaded Leads\") AS leads,",
        "    CASE WHEN lead_source LIKE '%PreOb%' OR",
        "      lead_source LIKE '%AI Connected%' THEN 'warm'",
        "      ELSE 'cold' END AS segment,",
        "    SUM(FT_after_upload) AS ft,",
        "    100.0 * SUM(FT_after_upload) / SUM(\"Uploaded Leads\") AS ft_rate",
        "  FROM one_lead GROUP BY lead_source",
        "  HAVING SUM(\"Uploaded Leads\") >= 100",
        ")",
        "SELECT *, RANK() OVER (PARTITION BY segment",
        "  ORDER BY ft_rate DESC) AS rank_in_segment FROM cohort;",
    ]
    code_top, code_height = y, len(code_lines) * 13 + 15
    page.rect(LEFT, code_top, RIGHT - LEFT, code_height, fill=PALE, stroke=GRID)
    for index, line in enumerate(code_lines):
        page.text(LEFT + 11, code_top + 8 + index * 13, line, size=7.8, mono=True, color=INK)
    y = code_top + code_height + 15
    page.text(LEFT, y, "Top three results by segment", size=12, bold=True)
    y += 21
    sql_rows = []
    for segment in ("cold", "warm"):
        for rank, (_, row) in enumerate(cohorts[cohorts["segment"] == segment].head(3).iterrows(), 1):
            sql_rows.append([
                segment.title(), str(rank), str(row["lead_source"]), f"{int(row['leads']):,}",
                str(int(row["attempted"])), str(int(row["connected"])), str(int(row["interested"])),
                str(int(row["ft"])), f"{row['ft_rate']:.3%}",
            ])
    y = add_table(page, y, ["Segment", "Rank", "Lead source", "Leads", "Attempt", "Connect", "Interest", "FT", "FT rate"],
                  sql_rows, [42, 28, 174, 46, 43, 43, 43, 25, 43], row_height=39, font_size=6.7)
    page.paragraph(LEFT, y + 12,
        "The table reports deduplicated funnel counts. The FT rate ranking matches the cohort section; the "
        "confidence intervals and anomaly flags are calculated alongside the SQL-style aggregation.",
        RIGHT - LEFT, size=8.5, leading=12, color=MUTED)

    # Page 5: model question, design, and candidate comparison.
    page = pages[4]
    y = add_page_header(page, "Exploratory model", "Which recorded fields are associated with First Trip?")
    y = page.paragraph(LEFT, y,
        f"Target: FT_after_upload. The model uses {models['n']:,} deduplicated records after excluding "
        f"{stats['ft_anomalies']} raw FT-without-connection anomalies; the remaining target has "
        f"{models['positives']} positive examples ({models['positive_rate']:.2%}). Five-fold stratified "
        "cross-validation is used because one small holdout contains very few FT events.",
        RIGHT - LEFT, size=9.5, leading=14) + 9
    page.text(LEFT, y, "Candidate model comparison", size=12, bold=True)
    y += 21
    metric_rows = []
    ordered_models = sorted(models["results"].items(), key=lambda item: item[1]["pr_auc"], reverse=True)
    for name, result in ordered_models:
        metric_rows.append([
            name, f"{result['roc_auc']:.3f}", f"{result['pr_auc']:.3f}",
            f"{result['top1_capture']:.1%}",
        ])
    y = add_table(page, y, ["Model", "ROC-AUC", "PR-AUC", "Top 1% FT capture"], metric_rows,
                  [210, 75, 75, RIGHT - LEFT - 360], row_height=36, font_size=8)
    y += 14
    for heading, body in [
        ("Candidate choice", "Logistic regression is included for readable coefficient directions. Random forest provides a non-linear comparison; gradient boosting tests another tree approach; the stratified dummy model is the no-signal reference."),
        ("Fields used", "Lead source, upload weekday, Attempted, Connected, Interested, onboarding flags, tag_filled, and Attempt per Lead. Percentage columns and FT_after_first_attempt are excluded because they encode the target or a closely related outcome. The unreliable timing field is also excluded."),
        ("Timing limitation", "The extract does not establish that call, interest, or onboarding fields are known before FT or at the time of lead assignment. Random cross-validation cannot fix this. Scores and capture rates below describe retrospective ranking only; they are not an operational forecast."),
    ]:
        page.text(LEFT, y, heading, size=11, bold=True)
        y += 17
        y = page.paragraph(LEFT, y, body, RIGHT - LEFT, size=8.8, leading=12.5) + 10

    # Page 6: classification detail, curves, and gains.
    page = pages[5]
    y = add_page_header(page, "Model evaluation", "Out-of-fold predictions from five-fold stratified cross-validation")
    matrix = models["matrix"]
    page.text(LEFT, y, "Logistic regression confusion matrix", size=11, bold=True)
    y += 19
    matrix_rows = [
        ["Actual / predicted", "No FT", "FT"],
        ["No FT", f"{int(matrix[0,0]):,}", f"{int(matrix[0,1]):,}"],
        ["FT", f"{int(matrix[1,0]):,}", f"{int(matrix[1,1]):,}"],
    ]
    y = add_table(page, y, matrix_rows[0], matrix_rows[1:], [190, 130, RIGHT - LEFT - 320], row_height=27, font_size=8)
    y += 6
    holdout = models["holdout"]
    page.paragraph(LEFT, y,
        f"Single 80/20 holdout: {holdout['n']:,} test records with {holdout['positives']} FT events; "
        f"FT precision {holdout['precision']:.3f}, recall {holdout['recall']:.3f}, F1 {holdout['f1']:.3f}. "
        "A few positives moving between folds can materially change these values.",
        RIGHT - LEFT, size=8.5, leading=12, color=MUTED)
    chart_top = 254
    page.text(LEFT, chart_top - 12, "Threshold-independent ranking views", size=11, bold=True)
    logistic_result = models["results"]["Logistic Regression"]
    add_curve_chart(page, LEFT, chart_top, 225, 145, f"ROC (AUC {logistic_result['roc_auc']:.3f})",
                    models["fpr"], models["tpr"], "False positive rate", "True positive rate")
    add_curve_chart(page, LEFT + 250, chart_top, 225, 145, f"Precision-recall (AP {logistic_result['pr_auc']:.3f})",
                    models["recall"], models["precision"], "Recall", "Precision", baseline=models["positive_rate"])
    gain_top = 428
    add_gain_chart(page, gain_top, models["gains"])
    one_percent = logistic_result["top1_capture"]
    page.paragraph(LEFT, 573,
        f"In this retrospective cross-validation, the highest-scored 1% contains {logistic_result['captured']} of "
        f"{models['positives']} FT events ({one_percent:.1%}). This is not evidence of equivalent capture in live calling.",
        RIGHT - LEFT, size=8.8, leading=12, color=MUTED)

    # Page 7: model coefficients and feature importances.
    page = pages[6]
    y = add_page_header(page, "Factors associated with FT", "Logistic regression coefficients and random forest importances")
    add_feature_chart(page, LEFT, y + 4, 235, "Logistic regression", models["logistic_features"], signed=True)
    add_feature_chart(page, LEFT + 250, y + 4, 235, "Random forest", models["forest_features"], signed=False)
    page.text(LEFT, y + 174, "How to interpret these rankings", size=11, bold=True)
    page.paragraph(LEFT, y + 194,
        "A positive logistic coefficient means a higher fitted association with FT within this model; a negative "
        "coefficient means the reverse. Random forest importance indicates how much a field contributes to tree "
        "splits, not its direction. Correlated and nested funnel fields can trade off coefficient weight. Neither "
        "model establishes a causal effect, and their shared use of downstream fields can inflate apparent signal.",
        RIGHT - LEFT, size=9, leading=13)
    page.text(LEFT, y + 280, "Feature timing remains unresolved", size=11, bold=True)
    page.paragraph(LEFT, y + 300,
        "OB_after_upload is an outcome-stage field, while Connected, Interested and tag_filled may be logged after "
        "a call. Their association is useful for describing this dataset, not for deciding which newly uploaded lead "
        "to call first. A prospective model needs timestamped features and a time-based holdout.",
        RIGHT - LEFT, size=9, leading=13, color=MUTED)

    # Page 8: recommendations and next steps.
    page = pages[7]
    y = add_page_header(page, "Recommendations", "Actions supported by the current evidence")
    olx = cohorts[(cohorts["segment"] == "cold") & cohorts["lead_source"].astype(str).str.contains("OLX", case=False)].head(1)
    olx_attempt = f" Its attempt rate is {olx.iloc[0]['attempt_rate']:.1%}." if len(olx) else ""
    recommendations = [
        ("Fix source attribution and event logging", "Investigate repeated phone hashes and FT records with no logged connection upstream. Keep missing phone hashes distinct unless an alternative stable identifier is available."),
        ("Treat the two leading cold sources as co-leaders", "Single Referral > 7 days and Khanna - 2W 26th Jul have similar FT rates and Fisher exact p = 1.0. A larger sample is needed before a budget decision depends on their ordering."),
        ("Review OLX attempt coverage before scaling", f"OLX has four FT events and three flagged connection anomalies.{olx_attempt} Review the early-funnel handoff before drawing a channel-quality conclusion."),
        ("Use the model section as retrospective analysis", "The model surfaces associations such as onboarding, connection and tag completion, but timing and causality are unverified. Do not filter leads or promise conversion capture from these scores."),
        ("Re-evaluate with timestamped data", "Define the decision point, retain only features known then, and hold out a later date range. Accumulate more FT outcomes before interpreting model differences as stable."),
    ]
    for index, (heading, body) in enumerate(recommendations, 1):
        page.rect(LEFT, y + 2, 22, 22, fill=BLUE)
        page.text(LEFT + 7, y + 7, str(index), size=9, color=(1, 1, 1), bold=True)
        page.text(LEFT + 34, y, heading, size=11, bold=True)
        y = page.paragraph(LEFT + 34, y + 17, body, RIGHT - LEFT - 34, size=9, leading=13) + 14
    page.line(LEFT, y + 1, RIGHT, y + 1)
    page.paragraph(LEFT, y + 12,
        f"Source: {input_csv.name}. FT means First Trip (`FT_after_upload`). Cohorts under {MIN_LEADS} leads are excluded. "
        "Wilson intervals are 95%; pairwise p-values are two-sided Fisher exact tests. Findings describe this extract.",
        RIGHT - LEFT, size=8, leading=11, color=MUTED)

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
