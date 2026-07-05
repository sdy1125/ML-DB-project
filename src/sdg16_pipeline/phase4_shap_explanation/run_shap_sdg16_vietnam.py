"""Generate SHAP-style XGBoost outputs for SDG16 Vietnam.

This runner intentionally avoids the external `shap` package because the project
environment already has XGBoost installed. XGBoost's Booster can return exact
tree SHAP contributions via `pred_contribs=True`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score


DATA_PATH = Path("data/clean/sdg16_spark.csv")
OUTPUT_DIR = Path("artifacts/shap")
SOURCE = "SDR2024"
TARGET = "goal16"

FEATURES = [
    "n_sdg16_admin",
    "n_sdg16_clabor",
    "n_sdg16_cpi",
    "n_sdg16_crime",
    "n_sdg16_crimepov",
    "n_sdg16_detain",
    "n_sdg16_exprop",
    "n_sdg16_homicide",
    "n_sdg16_homicides",
    "n_sdg16_justice",
    "n_sdg16_power",
    "n_sdg16_prs",
    "n_sdg16_rsf",
    "n_sdg16_safe",
    "n_sdg16_security",
    "n_sdg16_u5reg",
    "n_sdg16_weaponsexp",
]

FEATURE_LABELS = {
    "n_sdg16_admin": "Hành chính minh bạch",
    "n_sdg16_clabor": "Lao động trẻ em",
    "n_sdg16_cpi": "Chống tham nhũng (CPI)",
    "n_sdg16_crime": "Tội phạm chung",
    "n_sdg16_crimepov": "Tội phạm - nghèo đói",
    "n_sdg16_detain": "Tạm giam trước xét xử",
    "n_sdg16_exprop": "Chống tịch thu tài sản",
    "n_sdg16_homicide": "Tỷ lệ giết người",
    "n_sdg16_homicides": "Tử vong bạo lực",
    "n_sdg16_justice": "Tiếp cận tư pháp",
    "n_sdg16_power": "Quyền lực chia sẻ",
    "n_sdg16_prs": "Ổn định chính trị",
    "n_sdg16_rsf": "Tự do báo chí (RSF)",
    "n_sdg16_safe": "Cảm giác an toàn",
    "n_sdg16_security": "An ninh công cộng",
    "n_sdg16_u5reg": "Đăng ký khai sinh",
    "n_sdg16_weaponsexp": "Xuất khẩu vũ khí",
}

ASEAN_COUNTRIES = ["Thailand", "Indonesia", "Philippines", "Malaysia", "Singapore"]

BEST_PARAMS = {
    "max_depth": 5,
    "learning_rate": 0.0903,
    "n_estimators": 827,
    "subsample": 0.725,
    "colsample_bytree": 0.761,
    "reg_alpha": 1.738,
    "reg_lambda": 0.004,
    "random_state": 42,
    "tree_method": "hist",
    "objective": "reg:squarederror",
}


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def booster_contribs(model: xgb.XGBRegressor, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    matrix = xgb.DMatrix(frame, feature_names=FEATURES)
    contribs = model.get_booster().predict(matrix, pred_contribs=True)
    return contribs[:, :-1], contribs[:, -1]


def plot_bar_global(importance: pd.DataFrame) -> None:
    plot_df = importance.sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df["label"], plot_df["mean_abs_shap"], color="#4dabf7")
    ax.set_xlabel("Mean |SHAP contribution|")
    ax.set_title("Global SHAP importance - SDG16 XGBoost")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_bar_global.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_waterfall(year: int, years_vn: np.ndarray, x_vn: pd.DataFrame, y_vn: np.ndarray, pred_vn: np.ndarray, shap_vn: np.ndarray, base_value: float) -> None:
    if year not in years_vn:
        return
    idx = list(years_vn).index(year)
    values = shap_vn[idx]
    order = np.argsort(np.abs(values))[::-1][:12]

    running = base_value
    starts = []
    ends = []
    for contribution in values[order]:
        starts.append(running)
        running += contribution
        ends.append(running)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(order))
    for pos, feature_idx in enumerate(order):
        contribution = values[feature_idx]
        left = min(starts[pos], ends[pos])
        width = abs(contribution)
        color = "#2f9e44" if contribution >= 0 else "#e03131"
        ax.barh(y_pos[pos], width, left=left, color=color, alpha=0.85)
        ax.text(
            ends[pos],
            y_pos[pos],
            f"{contribution:+.2f}",
            va="center",
            ha="left" if contribution >= 0 else "right",
            fontsize=8,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([FEATURE_LABELS[FEATURES[index]] for index in order], fontsize=9)
    ax.axvline(base_value, color="gray", linestyle="--", label=f"base={base_value:.2f}")
    ax.axvline(float(pred_vn[idx]), color="#1971c2", label=f"pred={pred_vn[idx]:.2f}")
    ax.axvline(float(y_vn[idx]), color="black", linestyle=":", label=f"actual={y_vn[idx]:.2f}")
    ax.set_xlabel("Goal16 score")
    ax.set_title(f"Vietnam {year} SHAP waterfall")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"waterfall_vietnam_{year}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_gap(gap: pd.DataFrame) -> None:
    plot_df = gap[(gap["gap_vs_top20"].abs() > 2) | (gap["shap_vn2022"].abs() > 0.5)].copy()
    plot_df = plot_df.sort_values("gap_vs_top20", ascending=True)
    if plot_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, len(plot_df) * 0.5)))
    axes[0].barh(
        plot_df["label"],
        plot_df["gap_vs_top20"],
        color=["#e03131" if value > 0 else "#2f9e44" for value in plot_df["gap_vs_top20"]],
    )
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Gap vs Top-20")
    axes[0].set_xlabel("Top-20 mean - Vietnam")
    axes[0].grid(axis="x", alpha=0.25, linestyle="--")

    axes[1].barh(
        plot_df["label"],
        plot_df["shap_vn2022"],
        color=["#2f9e44" if value >= 0 else "#e03131" for value in plot_df["shap_vn2022"]],
    )
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Vietnam 2022 SHAP")
    axes[1].set_xlabel("SHAP contribution")
    axes[1].set_yticks([])
    axes[1].grid(axis="x", alpha=0.25, linestyle="--")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_gap_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_timeline(years_vn: np.ndarray, shap_vn: np.ndarray, importance: pd.DataFrame) -> None:
    top_features = importance["feature"].head(6).tolist()
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for ax, feature in zip(axes.flatten(), top_features):
        index = FEATURES.index(feature)
        ax.plot(years_vn, shap_vn[:, index], marker="o", linewidth=1.6)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        ax.set_title(FEATURE_LABELS[feature], fontsize=9)
        ax.grid(alpha=0.2, linestyle="--")
    plt.suptitle("Vietnam SHAP timeline")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_shap_timeline_vietnam.png", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df = df[df["source"] == SOURCE].copy()
    df = df.dropna(subset=[TARGET])

    train = df[df["Year"] <= 2018].copy()
    val = df[(df["Year"] >= 2019) & (df["Year"] <= 2021)].copy()
    test = df[df["Year"] >= 2022].copy()

    medians = train[FEATURES].median()
    for split in (train, val, test):
        split[FEATURES] = split[FEATURES].fillna(medians)

    model = xgb.XGBRegressor(**BEST_PARAMS)
    model.fit(
        train[FEATURES],
        train[TARGET],
        eval_set=[(val[FEATURES], val[TARGET])],
        verbose=False,
    )

    metrics = {}
    for name, split in [("validation", val), ("test", test)]:
        pred = model.predict(split[FEATURES])
        metrics[name] = {
            "r2": float(r2_score(split[TARGET], pred)),
            "rmse": rmse(split[TARGET], pred),
            "mae": float(np.mean(np.abs(split[TARGET] - pred))),
        }

    shap_train, base_train = booster_contribs(model, train[FEATURES])
    shap_test, _ = booster_contribs(model, test[FEATURES])
    base_value = float(np.mean(base_train))

    vn_df = df[df["Country"] == "Vietnam"].sort_values("Year").copy()
    vn_df[FEATURES] = vn_df[FEATURES].fillna(medians)
    x_vn = vn_df[FEATURES]
    y_vn = vn_df[TARGET].to_numpy()
    years_vn = vn_df["Year"].to_numpy()
    shap_vn, _ = booster_contribs(model, x_vn)
    pred_vn = model.predict(x_vn)

    np.save(OUTPUT_DIR / "shap_values_train.npy", shap_train)
    np.save(OUTPUT_DIR / "shap_values_vietnam.npy", shap_vn)

    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "label": [FEATURE_LABELS[feature] for feature in FEATURES],
            "mean_abs_shap": np.abs(shap_train).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(OUTPUT_DIR / "global_shap_importance.csv", index=False)
    plot_bar_global(importance)

    df_2022 = df[df["Year"] == 2022].copy()
    df_2022[FEATURES] = df_2022[FEATURES].fillna(medians)
    top20 = df_2022.nlargest(20, TARGET)
    asean = df_2022[df_2022["Country"].isin(ASEAN_COUNTRIES)]
    idx_2022 = list(years_vn).index(2022) if 2022 in years_vn else -1
    vn_2022 = x_vn.iloc[idx_2022]
    shap_2022 = shap_vn[idx_2022]

    gap = pd.DataFrame(
        {
            "feature": FEATURES,
            "label": [FEATURE_LABELS[feature] for feature in FEATURES],
            "vn_2022": vn_2022.values,
            "top20_mean": top20[FEATURES].mean().values,
            "asean_mean": asean[FEATURES].mean().values if not asean.empty else np.nan,
            "shap_vn2022": shap_2022,
        }
    )
    gap["gap_vs_top20"] = gap["top20_mean"] - gap["vn_2022"]
    gap["gap_vs_asean"] = gap["asean_mean"] - gap["vn_2022"]
    gap = gap.sort_values("shap_vn2022")
    gap.to_csv(OUTPUT_DIR / "gap_analysis.csv", index=False)
    plot_gap(gap)

    plot_waterfall(2022, years_vn, x_vn, y_vn, pred_vn, shap_vn, base_value)
    plot_waterfall(2023, years_vn, x_vn, y_vn, pred_vn, shap_vn, base_value)
    plot_timeline(years_vn, shap_vn, importance)

    increments = [5, 10, 15, 20, 30]
    priority_features = gap.head(5)["feature"].tolist()
    base_pred = float(pred_vn[idx_2022])
    rows = []
    for feature in priority_features:
        for increment in increments:
            x_cf = x_vn.iloc[[idx_2022]].copy()
            x_cf[feature] = min(100.0, float(x_cf[feature].iloc[0]) + increment)
            new_pred = float(model.predict(x_cf)[0])
            rows.append(
                {
                    "feature": feature,
                    "label": FEATURE_LABELS[feature],
                    "increment": increment,
                    "delta_goal16": round(new_pred - base_pred, 4),
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "counterfactual.csv", index=False)

    summary = {
        "source": SOURCE,
        "rows": int(len(df)),
        "countries": int(df["Country"].nunique()),
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "base_value": base_value,
        "metrics": metrics,
        "vietnam_latest_year": int(years_vn[-1]),
        "vietnam_latest_actual": float(y_vn[-1]),
        "vietnam_latest_predicted": float(pred_vn[-1]),
        "top_negative_vn2022": gap.head(5)[["feature", "label", "shap_vn2022"]].to_dict("records"),
        "top_global_importance": importance.head(10).to_dict("records"),
    }
    (OUTPUT_DIR / "shap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"SHAP outputs written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
