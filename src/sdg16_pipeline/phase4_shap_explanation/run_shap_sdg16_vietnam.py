"""Generate XGBoost tree-contribution outputs for SDG16 Vietnam.

The target ``goal16`` is a composite score constructed from the SDG16 component
indicators used as model inputs. This runner is therefore framed as
composite-score reconstruction and diagnostic decomposition, not as independent
causal governance prediction.

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

try:
    import shap  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    shap = None


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


def candidate_params() -> list[dict]:
    """Small deterministic tuning grid for the main XGBoost runner.

    This avoids adding Optuna as a hard dependency while still giving the
    project a reproducible tuning step before SHAP/export.
    """

    candidates = [dict(BEST_PARAMS)]
    for max_depth in [3, 4, 5]:
        for learning_rate in [0.045, 0.065, 0.09]:
            for n_estimators in [450, 650, 850]:
                candidates.append(
                    {
                        **BEST_PARAMS,
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "n_estimators": n_estimators,
                        "subsample": 0.82,
                        "colsample_bytree": 0.82,
                        "reg_alpha": 0.5,
                        "reg_lambda": 1.0,
                    }
                )
    return candidates


def tune_xgboost(train: pd.DataFrame, val: pd.DataFrame) -> tuple[xgb.XGBRegressor, dict, list[dict]]:
    tuning_rows = []
    best_model = None
    best_params = None
    best_rmse = float("inf")

    for index, params in enumerate(candidate_params(), start=1):
        model = xgb.XGBRegressor(**params)
        model.fit(
            train[FEATURES],
            train[TARGET],
            eval_set=[(val[FEATURES], val[TARGET])],
            verbose=False,
        )
        pred = model.predict(val[FEATURES])
        score = rmse(val[TARGET], pred)
        row = {
            "candidate": index,
            "validation_rmse": score,
            "validation_r2": float(r2_score(val[TARGET], pred)),
            **{
                key: params[key]
                for key in [
                    "max_depth",
                    "learning_rate",
                    "n_estimators",
                    "subsample",
                    "colsample_bytree",
                    "reg_alpha",
                    "reg_lambda",
                ]
            },
        }
        tuning_rows.append(row)
        if score < best_rmse:
            best_rmse = score
            best_model = model
            best_params = params

    assert best_model is not None and best_params is not None
    return best_model, best_params, tuning_rows


def correlation_leakage_report(df: pd.DataFrame) -> dict:
    """Check whether XGBoost features look like target leakage.

    The runner deliberately trains only on the predefined component indicators
    in ``FEATURES``. This report records correlations with the target, exact
    duplicate checks, and any suspicious high-correlation columns so reviewers
    can see whether the SHAP model is explaining true component indicators or a
    leaked copy of ``goal16``. It also records the larger methodological caveat:
    ``goal16`` is itself a composite of SDG16 component indicators, so strong
    accuracy is reconstruction accuracy rather than independent causal evidence.
    """
    numeric = df[[TARGET] + FEATURES].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False)

    duplicate_features: list[str] = []
    for feature in FEATURES:
        pair = numeric[[TARGET, feature]].dropna()
        if not pair.empty and float((pair[TARGET] - pair[feature]).abs().max()) < 1e-9:
            duplicate_features.append(feature)

    numeric_all = df.select_dtypes(include=[np.number]).copy()
    suspicious_all = {}
    if TARGET in numeric_all.columns:
        all_corr = numeric_all.corr(numeric_only=True)[TARGET].drop(TARGET).dropna()
        suspicious_all = {
            col: float(value)
            for col, value in all_corr.sort_values(key=lambda s: s.abs(), ascending=False).items()
            if abs(float(value)) >= 0.98
        }

    high_corr_features = {
        feature: float(value)
        for feature, value in corr.items()
        if abs(float(value)) >= 0.98
    }

    report_rows = pd.DataFrame(
        {
            "feature": corr.index,
            "correlation_with_goal16": corr.values,
            "abs_correlation_with_goal16": np.abs(corr.values),
            "exact_duplicate_of_target": [feature in duplicate_features for feature in corr.index],
            "used_for_training": True,
        }
    )
    report_rows.to_csv(OUTPUT_DIR / "leakage_correlation_report.csv", index=False)

    return {
        "target": TARGET,
        "feature_policy": (
            "Only predefined n_sdg16_* component indicators are used. "
            "Target, ranking, aggregate and non-feature numeric columns are excluded from training. "
            "This removes direct duplicate-target leakage but does not remove the conceptual "
            "circularity that goal16 is constructed from SDG16 component indicators."
        ),
        "circular_target_warning": (
            "goal16 is a composite score constructed from the n_sdg16_* component indicators. "
            "High XGBoost accuracy should be interpreted as composite-score reconstruction and "
            "indicator decomposition, not as causal or independent predictive validity."
        ),
        "recommended_framing": "composite_score_reconstruction_and_indicator_decomposition",
        "top_abs_correlations": report_rows.head(10).to_dict("records"),
        "high_corr_features_abs_ge_0_98": high_corr_features,
        "exact_duplicate_features": duplicate_features,
        "suspicious_numeric_columns_abs_ge_0_98": suspicious_all,
        "leakage_excluded_features": duplicate_features,
        "leakage_status": "review" if duplicate_features else "no_exact_target_duplicate_detected",
    }


def booster_contribs(model: xgb.XGBRegressor, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    matrix = xgb.DMatrix(frame, feature_names=FEATURES)
    contribs = model.get_booster().predict(matrix, pred_contribs=True)
    return contribs[:, :-1], contribs[:, -1]


def tree_contribs(model: xgb.XGBRegressor, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    """Return tree contributions, preferring the official SHAP package.

    XGBoost's ``pred_contribs=True`` returns Tree SHAP contributions from the
    booster itself. When the optional ``shap`` package is installed, we use
    ``shap.TreeExplainer`` explicitly so the paper can state the standard SHAP
    implementation. The fallback keeps the project runnable in lightweight
    Docker/local environments.
    """

    if shap is not None:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(frame)
        base = explainer.expected_value
        if isinstance(values, list):
            values = values[0]
        if isinstance(base, (list, np.ndarray)):
            base_arr = np.full(len(frame), float(np.asarray(base).ravel()[0]), dtype=float)
        else:
            base_arr = np.full(len(frame), float(base), dtype=float)
        return np.asarray(values, dtype=float), base_arr, "shap_treeexplainer"

    contribs, base = booster_contribs(model, frame)
    return contribs, base, "xgboost_builtin_tree_contributions_pred_contribs"


def add_data_quality_flags(gap: pd.DataFrame) -> pd.DataFrame:
    """Flag suspicious zero-valued Vietnam indicators before headline ranking."""

    flagged = gap.copy()
    flagged["data_quality_flag"] = ""
    flagged["headline_eligible"] = True

    suspicious_zero = (
        (flagged["vn_2022"].abs() < 1e-12)
        & (
            (flagged["top20_mean"].fillna(0) > 10)
            | (flagged["asean_mean"].fillna(0) > 10)
        )
    )
    flagged.loc[suspicious_zero, "data_quality_flag"] = (
        "vn_value_zero_with_nonzero_benchmarks_possible_missing_or_imputed"
    )
    flagged.loc[suspicious_zero, "headline_eligible"] = False
    return flagged


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
    leakage_report = correlation_leakage_report(df)

    train = df[df["Year"] <= 2018].copy()
    val = df[(df["Year"] >= 2019) & (df["Year"] <= 2021)].copy()
    test = df[df["Year"] >= 2022].copy()

    medians = train[FEATURES].median()
    for split in (train, val, test):
        split[FEATURES] = split[FEATURES].fillna(medians)

    model, selected_params, tuning_rows = tune_xgboost(train, val)
    pd.DataFrame(tuning_rows).sort_values("validation_rmse").to_csv(
        OUTPUT_DIR / "xgboost_tuning_results.csv",
        index=False,
    )

    metrics = {}
    for name, split in [("validation", val), ("test", test)]:
        pred = model.predict(split[FEATURES])
        metrics[name] = {
            "r2": float(r2_score(split[TARGET], pred)),
            "rmse": rmse(split[TARGET], pred),
            "mae": float(np.mean(np.abs(split[TARGET] - pred))),
        }

    shap_train, base_train, xai_method = tree_contribs(model, train[FEATURES])
    shap_test, _, _ = tree_contribs(model, test[FEATURES])
    base_value = float(np.mean(base_train))

    vn_df = df[df["Country"] == "Vietnam"].sort_values("Year").copy()
    vn_df[FEATURES] = vn_df[FEATURES].fillna(medians)
    x_vn = vn_df[FEATURES]
    y_vn = vn_df[TARGET].to_numpy()
    years_vn = vn_df["Year"].to_numpy()
    shap_vn, _, _ = tree_contribs(model, x_vn)
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
    gap = add_data_quality_flags(gap)
    gap = gap.sort_values("shap_vn2022")
    gap.to_csv(OUTPUT_DIR / "gap_analysis.csv", index=False)
    policy_priority = gap[(gap["headline_eligible"]) & (gap["shap_vn2022"] < 0)].copy()
    policy_priority.to_csv(OUTPUT_DIR / "policy_priority_indicators.csv", index=False)
    plot_gap(gap)

    plot_waterfall(2022, years_vn, x_vn, y_vn, pred_vn, shap_vn, base_value)
    plot_waterfall(2023, years_vn, x_vn, y_vn, pred_vn, shap_vn, base_value)
    plot_timeline(years_vn, shap_vn, importance)

    increments = [5, 10, 15, 20, 30]
    priority_source = policy_priority if not policy_priority.empty else gap
    priority_features = priority_source.head(5)["feature"].tolist()
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
        "task_framing": "composite_score_reconstruction_not_independent_governance_prediction",
        "interpretation_warning": (
            "The target goal16 is constructed from the SDG16 component indicators used as inputs. "
            "Report the high R2 as reconstruction accuracy and use contributions for diagnostic "
            "prioritization; do not present them as causal effects."
        ),
        "xai_method": xai_method,
        "xai_method_note": (
            "Uses shap.TreeExplainer when the optional shap package is installed; "
            "otherwise falls back to XGBoost pred_contribs, which returns Tree SHAP contributions."
        ),
        "rows": int(len(df)),
        "countries": int(df["Country"].nunique()),
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "base_value": base_value,
        "selected_params": selected_params,
        "tuning_candidates": len(tuning_rows),
        "metrics": metrics,
        "leakage_correlation_report": leakage_report,
        "vietnam_latest_year": int(years_vn[-1]),
        "vietnam_latest_actual": float(y_vn[-1]),
        "vietnam_latest_predicted": float(pred_vn[-1]),
        "data_quality_warning": (
            "Zero-valued Vietnam indicators with nonzero benchmarks are excluded from headline "
            "policy priorities and retained only as flagged diagnostic rows."
        ),
        "top_negative_vn2022": gap.head(5)[["feature", "label", "shap_vn2022"]].to_dict("records"),
        "headline_policy_priorities_vn2022": policy_priority.head(5)[
            ["feature", "label", "vn_2022", "shap_vn2022", "data_quality_flag"]
        ].to_dict("records"),
        "flagged_zero_value_indicators_vn2022": gap[~gap["headline_eligible"]][
            ["feature", "label", "vn_2022", "top20_mean", "asean_mean", "shap_vn2022", "data_quality_flag"]
        ].to_dict("records"),
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
