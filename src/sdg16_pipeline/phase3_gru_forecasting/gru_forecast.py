"""GRU forecasting/evaluation for SDG16 panel data."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - optional local experiment.
    raise SystemExit(
        "PyTorch is required for GRU. Install torch or run only scripts/evaluate_models.py "
        "with existing artifacts."
    ) from exc


DATA_PATH = Path("data/clean/sdg16_spark.csv")
OUTPUT_DIR = Path("artifacts/gru")
SOURCE = "SDR2024"
TARGET = "goal16"
SEQUENCE_LENGTH = 5

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


class GRURegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, max(8, hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(8, hidden_size // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_panel_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[df["source"] == SOURCE].copy()
    df = df.dropna(subset=[TARGET])
    df = df.sort_values(["Country", "Year"])
    return df


def build_sequences(df: pd.DataFrame, medians: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    years: list[int] = []
    countries: list[str] = []

    for country, group in df.groupby("Country"):
        group = group.sort_values("Year").copy()
        group[FEATURES] = group[FEATURES].fillna(medians)
        values = group[FEATURES].to_numpy(dtype=np.float32)
        target = group[TARGET].to_numpy(dtype=np.float32)
        year_values = group["Year"].to_numpy(dtype=np.int32)
        for index in range(SEQUENCE_LENGTH, len(group)):
            x_rows.append(values[index - SEQUENCE_LENGTH : index])
            y_rows.append(float(target[index]))
            years.append(int(year_values[index]))
            countries.append(str(country))

    return (
        np.stack(x_rows).astype(np.float32),
        np.asarray(y_rows, dtype=np.float32),
        np.asarray(years, dtype=np.int32),
        countries,
    )


def split_by_year(
    x: np.ndarray,
    y: np.ndarray,
    years: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "train": (x[years <= 2018], y[years <= 2018]),
        "validation": (x[(years >= 2019) & (years <= 2021)], y[(years >= 2019) & (years <= 2021)]),
        "test": (x[years >= 2022], y[years >= 2022]),
    }


def scale_sequences(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], StandardScaler, StandardScaler]:
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    x_train, y_train = splits["train"]
    feature_scaler.fit(x_train.reshape(-1, x_train.shape[-1]))
    target_scaler.fit(y_train.reshape(-1, 1))

    scaled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (x_split, y_split) in splits.items():
        x_scaled = feature_scaler.transform(x_split.reshape(-1, x_split.shape[-1])).reshape(x_split.shape)
        y_scaled = target_scaler.transform(y_split.reshape(-1, 1)).ravel()
        scaled[name] = (x_scaled.astype(np.float32), y_scaled.astype(np.float32))
    return scaled, feature_scaler, target_scaler


def tensors(x: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(x), torch.from_numpy(y)


def train_candidate(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    epochs: int = 160,
) -> tuple[GRURegressor, float]:
    model = GRURegressor(
        input_size=len(FEATURES),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    x_train, y_train = tensors(*splits["train"])
    x_val, y_val = tensors(*splits["validation"])

    best_state = None
    best_val = float("inf")
    patience = 18
    stale = 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def predict_inverse(model: GRURegressor, x: np.ndarray, target_scaler: StandardScaler) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(x)).numpy()
    return target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()


def metrics_for(model: GRURegressor, splits: dict[str, tuple[np.ndarray, np.ndarray]], raw_splits: dict[str, tuple[np.ndarray, np.ndarray]], target_scaler: StandardScaler) -> dict:
    metrics = {}
    for name in ["validation", "test"]:
        x_scaled, _ = splits[name]
        _, y_raw = raw_splits[name]
        pred = predict_inverse(model, x_scaled, target_scaler)
        metrics[name] = {
            "r2": float(r2_score(y_raw, pred)),
            "rmse": rmse(y_raw, pred),
            "mae": float(mean_absolute_error(y_raw, pred)),
        }
    return metrics


def vietnam_forecast(
    df: pd.DataFrame,
    model: GRURegressor,
    medians: pd.Series,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    years: range = range(2024, 2031),
) -> list[dict]:
    vn = df[df["Country"] == "Vietnam"].sort_values("Year").copy()
    vn[FEATURES] = vn[FEATURES].fillna(medians)
    history = vn[FEATURES].tail(SEQUENCE_LENGTH).to_numpy(dtype=np.float32)
    trend_window = vn.tail(max(SEQUENCE_LENGTH, 6)).copy()
    year_values = trend_window["Year"].to_numpy(dtype=float)
    feature_values = trend_window[FEATURES].to_numpy(dtype=float)

    slopes = np.zeros(len(FEATURES), dtype=np.float32)
    if len(trend_window) >= 3 and np.ptp(year_values) > 0:
        x_year = year_values - year_values.mean()
        denom = float(np.sum(x_year**2))
        if denom:
            slopes = ((x_year[:, None] * (feature_values - feature_values.mean(axis=0))).sum(axis=0) / denom).astype(np.float32)

    # Keep simulated future indicators within observed global ranges. This
    # prevents runaway synthetic values while still letting the sequence evolve,
    # fixing the previous flat forecast caused by reusing the last vector.
    feature_min = df[FEATURES].quantile(0.01).fillna(medians).to_numpy(dtype=np.float32)
    feature_max = df[FEATURES].quantile(0.99).fillna(medians).to_numpy(dtype=np.float32)

    current_features = history[-1].copy()
    rows = []
    for step, year in enumerate(years, start=1):
        x_scaled = feature_scaler.transform(history.reshape(-1, history.shape[-1])).reshape(1, SEQUENCE_LENGTH, len(FEATURES)).astype(np.float32)
        pred = float(predict_inverse(model, x_scaled, target_scaler)[0])
        damping = float(0.85 ** (step - 1))
        trend_norm = float(np.linalg.norm(slopes * damping))
        rows.append(
            {
                "year": year,
                "predicted_goal16": round(pred, 4),
                "feature_trend_norm": round(trend_norm, 6),
            }
        )
        current_features = np.clip(current_features + slopes * damping, feature_min, feature_max)
        history = np.vstack([history[1:], current_features])
    return rows


def main() -> int:
    if hasattr(__import__("sys").stdout, "reconfigure"):
        __import__("sys").stdout.reconfigure(encoding="utf-8", errors="replace")
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_panel_data()
    medians = df[df["Year"] <= 2018][FEATURES].median()
    x, y, years, countries = build_sequences(df, medians)
    raw_splits = split_by_year(x, y, years)
    splits, feature_scaler, target_scaler = scale_sequences(raw_splits)

    candidates = [
        {"hidden_size": 32, "num_layers": 1, "dropout": 0.0, "learning_rate": 0.01},
        {"hidden_size": 48, "num_layers": 1, "dropout": 0.0, "learning_rate": 0.006},
        {"hidden_size": 64, "num_layers": 2, "dropout": 0.15, "learning_rate": 0.005},
    ]
    tuning_rows = []
    best_model = None
    best_params = None
    best_val = float("inf")
    for index, params in enumerate(candidates, start=1):
        model, val_loss = train_candidate(splits, **params)
        candidate_metrics = metrics_for(model, splits, raw_splits, target_scaler)
        row = {
            "candidate": index,
            **params,
            "validation_loss_scaled": val_loss,
            "validation_rmse": candidate_metrics["validation"]["rmse"],
            "validation_r2": candidate_metrics["validation"]["r2"],
        }
        tuning_rows.append(row)
        if candidate_metrics["validation"]["rmse"] < best_val:
            best_val = candidate_metrics["validation"]["rmse"]
            best_model = model
            best_params = params

    assert best_model is not None and best_params is not None
    metrics = metrics_for(best_model, splits, raw_splits, target_scaler)
    forecast_rows = vietnam_forecast(df, best_model, medians, feature_scaler, target_scaler)

    pd.DataFrame(tuning_rows).sort_values("validation_rmse").to_csv(
        OUTPUT_DIR / "gru_tuning_results.csv",
        index=False,
    )
    pd.DataFrame(forecast_rows).to_csv(OUTPUT_DIR / "vietnam_forecast_2024_2030.csv", index=False)
    torch.save(best_model.state_dict(), OUTPUT_DIR / "gru_model_state.pt")

    summary = {
        "model_type": "pytorch_gru_regressor",
        "source": SOURCE,
        "sequence_length": SEQUENCE_LENGTH,
        "features": FEATURES,
        "forecast_method": "recursive_gru_with_damped_vietnam_feature_trends",
        "rows": int(len(df)),
        "sequences": int(len(x)),
        "countries": int(len(set(countries))),
        "selected_params": best_params,
        "tuning_candidates": len(candidates),
        "metrics": metrics,
        "vietnam_forecast_2024_2030": forecast_rows,
    }
    (OUTPUT_DIR / "gru_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"GRU outputs written to {OUTPUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
