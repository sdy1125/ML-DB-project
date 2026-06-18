import json
from dataclasses import dataclass
from pathlib import Path

from apps.api.app.schemas import FeatureContribution, PredictionRequest


class ModelNotReadyError(RuntimeError):
    pass


@dataclass
class LinearModelArtifact:
    intercept: float
    coefficients: dict[str, float]
    feature_defaults: dict[str, float]
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    model_version: str
    metrics: dict[str, float]


class ModelService:
    def __init__(self, metadata_path: Path):
        self.metadata_path = metadata_path
        self._artifact: LinearModelArtifact | None = None
        self.reload()

    @property
    def ready(self) -> bool:
        return self._artifact is not None

    def reload(self) -> None:
        if not self.metadata_path.exists():
            self._artifact = None
            return

        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self._artifact = LinearModelArtifact(
            intercept=float(payload["intercept"]),
            coefficients={
                key: float(value) for key, value in payload["coefficients"].items()
            },
            feature_defaults={
                key: float(value)
                for key, value in payload.get(
                    "feature_defaults", payload.get("feature_means", {})
                ).items()
            },
            feature_means={
                key: float(value)
                for key, value in payload.get("feature_means", {}).items()
            },
            feature_stds={
                key: float(value)
                for key, value in payload.get("feature_stds", {}).items()
            },
            model_version=payload.get("model_version", "unknown"),
            metrics=payload.get("metrics", {}),
        )

    def model_info(self) -> dict:
        if not self._artifact:
            return {
                "ready": False,
                "message": "Run the Spark training pipeline to create model artifacts.",
            }
        return {
            "ready": True,
            "version": self._artifact.model_version,
            "features": list(self._artifact.coefficients),
            "metrics": self._artifact.metrics,
        }

    def predict(self, request: PredictionRequest) -> tuple[float, list[str]]:
        artifact = self._require_artifact()
        missing = [
            feature
            for feature in artifact.coefficients
            if feature not in request.features
        ]

        score = artifact.intercept
        for feature, coefficient in artifact.coefficients.items():
            raw_value = request.features.get(
                feature,
                artifact.feature_defaults.get(
                    feature, artifact.feature_means.get(feature, 0.0)
                ),
            )
            score += coefficient * self._standardize(feature, raw_value, artifact)
        return score, missing

    def explain(
        self, request: PredictionRequest
    ) -> tuple[float, list[FeatureContribution], list[str]]:
        artifact = self._require_artifact()
        score, missing = self.predict(request)
        contributions = []

        for feature, coefficient in artifact.coefficients.items():
            raw_value = request.features.get(
                feature,
                artifact.feature_defaults.get(
                    feature, artifact.feature_means.get(feature, 0.0)
                ),
            )
            contribution = coefficient * self._standardize(
                feature, raw_value, artifact
            )
            contributions.append(
                FeatureContribution(
                    feature=feature,
                    value=raw_value,
                    coefficient=coefficient,
                    contribution=contribution,
                    direction="positive" if contribution >= 0 else "negative",
                )
            )

        contributions.sort(key=lambda item: abs(item.contribution), reverse=True)
        return score, contributions, missing

    def _standardize(
        self, feature: str, value: float, artifact: LinearModelArtifact
    ) -> float:
        mean = artifact.feature_means.get(feature, 0.0)
        std = artifact.feature_stds.get(feature, 1.0)
        return (value - mean) / std if std else value - mean

    def _require_artifact(self) -> LinearModelArtifact:
        if not self._artifact:
            raise ModelNotReadyError(
                f"Model metadata not found at {self.metadata_path}."
            )
        return self._artifact

    @property
    def artifact(self) -> LinearModelArtifact:
        return self._require_artifact()
