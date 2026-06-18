import json

from apps.api.app.schemas import PredictionRequest
from apps.api.app.services.model_service import ModelService


def test_linear_prediction_and_explanation(tmp_path):
    metadata = {
        "intercept": 10,
        "coefficients": {"x": 2},
        "feature_defaults": {"x": 5},
        "feature_means": {"x": 5},
        "feature_stds": {"x": 2},
        "model_version": "test",
        "metrics": {},
    }
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    service = ModelService(path)
    request = PredictionRequest(country="Vietnam", year=2025, features={"x": 7})

    score, missing = service.predict(request)
    explained_score, contributions, explained_missing = service.explain(request)

    assert score == 12
    assert explained_score == score
    assert missing == explained_missing == []
    assert contributions[0].contribution == 2
