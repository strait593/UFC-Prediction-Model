"""FastAPI application for UFC bout predictions."""
from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from ufc_pred_model.model import FEATURES, NUMERIC_FEATURES, TARGET, build_model


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "UFC-dataset" / "Large-set" / "large_dataset.csv"
WEB_PATH = Path(__file__).resolve().parent / "web"

WEIGHT_CLASSES = (
    "Strawweight",
    "Flyweight",
    "Bantamweight",
    "Featherweight",
    "Lightweight",
    "Welterweight",
    "Middleweight",
    "Light Heavyweight",
    "Heavyweight",
)

PROFILE_STATS = (
    "wins_total",
    "losses_total",
    "age",
    "height",
    "weight",
    "reach",
    "SLpM_total",
    "SApM_total",
    "sig_str_acc_total",
    "td_acc_total",
    "str_def_total",
    "td_def_total",
    "sub_avg",
    "td_avg",
)
DIFF_TO_STAT = {f"{stat}_diff": stat for stat in PROFILE_STATS}


class PredictionRequest(BaseModel):
    red_fighter: str = Field(min_length=1, max_length=100)
    blue_fighter: str = Field(min_length=1, max_length=100)
    weight_class: str | None = Field(default=None, max_length=80)

def normalize_weight_class(value: Any) -> str | None:
    """Extract a model-supported division from title and tournament names."""
    if not isinstance(value, str):
        return None
    normalized = value.casefold()
    is_women = "women" in normalized
    for division in sorted(WEIGHT_CLASSES, key=len, reverse=True):
        if division.casefold() in normalized:
            return f"Women's {division}" if is_women else division
    return None

def _profile_from_row(row: pd.Series, corner: str) -> dict[str, Any]:
    prefix = "r" if corner == "Red" else "b"
    profile: dict[str, Any] = {
        "name": row[f"{prefix}_fighter"],
        "gender": row["gender"],
        "_gender_counts": {},
        "stance": row.get(f"{prefix}_stance"),
        "latest_weight_class": normalize_weight_class(row["weight_class"]),
        "divisions": [],
        "division_counts": {},
    }
    for stat in PROFILE_STATS:
        profile[stat] = row.get(f"{prefix}_{stat}")
    return profile

class PredictionService:
    def __init__(
        self,
        model: Any,
        profiles: dict[str, dict[str, Any]],
        metrics: dict[str, float],
    ) -> None:
        self.model = model
        self.profiles = profiles
        self.metrics = metrics
        self.names = sorted(profile["name"] for profile in profiles.values())
        self.divisions_by_gender = {
            gender: sorted(
                {
                    division
                    for profile in profiles.values()
                    if profile["gender"] == gender
                    for division in profile["division_counts"]
                },
                key=lambda value: (value.startswith("Women's"), value),
            )
            for gender in ("Men", "Women")
        }

    @classmethod
    def load(cls, dataset_path: Path = DATASET_PATH) -> PredictionService:
        if not dataset_path.is_file():
            raise FileNotFoundError(f"UFC bout dataset was not found: {dataset_path}")
        data = pd.read_csv(dataset_path)
        required = set(FEATURES + [TARGET, "r_fighter", "b_fighter"])
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(
                f"{dataset_path} is missing required columns: {', '.join(missing)}"
            )
        data = data[data[TARGET].isin(["Red", "Blue"])].copy()
        if len(data) < 10:
            raise ValueError(f"{dataset_path} has too few usable Red/Blue bouts")

        holdout_size = int(len(data) * 0.2)
        train, test = data.iloc[holdout_size:], data.iloc[:holdout_size]
        if train[TARGET].nunique() < 2 or test[TARGET].nunique() < 2:
            raise ValueError("Dataset row order does not provide a valid holdout split")
        evaluation_model = build_model()
        evaluation_model.fit(train[FEATURES], train[TARGET])
        predictions = evaluation_model.predict(test[FEATURES])
        red_index = list(evaluation_model.classes_).index("Red")
        red_probabilities = evaluation_model.predict_proba(test[FEATURES])[:, red_index]
        metrics = {
            "accuracy": float(accuracy_score(test[TARGET], predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(test[TARGET], predictions)
            ),
            "roc_auc_red": float(
                roc_auc_score((test[TARGET] == "Red").astype(int), red_probabilities)
            ),
        }

        model = build_model()
        model.fit(data[FEATURES], data[TARGET])
        profiles: dict[str, dict[str, Any]] = {}
        for _, row in data.iterrows():
            for corner in ("Red", "Blue"):
                name = str(row[f"{'r' if corner == 'Red' else 'b'}_fighter"]).strip()
                if not name:
                    continue
                key = name.casefold()
                division = normalize_weight_class(row["weight_class"])
                if key not in profiles:
                    profiles[key] = _profile_from_row(row, corner)
                profile = profiles[key]
                gender = row["gender"]
                if gender in ("Men", "Women"):
                    profile["_gender_counts"][gender] = (
                        profile["_gender_counts"].get(gender, 0) + 1
                    )
                if division:
                    profile["division_counts"][division] = (
                        profile["division_counts"].get(division, 0) + 1
                    )
                    if division not in profile["divisions"]:
                        profile["divisions"].append(division)
        for profile in profiles.values():
            gender_counts = profile.pop("_gender_counts")
            if gender_counts:
                profile["gender"] = max(gender_counts, key=gender_counts.get)
        return cls(model, profiles, metrics)

    def find_fighter(self, query: str) -> dict[str, Any] | None:
        return self.profiles.get(query.strip().casefold())

    def search_fighters(self, query: str, limit: int = 10) -> list[str]:
        search = query.strip().casefold()
        if not search:
            return []
        starts_with = [
            name for name in self.names if name.casefold().startswith(search)
        ]
        contains = [
            name
            for name in self.names
            if search in name.casefold() and name not in starts_with
        ]
        return (starts_with + contains)[:limit]

    def predict(
        self, red_name: str, blue_name: str, requested_division: str | None
    ) -> dict[str, Any]:
        red = self.find_fighter(red_name)
        blue = self.find_fighter(blue_name)
        if red is None:
            raise HTTPException(
                status_code=404,
                detail=f"Fighter not found: {red_name}. Use fighter search to find a dataset match.",
            )
        if blue is None:
            raise HTTPException(
                status_code=404,
                detail=f"Fighter not found: {blue_name}. Use fighter search to find a dataset match.",
            )
        if red_name.strip().casefold() == blue_name.strip().casefold():
            raise HTTPException(
                status_code=422, detail="Choose two different fighters."
            )
        if red["gender"] != blue["gender"]:
            raise HTTPException(
                status_code=422,
                detail="The selected fighters have different recorded divisions by gender.",
            )

        common_divisions = set(red["division_counts"]) & set(blue["division_counts"])
        if requested_division:
            division = normalize_weight_class(requested_division)
            valid_divisions = self.divisions_by_gender.get(red["gender"], [])
            if division not in valid_divisions:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unsupported weight class. Choose one of: {', '.join(valid_divisions)}.",
                )
        elif common_divisions:
            division = max(
                common_divisions,
                key=lambda name: (
                    red["division_counts"][name] + blue["division_counts"][name],
                    -WEIGHT_CLASSES.index(name.removeprefix("Women's ")),
                ),
            )
        else:
            raise HTTPException(
                status_code=422,
                detail="No shared weight division found. Specify a weight_class for this matchup.",
            )

        features: dict[str, Any] = {
            feature: None for feature in NUMERIC_FEATURES
        }
        for feature, stat in DIFF_TO_STAT.items():
            red_value, blue_value = red[stat], blue[stat]
            if pd.notna(red_value) and pd.notna(blue_value):
                features[feature] = float(red_value) - float(blue_value)
        features.update(
            {
                "weight_class": division,
                "gender": red["gender"],
                "r_stance": red["stance"],
                "b_stance": blue["stance"],
            }
        )
        red_index = list(self.model.classes_).index("Red")
        probabilities = self.model.predict_proba(pd.DataFrame([features]))[0]
        probability_red = float(probabilities[red_index])
        probability_blue = 1.0 - probability_red
        predicted_corner = "Red" if probability_red >= probability_blue else "Blue"

        return {
            "red_fighter": self._fighter_summary(red),
            "blue_fighter": self._fighter_summary(blue),
            "weight_class": division,
            "predicted_winner": red["name"] if predicted_corner == "Red" else blue["name"],
            "predicted_corner": predicted_corner,
            "probability_red": probability_red,
            "probability_blue": probability_blue,
            "odds": {
                "red": self._odds(probability_red),
                "blue": self._odds(probability_blue),
            },
            "probability_gap": abs(probability_red - probability_blue),
            "model_metrics": self.metrics,
            "caveat": (
                "Probabilities and odds are model estimates from historical data, "
                "not calibrated sportsbook odds. Fighter stats use the latest "
                "available bout record in this dataset."
            ),
        }

    @staticmethod
    def _fighter_summary(profile: dict[str, Any]) -> dict[str, Any]:
        def numeric(stat: str) -> float | int | None:
            value = profile.get(stat)
            if pd.isna(value):
                return None
            number = float(value)
            return int(number) if stat in {"wins_total", "losses_total"} else number

        return {
            "name": profile["name"],
            "wins": numeric("wins_total"),
            "losses": numeric("losses_total"),
            "age": numeric("age"),
            "height_cm": numeric("height"),
            "reach_cm": numeric("reach"),
            "weight_kg": numeric("weight"),
            "stance": profile.get("stance"),
            "significant_strikes_per_minute": numeric("SLpM_total"),
            "significant_strike_accuracy": numeric("sig_str_acc_total"),
            "takedowns_per_15_min": numeric("td_avg"),
            "submission_attempts_per_15_min": numeric("sub_avg"),
        }

    @staticmethod
    def _odds(probability: float) -> dict[str, float | int]:
        american = (
            round((1 - probability) / probability * 100)
            if probability < 0.5
            else -round(probability / (1 - probability) * 100)
        )
        return {
            "implied_probability": probability,
            "decimal": round(1 / probability, 2),
            "american": american,
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prediction_service = PredictionService.load()
    yield


app = FastAPI(
    title="UFC Bout Predictor",
    description="Prototype UFC bout predictions based on historical fighter statistics.",
    version="0.1.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=WEB_PATH), name="static")

def get_service(request: Request) -> PredictionService:
    return request.app.state.prediction_service


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_PATH / "index.html")


@app.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    service = get_service(request)
    return {
        "status": "ok",
        "fighters_in_dataset": len(service.profiles),
        "model_metrics": service.metrics,
    }


@app.get("/api/fighters")
def search_fighters(
    request: Request,
    q: Annotated[str, Query(min_length=2, max_length=100)],
) -> dict[str, list[str]]:
    return {"fighters": get_service(request).search_fighters(q)}


@app.get("/api/weight-classes")
def weight_classes(
    request: Request,
    gender: Annotated[str, Query(pattern="^(Men|Women)$")],
) -> dict[str, list[str]]:
    service = get_service(request)
    return {"weight_classes": service.divisions_by_gender[gender]}


@app.post("/api/predict")
def predict_bout(
    payload: PredictionRequest, request: Request
) -> dict[str, Any]:
    return get_service(request).predict(
        payload.red_fighter, payload.blue_fighter, payload.weight_class
    )