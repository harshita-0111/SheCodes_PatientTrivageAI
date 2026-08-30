"""
ml/predict.py
================

PatientTriage.ai — Prediction Interface
------------------------------------------
The single public entry point for scoring a patient:

    predict(patient_dict) -> dict

Pipeline (matches the platform's Hybrid Intelligence Layer):

    Patient data -> Rule Engine -> Ensemble ML -> Uncertainty -> SHAP -> Recommendation

Governing rule: "The AI recommends. The nurse decides." This function
returns a recommendation dict. It does not write to the queue, does not
move a patient, and every field it returns is meant to be shown to and
overridable by a clinician — see `app/api/routes/triage.py` for how the
result is persisted and logged.

Rule-engine-first, always
------------------------------
`rule_engine.evaluate()` runs before the ensemble on every call. If it
fires, the returned priority is CLAMPED to the rule's floor — the
ensemble still runs and its output is still surfaced (so an
explanation is always available), but it can never soften a red flag.
See the module docstring in `rule_engine.py` for the full rationale.

P4/P5 handling
------------------
The ensemble is trained only on P1-P3 (see `train_model.py` for why).
A patient the ensemble does not consider high-risk, with no red flag
and no abnormal vitals, is assigned P4 by an explicit, documented rule
— never by an ML boundary the training data cannot support. This
function never returns P5 from the ML path; P5 is reserved for future
data that actually contains P5 examples.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.core.logging_config import get_logger
from ml import rule_engine
from ml.model_utils import ModelArtifact, build_features, features_to_row, load_artifact
from ml.uncertainty import estimate_confidence

logger = get_logger(__name__)

PRIORITY_LABELS = {1: "P1", 2: "P2", 3: "P3", 4: "P4", 5: "P5"}

# Human-readable names for the top SHAP/ensemble-driven features, used to
# turn "o2sat" into "Low SpO2" the way the platform's output contract
# expects. Direction (low/high) is resolved against the patient's own
# value at call time, not hard-coded here.
FEATURE_DISPLAY_NAMES = {
    "o2sat": "Oxygen Saturation",
    "heartrate": "Heart Rate",
    "resprate": "Respiratory Rate",
    "sbp": "Systolic Blood Pressure",
    "dbp": "Diastolic Blood Pressure",
    "temperature_c": "Temperature",
    "pain": "Pain Score",
    "shock_index": "Shock Index",
    "pulse_pressure": "Pulse Pressure",
    "mean_arterial_pressure": "Mean Arterial Pressure",
    "abnormal_vitals_count": "Number of Abnormal Vitals",
    "vitals_missing_count": "Missing Vitals",
    "age": "Age",
}


@lru_cache
def _get_artifact() -> ModelArtifact | None:
    """Cached load — the trained ensemble does not change within a process lifetime."""
    return load_artifact()


def reset_artifact_cache() -> None:
    """Call after retraining so a running API process picks up the new model without a restart."""
    _get_artifact.cache_clear()
    _predict_cache.clear()



_VITAL_SOURCE_KEYS = {
    "o2sat": ("o2sat", "o2_sat", "spo2"),
    "heartrate": ("heartrate", "heart_rate"),
    "resprate": ("resprate", "resp_rate"),
    "sbp": ("sbp",),
    "dbp": ("dbp",),
    "temperature_c": ("temperature",),
    "pain": ("pain",),
}


def _was_recorded(name: str, patient: dict) -> bool:
    """True if the source field behind this feature was actually provided,
    distinct from the feature's numeric value being 0.0 by zero-fill default."""
    keys = _VITAL_SOURCE_KEYS.get(name)
    if keys is None:
        return True  # non-vital features (engineered composites, flags) aren't "recorded" in this sense
    return any(patient.get(k) is not None for k in keys)


def _describe_feature(name: str, value: float, patient: dict | None = None) -> str:
    label = FEATURE_DISPLAY_NAMES.get(name, name.replace("_", " ").title())
    if patient is not None and name in _VITAL_SOURCE_KEYS and not _was_recorded(name, patient):
        # A zero-filled missing vital must never be described as a measured
        # extreme — "Low SpO2" implies a reading was taken. It wasn't.
        return f"{label} not recorded"
    low_is_bad = {"o2sat", "sbp", "dbp", "mean_arterial_pressure"}
    high_is_bad = {"heartrate", "resprate", "temperature_c", "pain", "shock_index",
                  "abnormal_vitals_count", "vitals_missing_count"}
    if name in low_is_bad:
        return f"Low {label}"
    if name in high_is_bad:
        return f"High {label}"
    return label


def _top_features(artifact: ModelArtifact, row, features: dict, predicted_class_index: int, patient: dict, n: int = 3) -> list[str]:
    """
    Best-effort SHAP-driven top features for THIS prediction, for THIS
    predicted class. Falls back to the largest-magnitude non-zero
    features if SHAP is unavailable or fails, so `predict()` never
    blocks on explainability — `explain.py` is the module responsible
    for the full, saved SHAP report; this is a fast, request-time
    approximation for the response payload.

    `predicted_class_index` must be the REMAPPED 0..n-1 index (matching
    `artifact.base_models["hist_gradient_boosting"]`'s own label space,
    not the true priority value) — see `train_model.py` for why the
    remap exists. Passing the true priority value here would silently
    explain the wrong class.
    """
    try:
        import shap

        explainer = shap.TreeExplainer(artifact.base_models["hist_gradient_boosting"])
        values = np.asarray(explainer.shap_values(row))
        if values.ndim == 3:
            arr = values[0, :, predicted_class_index]
        else:
            arr = values[0]
        ranked = sorted(zip(artifact.feature_names, arr.tolist()), key=lambda t: -abs(t[1]))
    except Exception as exc:  # noqa: BLE001 — explanation must never break a prediction
        logger.debug("Request-time SHAP unavailable (%s) — falling back to feature magnitude.", exc)
        ranked = sorted(
            ((k, v) for k, v in features.items() if v != 0.0),
            key=lambda t: -abs(t[1]),
        )

    out, seen = [], set()
    for name, _ in ranked:
        if name.startswith("arrival_transport_") or name.startswith("gender_") or name.startswith("age_group_"):
            continue
        label = _describe_feature(name, features.get(name, 0.0), patient)
        if label in seen:
            continue
        seen.add(label)
        out.append(label)
        if len(out) >= n:
            break
    return out


def _risk_score(probabilities: dict[int, float], predicted_class: int, classes: list[int]) -> int:
    """
    0-100 risk score. Anchored to acuity rather than a raw probability:
    a confident P1 call should score higher than an equally confident P3
    call, since risk_score is meant to rank urgency, not just certainty.
    """
    urgency = (max(classes) - predicted_class) / max(max(classes) - min(classes), 1)
    confidence_component = probabilities.get(predicted_class, 0.0)
    score = 100 * (0.7 * urgency + 0.3 * confidence_component)
    return int(round(min(100, max(0, score))))


_predict_cache: dict[tuple, dict] = {}


def predict(patient: dict) -> dict:
    # Convert patient dict to a hashable tuple to cache the prediction
    def _make_hashable(val):
        if isinstance(val, dict):
            return tuple(sorted((k, _make_hashable(v)) for k, v in val.items()))
        if isinstance(val, list):
            return tuple(_make_hashable(v) for v in val)
        return val

    try:
        cache_key = tuple(sorted((k, _make_hashable(v)) for k, v in patient.items()))
        if cache_key in _predict_cache:
            return _predict_cache[cache_key]
    except Exception:
        cache_key = None

    """
    Score one patient end to end: rule engine, ensemble, uncertainty.

    Parameters
    ----------
    patient : dict
        Accepts: age, gender, heartrate, sbp, dbp, resprate, temperature,
        o2sat, pain, chief_complaint, arrival_transport, arrival_hour,
        night_shift_flag, weekend_flag, medications, history, zero_history,
        and any rule-engine finding flags (chest_pain, diaphoresis,
        fast_positive, unresponsive, seizing, airway_compromise, stridor).
        Every field is optional; missing fields increase uncertainty
        rather than being assumed normal (see `ml/uncertainty.py`).

    Returns
    -------
    dict
        {
          "priority": "P2",
          "risk_score": 84,
          "confidence": 0.91,
          "uncertainty_reason": "Missing medication history",
          "top_features": ["Low SpO2", "High Respiratory Rate", "Low Blood Pressure"],
          "escalated": true
        }
    """
    rule_result = rule_engine.evaluate(patient)

    artifact = _get_artifact()
    features = build_features(patient)

    if artifact is None:
        # No trained model — the rule engine alone still produces a safe,
        # explainable recommendation rather than a 503. This is the
        # platform's fail-safe-default principle applied to the ML layer
        # itself: absence of a model must escalate caution, not block care.
        logger.warning("No trained ensemble available — falling back to rule-engine-only priority.")
        priority = rule_result.priority_floor if rule_result.escalated else 4
        return {
            "priority": PRIORITY_LABELS[priority],
            "risk_score": 80 if rule_result.escalated else 30,
            "confidence": 0.4 if rule_result.escalated else 0.3,
            "uncertainty_reason": "No trained ML model available — rule-engine-only assessment",
            "top_features": [h.reason for h in rule_result.hits[:3]] or ["No trained model loaded"],
            "escalated": rule_result.escalated,
        }

    row = features_to_row(features)
    proba = artifact.meta_learner.predict_proba(row)[0]
    class_probabilities = {c: float(p) for c, p in zip(artifact.classes, proba)}
    predicted_class_index = int(np.argmax(proba))
    ml_predicted_class = int(artifact.classes[predicted_class_index])

    # Data-sufficiency guard — the ensemble uses vitals_missing_count as a
    # feature, so a patient submitted with ONLY demographics and no vitals
    # produces a spuriously high missing_count that the model interprets as
    # risk, inflating priority to P1/P2 with no clinical basis. If 6 or
    # more of the 7 vital fields are absent AND no rule-engine red flag
    # fired, we cannot responsibly issue a high-acuity recommendation.
    # Enforce P3 floor and flag low confidence so the nurse knows more
    # data is required before a reliable assessment can be made.
    VITAL_FIELDS = ("o2sat", "heartrate", "resprate", "sbp", "dbp", "temperature_c", "pain")
    n_missing = sum(1 for f in VITAL_FIELDS if features.get(f, 0.0) == 0.0
                    and not any(patient.get(k) is not None for k in _VITAL_SOURCE_KEYS.get(f, ())))
    insufficient_data = (not rule_result.escalated) and n_missing >= 6

    # No red flag, ensemble favors its least-urgent trained class, and
    # vitals are unremarkable: documented rule-based P4 floor — see
    # module docstring. Never an ML-invented P4/P5 boundary.
    if insufficient_data:
        # Cannot reliably assess acuity with almost no vitals. Assign P3
        # as a cautious default (not P4/P5 — unknown is not "safe") so the
        # patient still gets clinical attention while more data is gathered.
        final_priority = 3
    elif not rule_result.escalated and ml_predicted_class == max(artifact.classes) and features["abnormal_vitals_count"] == 0:
        final_priority = 4
    else:
        # THE CLAMP: the rule engine's floor always wins over the ensemble's
        # class if the rule floor is more urgent (lower number).
        final_priority = min(ml_predicted_class, rule_result.priority_floor)

    confidence_result = estimate_confidence(
        artifact=artifact, patient=patient, features=features,
        predicted_class=ml_predicted_class, class_probabilities=class_probabilities,
    )

    top_features = _top_features(artifact, row, features, predicted_class_index, patient)
    if rule_result.escalated:
        # Lead with the clinical reason that actually drove escalation —
        # a nurse should see "chest pain with diaphoresis" before "high
        # shock index", since the former is why this patient is a P1.
        rule_reasons = [h.reason for h in rule_result.hits[:2]]
        top_features = (rule_reasons + top_features)[:3]

    risk_score = _risk_score(class_probabilities, ml_predicted_class, artifact.classes)
    if rule_result.escalated:
        risk_score = max(risk_score, 85)  # a fired red flag is never reported as low-risk
    elif insufficient_data:
        risk_score = min(risk_score, 35)  # cap risk score when data is insufficient

    # Override confidence/reason for data-insufficient cases so the nurse
    # sees a clear explanation rather than a misleadingly peaked model score.
    if insufficient_data:
        uncertainty_reason_override = (
            f"{n_missing} of 7 vital signs not recorded — assessment requires more clinical data"
        )
        confidence_override = min(confidence_result.confidence, 0.35)
    else:
        uncertainty_reason_override = confidence_result.uncertainty_reason
        confidence_override = confidence_result.confidence

    result = {
        "priority": PRIORITY_LABELS[final_priority],
        "risk_score": risk_score,
        "confidence": confidence_override,
        "uncertainty_reason": uncertainty_reason_override,
        "top_features": top_features,
        "escalated": rule_result.escalated,
    }

    logger.info(
        "predict() -> priority=%s risk=%d confidence=%.2f escalated=%s (rule_floor=%s ml_class=%s)",
        result["priority"], result["risk_score"], result["confidence"], result["escalated"],
        rule_result.priority_label, PRIORITY_LABELS[ml_predicted_class],
    )
    if cache_key is not None:
        if len(_predict_cache) > 2000:
            _predict_cache.clear()
        _predict_cache[cache_key] = result
    return result
