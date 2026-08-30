"""
api/routes/triage.py
=======================

PatientTriage.ai — Triage Recommendation Endpoint
--------------------------------------------------------
POST /api/v1/triage

Full workflow, matching the platform's stated intake -> queue path:

    Patient input -> validation -> rule engine -> ensemble -> confidence
    -> explanation -> PERSIST as a new triage_stays row -> audit log
    -> WebSocket broadcast ("new_patient") -> return recommendation

Persistence is what makes "the patient enters the Live Queue" literally
true: earlier, this endpoint only wrote to the audit log, so a freshly
-submitted patient never actually showed up anywhere `GET /queue` could
see. See `app/services/patient_registry.py` for why and how that gap
was closed.

Governing rule: "The AI recommends. The nurse decides." This endpoint
never writes to a patient's `acuity` (ground truth) field and never
reorders anything — it creates a new record and returns a
recommendation alongside it, both fully reviewable and overridable.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.logging_config import get_logger
from app.models.audit_log import AuditLog
from app.schemas.triage import TriageRequest, TriageResponse
from app.services.cps import compute_cps, format_patient_id
from app.services.patient_registry import create_intake_stay
from app.websocket.connection_manager import manager
from ml.predict import predict
from ml.sepsis import assess_sepsis
from ml.protocol_triggers import detect_protocols, protocols_to_dict

router = APIRouter(prefix="/triage", tags=["triage"])
logger = get_logger(__name__)


def _persist_recommendation(db: Session, patient: dict, result: dict, stay_id: int) -> str | None:
    """Audit-log write is best-effort — a storage failure must never hide a recommendation from the nurse."""
    try:
        entry = AuditLog(
            event_type="triage_recommendation",
            actor="ml_pipeline",
            resource_type="triage_stay",
            resource_id=str(stay_id),
            details=json.dumps({"input": patient, "recommendation": result}, default=str),
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        return entry.id
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to persist triage recommendation to audit log: %s", exc)
        db.rollback()
        return None


@router.post("", response_model=TriageResponse)
async def create_triage_recommendation(
    body: TriageRequest,
) -> TriageResponse:
    """
    Score a new patient, persist them into the Live Queue, and return
    the recommendation. The rule engine always runs first inside
    `predict()` — a red flag sets a priority floor the ensemble cannot
    soften.
    """
    patient = body.to_patient_dict()
    # predict() is CPU-bound (ML inference). Run it in a thread-pool executor
    # so it doesn't block the async event loop and starve other requests.
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, predict, patient)

    # --- Sepsis screening (fast, pure-Python — runs synchronously) ---
    sepsis = assess_sepsis(patient)

    # --- Protocol triggers (rule-based, no ML) ---
    protocols = detect_protocols(patient)
    protocols_data = protocols_to_dict(protocols)

    with SessionLocal() as db:
        stay = create_intake_stay(db, patient)
        # Write the ML recommendation onto the stay so the waiting-room monitor
        # can read priority without re-running predict() on every cycle.
        stay.recommended_priority = result["priority"]
        stay.recommended_confidence = result["confidence"]
        db.commit()
        db.refresh(stay)

        patient_id = format_patient_id(stay.stay_id)
        stay_id = stay.stay_id
        cps_info = compute_cps(stay, result, age=body.age)

        prediction_id = _persist_recommendation(db, patient, result, stay.stay_id)

    await manager.broadcast({
        "event": "new_patient",
        "patient_id": patient_id,
        "stay_id": stay_id,
        "priority": result["priority"],
        "clinical_priority_score": cps_info["cps_100"],
        "escalated": result["escalated"],
        "chief_complaint": patient.get("chief_complaint"),
        "sepsis_alert": sepsis.alert,
        "triggered_protocols": [p["code"] for p in protocols_data],
    })

    return TriageResponse(
        patient_id=patient_id,
        priority=result["priority"],
        risk_score=result["risk_score"],
        clinical_priority_score=cps_info["cps_100"],
        confidence=result["confidence"],
        uncertainty_reason=result["uncertainty_reason"],
        top_features=result["top_features"],
        escalated=result["escalated"],
        prediction_id=prediction_id,
        # Sepsis
        sepsis_alert=sepsis.alert,
        sepsis_risk_level=sepsis.risk_level,
        sepsis_qsofa=sepsis.qsofa_score,
        sepsis_criteria=sepsis.qsofa_criteria + sepsis.sirs_criteria,
        sepsis_message=sepsis.message,
        sepsis_requires_acknowledgement=sepsis.requires_acknowledgement,
        # Protocols
        triggered_protocols=protocols_data,
    )
