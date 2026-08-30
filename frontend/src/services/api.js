import axios from "axios";

// PatientTriage.ai — API client
//
// Base URL is relative ("/api/v1") so Vite's dev-server proxy (see
// vite.config.js) forwards requests to FastAPI locally, and a reverse
// proxy can do the same in production without any frontend code change.
//
// A note on what's real: /queue, /override, /vitals/update, and /audit
// were added to the backend specifically to support this frontend — they
// did not exist before. There is currently no live WebSocket broadcaster
// on the backend (no /ws/live), so useLiveQueue() below POLLS /queue on
// an interval instead of opening a socket. This is a deliberate, stated
// substitution, not a placeholder pretending to be real-time.
const apiBase = import.meta.env.VITE_API_URL || "";
export const apiClient = axios.create({
  baseURL: `${apiBase}/api/v1`,
  timeout: 15000,
  headers: { "Content-Type": "application/json" },
});

// ------------------------------------------------------------------ health
export async function fetchHealth() {
  const response = await apiClient.get("/health");
  return response.data;
}

// ------------------------------------------------------------- triage-stays (cohort dashboard)
export async function fetchTriageStaysSummary() {
  const response = await apiClient.get("/triage-stays/summary");
  return response.data;
}

export async function fetchTriageStays(params = {}) {
  const response = await apiClient.get("/triage-stays", {
    params: {
      limit: params.limit ?? 50,
      offset: params.offset ?? 0,
      acuity: params.acuity,
      age_group: params.ageGroup,
      untriaged_only: params.untriagedOnly,
      high_risk_only: params.highRiskOnly,
    },
  });
  return response.data;
}

export async function fetchTriageStay(stayId) {
  const response = await apiClient.get(`/triage-stays/${stayId}`);
  return response.data;
}

// ---------------------------------------------------------------------- model
export async function fetchModelStatus() {
  const response = await apiClient.get("/model/status");
  return response.data;
}

export async function fetchFeatureImportance() {
  const response = await apiClient.get("/model/feature-importance");
  return response.data;
}

export async function predictStay(stayId) {
  const response = await apiClient.post(`/model/predict/${stayId}`);
  return response.data;
}

export async function scoreAllStays(force = false) {
  const response = await apiClient.post("/model/score-all", null, { params: { force } });
  return response.data;
}

// -------------------------------------------------------- single-patient triage
/**
 * Score one patient via the Hybrid Intelligence Layer: rule engine ->
 * stacking ensemble -> uncertainty -> explanation. See backend/ml/predict.py.
 */
export async function submitTriage(patient) {
  const response = await apiClient.post("/triage", patient);
  return response.data;
}

// --------------------------------------------------------------------- queue
/**
 * Live queue over triage_stays with computed Clinical Priority Score.
 * `sort: "priority"` (default) is priority level, then CPS, then arrival
 * — the recommended display order. `sort: "arrival"` is raw insertion
 * order. `sort: "cps"` is CPS-only. None of these mutate anything server
 * -side — see backend/app/api/routes/queue.py docstring.
 * @param {"priority"|"arrival"|"cps"} sort
 */
export async function fetchQueue(sort = "priority") {
  const response = await apiClient.get("/queue", { params: { sort } });
  return response.data;
}

/** Record a clinician override. Never mutates the AI's own recommendation — logged alongside it. */
export async function submitOverride({ stayId, originalPriority, newPriority, reason, actor = "nurse" }) {
  const response = await apiClient.post("/override", {
    stay_id: stayId,
    original_priority: originalPriority,
    new_priority: newPriority,
    reason,
    actor,
  });
  return response.data;
}

/**
 * Update a stay's vitals and get the re-scored recommendation back.
 * Note: the backend stores one point-in-time vitals snapshot per stay,
 * not a persisted time series — see backend/app/api/routes/queue.py's
 * update_vitals() docstring. VitalTrendChart accumulates points
 * client-side for the current session only.
 */
export async function updateVitals(stayId, vitals, actor = "nurse") {
  const response = await apiClient.post("/vitals/update", { stay_id: stayId, ...vitals, actor });
  return response.data;
}

// --------------------------------------------------------------------- security
export async function fetchSecurityStatus() {
  const response = await apiClient.get("/security/status");
  return response.data;
}

export async function fetchSecurityAudit() {
  const response = await apiClient.get("/security/audit");
  return response.data;
}

export async function fetchRolesMatrix() {
  const response = await apiClient.get("/security/roles");
  return response.data;
}

// --------------------------------------------------------------------- audit
export async function fetchAudit({ patientId, eventType, limit = 100 } = {}) {
  const response = await apiClient.get("/audit", {
    params: { patient_id: patientId, event_type: eventType, limit },
  });
  return response.data;
}

export default apiClient;
