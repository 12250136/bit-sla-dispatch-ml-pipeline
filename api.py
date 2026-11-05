# ==============================================================================
# LAYANAN MICROSERVICE REST API INFERENSI (api.py)
# Kelompok 04 - PT Bringin Inti Teknologi (Divisi IT Operational / DIIO SME)
# Menggunakan FastAPI & Pydantic Schema Validation
# ==============================================================================
import time
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="BIT DIIO SME - SLA Breach Prediction REST API",
    description="Endpoint Inferensi Real-Time Ticket Dispatching Risk Model (Kelompok 04)",
    version="1.5.0"
)

# Muat artefak pipeline utuh sekali saat startup server
MODEL_PIPELINE = joblib.load("model_final_bit_sla.joblib")

class TicketInputSchema(BaseModel):
    ticket_id: Optional[str] = Field(default="TICKET_API_001", description="Kode unik tiket anonim")
    project_owner: str = Field(default="BRI", description="Pemilik proyek: BRI, External, atau Internal")
    service_category: Optional[str] = Field(default="Unclassified", description="Kategori layanan TI")
    assigned_pic_code: Optional[str] = Field(default="ENG_01", description="Kode anonim engineer PIC")
    task_complexity_tier: int = Field(default=2, ge=1, le=3, description="Tingkat kompleksitas (1=Low, 2=Medium, 3=High)")
    submit_day_of_week: int = Field(default=3, ge=1, le=7, description="Hari tiket masuk (1=Senin s.d. 7=Minggu)")
    daily_ticket_load: float = Field(default=4.0, ge=1.0, le=50.0, description="Jumlah tiket masuk pada hari yang sama")
    engineer_queue_load: float = Field(default=99.0, ge=1.0, le=1000.0, description="Total beban kumulatif tiket pada PIC")
    tfidf_urgency_score: float = Field(default=0.35, ge=0.0, le=1.0, description="Skor urgensi kata kunci TF-IDF")
    action_description: str = Field(..., min_length=3, description="Deskripsi singkat keluhan/tiket TI")
    action_detail: Optional[str] = Field(default="", description="Rincian rencana tindakan teknis")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_artifact": "model_final_bit_sla.joblib"}

@app.post("/predict")
def predict_sla_breach(item: TicketInputSchema):
    t0 = time.perf_counter()
    try:
        desc_clean = item.action_description.strip()
        det_clean = (item.action_detail or "").strip()
        desc_len = max(len(desc_clean), 1)
        det_len = max(len(det_clean), 16) if det_clean else 41.0
        is_weekend_peak = 1 if item.submit_day_of_week >= 5 else 0
        is_cat_missing = 1 if (not item.service_category or item.service_category == "Unclassified") else 0
        combined_text = f"{desc_clean} {det_clean}".lower().strip()

        df_input = pd.DataFrame([{
            "Project Owner": item.project_owner,
            "Service_Category": item.service_category or "Unclassified",
            "Assigned_PIC_Code": item.assigned_pic_code or "ENG_01",
            "Task_Complexity_Tier": item.task_complexity_tier,
            "Submit_Day_of_Week": item.submit_day_of_week,
            "Is_Weekend_or_Peak": is_weekend_peak,
            "Daily_Ticket_Load": item.daily_ticket_load,
            "Engineer_Queue_Load": item.engineer_queue_load,
            "Desc_Char_Length": desc_len,
            "Detail_Char_Length": det_len,
            "Is_Category_Missing": is_cat_missing,
            "TFIDF_Urgency_Score": item.tfidf_urgency_score,
            "Combined_Text": combined_text
        }])

        pred_cls = int(MODEL_PIPELINE.predict(df_input)[0])
        probs = MODEL_PIPELINE.predict_proba(df_input)[0]
        prob_safe = round(float(probs[0]), 4)
        prob_breach = round(float(probs[1]), 4)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        in_uncertainty = 0.40 <= prob_breach <= 0.60
        if in_uncertainty:
            action_rec = "HUMAN_IN_THE_LOOP_VERIFICATION_REQUIRED"
        elif pred_cls == 1:
            action_rec = "ESCALATE_PRIORITY_AND_ASSIGN_BACKUP_PIC"
        else:
            action_rec = "STANDARD_DISPATCH_QUEUE"

        return {
            "status": "SUCCESS",
            "ticket_id": item.ticket_id,
            "predicted_class": pred_cls,
            "predicted_label": "SLA_BREACHED" if pred_cls == 1 else "SLA_MET",
            "probability_sla_met": prob_safe,
            "probability_sla_breached": prob_breach,
            "confidence_score": max(prob_safe, prob_breach),
            "uncertainty_band_flag": in_uncertainty,
            "recommended_action": action_rec,
            "inference_latency_ms": latency_ms
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference Pipeline Error: {str(e)}")
