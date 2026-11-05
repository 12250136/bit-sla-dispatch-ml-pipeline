# ==============================================================================
# PROTOTIPE APLIKASI WEB INTERAKTIF STREAMLIT (app.py)
# Kelompok 04 - PT Bringin Inti Teknologi (Divisi IT Operational / DIIO SME)
# Mata Kuliah: Pembelajaran Mesin (S1) - Minggu 05
# ==============================================================================
import time
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sistem Prediksi Risiko SLA Tiket TI - PT Bringin Inti Teknologi",
    layout="wide"
)

@st.cache_resource
def load_serialized_pipeline():
    model = joblib.load("model_final_bit_sla.joblib")
    with open("model_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

model_pipeline, meta_info = load_serialized_pipeline()

# Sidebar Informasi Model & Kebijakan Human-in-the-Loop
with st.sidebar:
    st.markdown("### PT BRINGIN INTI TEKNOLOGI")
    st.caption("Divisi IT Operational (DIIO SME)")
    st.divider()
    st.markdown("**Spesifikasi Artefak Model:**")
    st.write(f"- File: `{meta_info['artifact_info']['filename']}`")
    st.write(f"- Ukuran: `{meta_info['artifact_info']['file_size_kb']} KB`")
    st.write(f"- Algoritma: `{meta_info['model_hyperparameters']['algorithm']}`")
    st.divider()
    st.markdown("**Metrik Evaluasi (Unseen Test Set):**")
    st.write(f"- Accuracy: **{meta_info['validation_metrics_test_set']['accuracy']*100:.2f}%**")
    st.write(f"- F1-Macro: **{meta_info['validation_metrics_test_set']['f1_macro']*100:.2f}%**")
    st.write(f"- Recall SLA Breach: **{meta_info['validation_metrics_test_set']['recall_class_1_breached']*100:.2f}%**")
    st.write(f"- ROC-AUC: **{meta_info['validation_metrics_test_set']['roc_auc']:.4f}**")
    st.write(f"- Mean Latency: **{meta_info['latency_benchmark_100_requests_ms']['mean_ms']} ms**")
    st.divider()
    st.markdown("**Kebijakan Human-in-the-Loop:**")
    st.success("Zona Hijau (< 40%): Auto-Dispatch Normal")
    st.warning("Zona Kuning (40% - 60%): Uncertainty Band - Wajib Verifikasi Koordinator")
    st.error("Zona Merah (> 60%): Eskalasi Prioritas P1 & Backup Co-PIC")

st.title("Dashboard Prediksi Risiko Pelanggaran SLA Tiket Helpdesk TI")
st.caption("Ticket Dispatching Risk Model (Pre-Resolution Inference Pipeline) — Kelompok 04 (Kelas 12.8A.12)")

# Pilihan Preset Skenario Uji
scenario_preset = st.selectbox(
    "Pilih Skenario Pengujian Cepat (Preset):",
    [
        "Skenario 1: Kasus Normal / Ideal (Pembuatan Profil OVPN Internal)",
        "Skenario 2: Kasus Berisiko Tinggi (Insiden Kritis Storage Alletra BRI)",
        "Skenario 3: Kasus Batas Ambang / Edge Case (Setup NGINX WAF Saat Beban Padat)",
        "Input Kustom Manual"
    ]
)

if "Skenario 1" in scenario_preset:
    def_po, def_cat, def_pic, def_tier = "Internal", "Unclassified", "ENG_01", 1
    def_dow, def_load, def_queue, def_urg = 2, 2.0, 479.0, 0.14
    def_desc = "Create User Profile OVPN"
    def_det = "Create User Profile OVPN Team Internal"
elif "Skenario 2" in scenario_preset:
    def_po, def_cat, def_pic, def_tier = "BRI", "Storage", "ENG_04", 3
    def_dow, def_load, def_queue, def_urg = 5, 8.0, 34.0, 0.86
    def_desc = "Troubleshoot Storage Alletra Tabanan down dan error replikasi cluster"
    def_det = "Cek log controller Storage Alletra Tabanan yang mengalami link down, lakukan failover LUN serta sinkronisasi ulang replikasi cluster Nutanix VM bersama tim Network dan Compute."
else:
    def_po, def_cat, def_pic, def_tier = "External", "Unclassified", "ENG_01", 2
    def_dow, def_load, def_queue, def_urg = 4, 5.0, 479.0, 0.472
    def_desc = "Setup and Config NGINX WAF microsite PARI"
    def_det = "Setup and Config NGINX WAF microsite PARI"

with st.form("ticket_dispatch_form"):
    st.subheader("Formulir Atribut Pra-Penugasan Tiket Baru (Pre-Dispatching Input)")
    col1, col2 = st.columns(2)
    with col1:
        project_owner = st.selectbox("1. Pemilik Proyek (Project Owner):", ["BRI", "External", "Internal"], index=["BRI", "External", "Internal"].index(def_po))
        service_cat = st.selectbox("2. Kategori Layanan TI (Service_Category):", ["Unclassified", "ICD", "Storage", "Network", "Server", "Britam Mobile", "Discovery", "CMDB"], index=["Unclassified", "ICD", "Storage", "Network", "Server", "Britam Mobile", "Discovery", "CMDB"].index(def_cat))
        pic_code = st.selectbox("3. Kode Teknisi PIC (Assigned_PIC_Code):", [f"ENG_{i:02d}" for i in range(1, 11)], index=[f"ENG_{i:02d}" for i in range(1, 11)].index(def_pic))
        complexity_tier = st.selectbox("4. Tingkat Kompleksitas Tugas (Task_Complexity_Tier):", [1, 2, 3], index=[1, 2, 3].index(def_tier), format_func=lambda x: {1: "1 = Low / Routine", 2: "2 = Medium / Configuration", 3: "3 = High / Critical Incident"}[x])
    with col2:
        submit_dow = st.slider("5. Hari Tiket Masuk (1=Senin s.d. 7=Minggu):", min_value=1, max_value=7, value=int(def_dow))
        daily_load = st.slider("6. Beban Tiket Harian (Daily_Ticket_Load):", min_value=1.0, max_value=15.0, value=float(def_load), step=1.0)
        queue_load = st.number_input("7. Antrean Kumulatif Teknisi (Engineer_Queue_Load):", min_value=1.0, max_value=600.0, value=float(def_queue))
        tfidf_urg = st.slider("8. Skor Urgensi Teks (TFIDF_Urgency_Score):", min_value=0.05, max_value=0.95, value=float(def_urg), step=0.01)

    action_desc = st.text_input("9. Deskripsi Singkat Tiket (Action Description):", value=def_desc)
    action_detail = st.text_area("10. Rincian Rencana Tindakan Teknis (Action Detail):", value=def_det)
    submitted = st.form_submit_button("Jalankan Prediksi Risiko SLA & Analisis XAI")

if submitted:
    t_start = time.perf_counter()
    is_weekend_peak = 1 if submit_dow >= 5 else 0
    is_cat_missing = 1 if service_cat == "Unclassified" else 0
    desc_len = max(len(action_desc.strip()), 1)
    detail_len = max(len(action_detail.strip()), 1)
    combined_txt = f"{action_desc} {action_detail}".lower().strip()

    input_df = pd.DataFrame([{
        "Project Owner": project_owner,
        "Service_Category": service_cat,
        "Assigned_PIC_Code": pic_code,
        "Task_Complexity_Tier": complexity_tier,
        "Submit_Day_of_Week": submit_dow,
        "Is_Weekend_or_Peak": is_weekend_peak,
        "Daily_Ticket_Load": daily_load,
        "Engineer_Queue_Load": queue_load,
        "Desc_Char_Length": desc_len,
        "Detail_Char_Length": detail_len,
        "Is_Category_Missing": is_cat_missing,
        "TFIDF_Urgency_Score": tfidf_urg,
        "Combined_Text": combined_txt
    }])

    pred_class = int(model_pipeline.predict(input_df)[0])
    pred_probs = model_pipeline.predict_proba(input_df)[0]
    prob_safe = float(pred_probs[0])
    prob_breach = float(pred_probs[1])
    latency_ms = (time.perf_counter() - t_start) * 1000.0

    st.divider()
    st.subheader(f"Hasil Keputusan Inferensi (Waktu Respon: {latency_ms:.2f} ms)")

    in_uncertainty_band = 0.40 <= prob_breach <= 0.60
    if in_uncertainty_band:
        st.warning(f"**PERINGATAN ZONA ABU-ABU (UNCERTAINTY BAND 40%-60%)** — Probabilitas Pelanggaran SLA: **{prob_breach*100:.2f}%** (vs Tepat Waktu: **{prob_safe*100:.2f}%**). Sistem menahan keputusan otomatis dan mewajibkan **Human-in-the-Loop Review** oleh Koordinator Helpdesk!")
    elif pred_class == 1:
        st.error(f"**STATUS PREDIKSI: 1 = SLA BREACHED (BERISIKO TINGGI MELANGGAR SLA)** — Keyakinan Risiko: **{prob_breach*100:.2f}%** (vs Tepat Waktu: **{prob_safe*100:.2f}%**). Rekomendasi: Eskalasi Prioritas P1 & Alokasikan Teknisi Pendamping (Co-PIC)!")
    else:
        st.success(f"**STATUS PREDIKSI: 0 = SLA MET (TEPAT WAKTU / AMAN)** — Keyakinan Aman: **{prob_safe*100:.2f}%** (Risiko Breach hanya **{prob_breach*100:.2f}%**). Rekomendasi: Lanjutkan Auto-Dispatch Sesuai Antrean Standar.")

    st.progress(prob_breach, text=f"Indeks Risiko Pelanggaran SLA: {prob_breach*100:.2f}%")

    col_act1, col_act2 = st.columns(2)
    with col_act1:
        st.button("Setujui Rekomendasi AI & Kirim Notifikasi PIC")
    with col_act2:
        st.button("Override Manual (Human-in-the-Loop Review & Catat Feedback)")
