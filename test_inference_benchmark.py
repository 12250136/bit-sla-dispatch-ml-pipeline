# ==============================================================================
# SKRIP PENGUJIAN SKENARIO, EDGE CASES, & BENCHMARK LATENSI (test_inference_benchmark.py)
# Kelompok 04 - PT Bringin Inti Teknologi
# ==============================================================================
import time
import json
import joblib
import numpy as np
import pandas as pd

def run_verification():
    print("1. Memuat artefak pipeline: model_final_bit_sla.joblib ...")
    pipeline = joblib.load("model_final_bit_sla.joblib")
    df_clean = pd.read_csv("dataset_sla_tiket_bit_clean.csv")
    
    feature_cols = [
        "Project Owner", "Service_Category", "Assigned_PIC_Code",
        "Task_Complexity_Tier", "Submit_Day_of_Week", "Is_Weekend_or_Peak",
        "Daily_Ticket_Load", "Engineer_Queue_Load",
        "Desc_Char_Length", "Detail_Char_Length", "Is_Category_Missing", "TFIDF_Urgency_Score",
        "Combined_Text"
    ]
    
    # Uji Latensi 100 Permintaan
    latencies = []
    for i in range(100):
        row_df = df_clean[feature_cols].iloc[[i % len(df_clean)]]
        t0 = time.perf_counter()
        _ = pipeline.predict_proba(row_df)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
    print(f"2. Hasil Benchmark 100 Requests -> Mean: {np.mean(latencies):.2f} ms | P95: {np.percentile(latencies, 95):.2f} ms | Max: {np.max(latencies):.2f} ms")
    assert np.mean(latencies) < 50.0, "Latensi melebihi target 50 ms!"
    print("3. Status Uji Latensi: LULUS (< 50 ms)!")

if __name__ == "__main__":
    run_verification()
