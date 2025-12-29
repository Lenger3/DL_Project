import gradio as gr
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

def kalp_hastaligi_tahmin_sistemi(file_obj):
    try:
        model = tf.keras.models.load_model('heart_disease_model.keras')
        scaler = joblib.load('std_scaler.bin')
        # Bu dosyayı eğitim notebook'unda joblib.dump(X_train.columns.tolist(), 'model_columns.bin') ile kaydetmiş olmalısın
        model_columns = joblib.load('model_columns.bin') 
        
        df = pd.read_csv(file_obj.name)
        df_original = df.copy()
        
        # 1. Kategorik Dönüşümler (Eğitimle BİREBİR AYNI OLMALI: drop_first=True ekledik)
        kategorik_sutunlar = ['Slope of ST', 'Number of vessels fluro', 'EKG results', 'Chest pain type', 'Thallium']
        
        for col in kategorik_sutunlar:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int) # drop_first eklendi!
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
        
        # 2. Sütunları Hizala (25 sütunu 20'ye düşürecek ve sıraya sokacak yer burası)
        df = df.reindex(columns=model_columns, fill_value=0)
        
        # 3. Sayısal Sütunları Ölçeklendir
        sayisal_sutunlar = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
        df[sayisal_sutunlar] = scaler.transform(df[sayisal_sutunlar])
        
        # 4. Tahmin
        tahminler_proba = model.predict(df)
        tahminler_sinif = (tahminler_proba > 0.5).astype(int)
        
        df_original['Risk_Durumu'] = ["Presence" if x == 1 else "Absence" for x in tahminler_sinif]
        
        output_path = "sonuclar.csv"
        df_original.to_csv(output_path, index=False)
        return output_path

    except Exception as e:
        # Hata mesajını print et ki terminalden görebilesin
        print(f"HATA DETAYI: {e}")
        return None # Hata durumunda dosya döndürme
    
# Gradio Arayüz Tasarımı
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Kalp Hastalığı Risk Analizi (Toplu Tahmin)")
    gr.Markdown("Lütfen hastaların klinik verilerini içeren ham CSV dosyasını yükleyin.")
    
    with gr.Row():
        dosya_input = gr.File(label="CSV Dosyası Yükle", file_types=[".csv"])
        dosya_output = gr.File(label="Tahmin Edilmiş CSV'yi İndir")
    
    predict_btn = gr.Button("Analiz Et ve Sonuçları Hazırla", variant="primary")
    predict_btn.click(fn=kalp_hastaligi_tahmin_sistemi, inputs=dosya_input, outputs=dosya_output)

    gr.Markdown("### Önemli Not:")
    gr.Markdown("Yüklenen CSV dosyasında şu sütunlar mutlaka bulunmalıdır: *Age, Sex, BP, Cholesterol, Max HR, ST depression, Slope of ST, Number of vessels fluro, EKG results, Chest pain type, Thallium*")

# Uygulamayı başlat
if __name__ == "__main__":
    demo.launch(share=True)