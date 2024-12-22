#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import joblib
import os

# Model dosyasının yolu
model_path = os.path.join(os.getcwd(), "random_forest_model.pkl")

# Modeli yükle
try:
    model = joblib.load(model_path)
    st.success("Model başarıyla yüklendi!")
except FileNotFoundError:
    st.error("Model dosyası bulunamadı. Lütfen 'random_forest_model.pkl' dosyasının doğru konumda olduğundan emin olun.")
    st.stop()
except Exception as e:
    st.error(f"Model yüklenirken bir hata oluştu: {e}")
    st.stop()

# Başlık
st.title("Dolandırıcılık Tespiti Uygulaması")

# Kullanıcıdan giriş verilerini al
st.subheader("Lütfen işlem özelliklerini giriniz:")
input_data = {}

# V1'den V28'e kadar olan veriler için giriş kutuları
for i in range(1, 29):
    feature = f"V{i}"
    input_data[feature] = st.number_input(f"{feature} değerini giriniz:", value=0.0, format="%.4f")

# Amount verisi için giriş
input_data['Amount'] = st.number_input("Amount (Miktar) değerini giriniz:", value=0.0, format="%.2f")

# Veri çerçevesine dönüştür
input_df = pd.DataFrame([input_data])

# Tahmin yap
if st.button("Tahmin Yap"):
    try:
        prediction = model.predict(input_df)
        result = "Dolandırıcılık" if prediction[0] == 1 else "Normal İşlem"
        st.success(f"Tahmin Sonucu: {result}")
    except Exception as e:
        st.error(f"Tahmin yapılırken bir hata oluştu: {e}")
