#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import joblib

# Modeli yükle
model = joblib.load("random_forest_model.pkl")

# Başlık
st.write("Lütfen özellik değerlerini giriniz.")

# Kullanıcıdan giriş verilerini al
input_data = {}

# V1'den V28'e kadar olan veriler için giriş kutuları
for i in range(1, 29):
    feature = f"V{i}"
    input_data[feature] = st.number_input(f"{feature} değerini giriniz:", value=0.0)

# Amount verisi için giriş
input_data['Amount'] = st.number_input("Amount (Miktar) değerini giriniz:", value=0.0)

# Veri çerçevesine dönüştür
input_df = pd.DataFrame([input_data])

# Tahmin yap
if st.button("Tahmin Yap"):
    prediction = model.predict(input_df)
    result = "Dolandırıcılık" if prediction[0] == 1 else "Normal İşlem"
    st.write(f"Tahmin Sonucu: {result}")


# In[ ]:




