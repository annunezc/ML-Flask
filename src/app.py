from xgboost import XGBRegressor
import streamlit as st
import json

model = XGBRegressor()
model.load_model('../models/xgboost_optimizado.json')

with open('../data/processed/dic_cl.json','r', encoding='utf-8') as archivo:
    dic_cl = json.load(archivo)
with open('../data/processed/dic_el.json','r', encoding='utf-8') as archivo:
    dic_el = json.load(archivo)
with open('../data/processed/dic_et.json','r', encoding='utf-8') as archivo:
    dic_et = json.load(archivo)
with open('../data/processed/dic_er.json','r', encoding='utf-8') as archivo:
    dic_er = json.load(archivo)
with open('../data/processed/dic_cs.json','r', encoding='utf-8') as archivo:
    dic_cs = json.load(archivo)

st.title("Predicción Sueldo Data Science")

val1 = st.slider("Año", min_value = 2020, max_value = 2025, step = 1)
val2 = st.selectbox(
    "Nivel de Conocimiento del Empleado",
    (dic_el.keys()),
    index=None,
    placeholder="Selecciona nivel de experiencia....",
)
val3 = st.selectbox(
    "Tipo de Contrato",
    (dic_et.keys()),
    index=None,
    placeholder="Selecciona tipo de empleo....",
)
val4 = st.slider("Porcentaje de Trabajo Remoto", min_value = 0, max_value = 100, step = 1)
val5 = st.selectbox(
    "Tamaño de la Compañia",
    (dic_cs.keys()),
    index=None,
    placeholder="Selecciona el tamaño...",
)

val6 = st.selectbox(
    "Pais de Residencia del Empleado",
    (dic_er.keys()),
    index=None,
    placeholder="Selecciona el pais...",
)

val7 = st.selectbox(
    "Pais de Origen de la Compañia",
    (dic_cl.keys()),
    index=None,
    placeholder="Selecciona el pais...",
)


if st.button("Predecir"):
    prediction = str(model.predict([[val1, dic_el[val2], dic_et[val3], val4,dic_cs[val5],dic_er[val6],dic_cl[val7]]])[0])
    pred_class = prediction
    st.write("Prediction:", pred_class)