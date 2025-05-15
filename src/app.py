from utils import db_connect
#engine = db_connect()

from pickle import load
import streamlit as st

model = load(open("/workspaces/Maria_Miura1_streamlit/models/tree_classifier_crit-entro_maxdepth-5_minleaf-4_minsplit2_42.sav", "rb"))
import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np

st.title("Predicción de Diabetes")
from joblib import load

# Carga del modelo con ruta relativa correcta
model = load("/workspaces/Maria_Miura1_streamlit/models/tree_classifier_crit-entro_maxdepth-5_minleaf-4_minsplit2_42.sav")


# Formulario de entrada de usuario
st.subheader("Introduce los datos del paciente")

def user_input_features():
    pregnancies = st.number_input("Embarazos", min_value=0, max_value=20, value=1)
    glucose = st.slider("Glucosa", 0, 200, 100)
    blood_pressure = st.slider("Presión Arterial", 0, 150, 70)
    skin_thickness = st.slider("Grosor de piel", 0, 100, 20)
    insulin = st.slider("Insulina", 0, 900, 80)
    bmi = st.slider("IMC", 0.0, 70.0, 25.0)
    dpf = st.slider("Índice Hereditario (DPF)", 0.0, 2.5, 0.5)
    age = st.slider("Edad", 10, 100, 33)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Botón para predecir
if st.button("Predecir"):
    prediction = model.predict(input_df)
    resultado = "✅ Diabetes" if prediction[0] == 1 else "🟢 No Diabetes"
    st.success(f"Resultado de la predicción: {resultado}")