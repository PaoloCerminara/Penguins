import streamlit as st
import joblib 
import pandas as pd

st.title("app machine learning")

model = joblib.load("penguinspipe.pkl")

island= st.selectbox("inserire l'isola", ["Torgersen0", "Dream", "Biscoe"])
bill_length_mm = st.number_input("inserire la lunghezza", 5.0, 50.0, 39.1)
bill_depth_mm = st.number_input("inserire la larghezza", 5.0, 50.0, 18.7)
flipper_length_mm = st.number_input("inserire la lunghezza della pinna ", 5.0, 250.0, 181.0)
body_mass_g = st.number_input("inserire la il peso", 5.0, 5000.0, 3750.1)
sex = st.selectbox("inserire il sesso", ["male", "female"])


# island= 'Torgersen'
# bill_length_mm = 39.1
# bill_depth_mm = 18.7
# flipper_length_mm = 181
# body_mass_g = 3750
# sex = 'male'

data = {
        "island": [island],
        "bill_length_mm": [bill_length_mm],
        "bill_depth_mm": [bill_depth_mm],
        "flipper_length_mm":[flipper_length_mm],
        "body_mass_g": [body_mass_g],
        "sex": [sex],
        }

input_df = pd.DataFrame(data)
res = model.predict(input_df).astype(int)[0]

classes = {0:'Adelie',
           1:'Gentoo',
           2:'Chinstrap',
           }

y_pred = classes[res]

if st.button("prediction"):
    st.success(f"la specie predetta è {y_pred} ")