import pickle
import numpy as np
import streamlit as st

@st.cache_resource # Caches the model so it only loads once
def load_models():
    pipe = pickle.load(open('pipe.pkl','rb'))
    df = pickle.load(open('df.pkl','rb'))
    return pipe, df

def predict_price(pipe, company, type_name, ram, weight, touchscreen_val, ips_val, ppi, cpu, hdd, ssd, gpu, os):
    query = np.array([company, type_name, ram, weight, touchscreen_val, ips_val, ppi, cpu, hdd, ssd, gpu, os]).reshape(1,12)
    return int(np.exp(pipe.predict(query)[0]))