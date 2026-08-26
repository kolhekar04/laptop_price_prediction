import streamlit as st
import pandas as pd
from ml_backend import load_models, predict_price
from ai_assistant import query_llama

# MUST be the first Streamlit command!
st.set_page_config(page_title="Laptop Price Predictor", layout="wide")

# 1. Initialization
pipe, df = load_models()
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("💻 Laptop Price Prediction & Analysis")

# 2. Main Tabs
tab1, tab2 = st.tabs(["⚙️ Input & Prediction", "📊 Compare & Ask AI"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox('Brand', df['Company'].unique())
        type_name = st.selectbox('Type', df['TypeName'].unique())
        ram = st.selectbox('RAM (GB)', [2,4,6,8,12,16,24,32,64])
        weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=1.5, step=0.1)
        touchscreen = st.radio('Touchscreen', ['No','Yes'], horizontal=True)
        ips = st.radio('IPS Display', ['No','Yes'], horizontal=True)
    with col2:
        screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.0, step=0.1)
        resolution = st.selectbox('Screen Resolution', [
            '1920x1080','1366x768','1600x900','3840x2160',
            '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'
        ])
        cpu = st.selectbox('CPU Brand', df['cpu brand'].unique())
        hdd = st.selectbox('HDD (GB)', [0,128,256,512,1024,2048])
        ssd = st.selectbox('SSD (GB)', [0,8,128,256,512,1024])
        gpu = st.selectbox('GPU Brand', df['GPU BRAND'].unique())
        os = st.selectbox('Operating System', df['os'].unique())

    if st.button('Predict Price', type="primary"):
        touchscreen_val = 1 if touchscreen == 'Yes' else 0
        ips_val = 1 if ips == 'Yes' else 0
        x_res, y_res = map(int, resolution.split('x'))
        ppi = ((x_res**2) + (y_res**2))**0.5 / screen_size

        price = predict_price(pipe, company, type_name, ram, weight, touchscreen_val, ips_val, ppi, cpu, hdd, ssd, gpu, os)
        
        new_entry = {
            "company": company, "type": type_name, "ram": ram, "weight": weight,
            "touchscreen": touchscreen, "ips": ips, "screen_size": screen_size,
            "resolution": resolution, "cpu": cpu, "hdd": hdd, "ssd": ssd,
            "gpu": gpu, "os": os, "price": price,
            "specs": f"{company} {type_name} | {cpu} | {ram}GB RAM | {gpu}",
            "full_details": f"{company} laptop with {cpu}, {gpu}, {ram}GB RAM, {ssd}GB SSD, {hdd}GB HDD, running {os}."
        }
        
        if not st.session_state.history or st.session_state.history[-1]['full_details'] != new_entry['full_details']:
            st.session_state.history.append(new_entry)
            
        st.success(f"Estimated Price: ₹ {price:,}")

with tab2:
    if not st.session_state.history:
        st.info("Run a prediction in the first tab to see comparisons here.")
    else:
        st.subheader("🔍 Side-by-Side Specification Comparison")
        
        history_labels = [
            f"#{i+1}: {h.get('company')} {h.get('type')} - ₹{h.get('price'):,}" 
            for i, h in enumerate(st.session_state.history)
        ]
        
        selected_comparisons = st.multiselect(
            "Select laptops to compare side-by-side:",
            options=history_labels,
            default=history_labels[-2:] if len(history_labels) >= 2 else history_labels
        )
        
        if selected_comparisons:
            selected_indices = [history_labels.index(lbl) for lbl in selected_comparisons]
            compare_data = [st.session_state.history[i] for i in selected_indices]
            
            spec_labels = [
                'Brand', 'Type', 'RAM (GB)', 'Weight (kg)', 'Touchscreen', 
                'IPS Display', 'Screen Size', 'Resolution', 'CPU', 
                'HDD (GB)', 'SSD (GB)', 'GPU', 'OS', 'Estimated Price (₹)'
            ]
            
            table_dict = {"Specification": spec_labels}
            for idx, item in enumerate(compare_data):
                col_name = f"Laptop {idx+1} ({item.get('company')} {item.get('type')})"
                table_dict[col_name] = [
                    item.get('company'), item.get('type'), item.get('ram'),
                    item.get('weight'), item.get('touchscreen'), item.get('ips'),
                    item.get('screen_size'), item.get('resolution'), item.get('cpu'),
                    item.get('hdd'), item.get('ssd'), item.get('gpu'),
                    item.get('os'), f"₹ {item.get('price'):,}"
                ]
            
            comparison_df = pd.DataFrame(table_dict).set_index("Specification")
            st.table(comparison_df)
            
            # UX Polish: Download Button for Comparison Report
            csv_data = comparison_df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Comparison Report (CSV)",
                data=csv_data,
                file_name="laptop_comparison_report.csv",
                mime="text/csv"
            )
            
            # Price Bar Chart
            st.subheader("📊 Price Comparison Chart")
            df_chart = pd.DataFrame({
                "Laptop": [f"#{i+1} {h.get('company')} {h.get('type')}" for h in enumerate(compare_data) if False] # Placeholder fix for loop below
            }) # Replaced cleanly below:
            
            df_chart = pd.DataFrame({
                "Laptop": [f"#{i+1} {h.get('company')} {h.get('type')}" for i, h in enumerate(compare_data)],
                "Price (₹)": [h.get('price') for h in compare_data]
            }).set_index("Laptop")
            st.bar_chart(df_chart, color="#27AE60")
            
        else:
            st.warning("Please select at least one laptop configuration above to view the comparison table.")

        st.divider()
        
        # --- HARDWARE ASSISTANT CHAT WITH UX CLEAR BUTTON ---
        col_chat_title, col_chat_btn = st.columns([0.8, 0.2])
        with col_chat_title:
            st.subheader("🤖 Hardware Assistant")
        with col_chat_btn:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_messages = []
                st.rerun()

        latest = st.session_state.history[-1]['full_details']
        st.caption(f"Active Context: **{latest}**")
        
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
            
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
        if question := st.chat_input("Ask a follow-up question about this configuration..."):
            st.session_state.chat_messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
                
            with st.chat_message("assistant"):
                with st.spinner("Llama 3.1 is thinking..."):
                    answer = query_llama(latest, st.session_state.chat_messages)
                    st.write(answer)
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})