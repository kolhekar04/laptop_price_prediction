import os
import streamlit as st
from huggingface_hub import InferenceClient

def query_llama(latest_laptop, message_history):
    # Try fetching from Render's environment variables first, then local Streamlit secrets
    hf_token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")
    
    if not hf_token:
        raise ValueError("HF_TOKEN is missing! Please configure it in your Render environment variables or .streamlit/secrets.toml")

    client = InferenceClient(
        model="meta-llama/Llama-3.1-8B-Instruct",
        token=hf_token
    )
    
    system_prompt = {
        "role": "system", 
        "content": f"You are a helpful, concise AI shopping assistant expert in laptops. The user is asking about this specific laptop configuration: {latest_laptop}. Answer questions directly based on this context."
    }
    
    messages = [system_prompt] + message_history
    
    response = client.chat_completion(
        messages=messages, 
        max_tokens=250
    )
    
    return response.choices[0].message.content