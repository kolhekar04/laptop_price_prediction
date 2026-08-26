import streamlit as st
from huggingface_hub import InferenceClient

def query_llama(latest_laptop, message_history):
    client = InferenceClient(
        model="meta-llama/Llama-3.1-8B-Instruct",
        token=st.secrets["HF_TOKEN"]
    )
    
    # System prompt injects the laptop context for the whole conversation
    system_prompt = {
        "role": "system", 
        "content": f"You are a helpful, concise AI shopping assistant expert in laptops. The user is asking about this specific laptop configuration: {latest_laptop}. Answer questions directly based on this context."
    }
    
    # Send system instructions + all historical messages to Llama 3.1
    messages = [system_prompt] + message_history
    
    response = client.chat_completion(
        messages=messages, 
        max_tokens=250
    )
    
    return response.choices[0].message.content