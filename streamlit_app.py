import streamlit as st
from openai import OpenAI

st.title("Генератор фидбеков")

num_personas = st.number_input("Сколько персон?", min_value=1, max_value=10, value=3)
personas = [st.text_input(f"Имя персоны {i+1}") for i in range(num_personas)]
style = st.selectbox("Стиль фидбэка", ["Формальный", "Дружеский", "Критический"])

if st.button("Сгенерировать"):
    
    client = OpenAI(
        base_url="https://api.studio.nebius.ai/v1/",
        api_key=st.secrets("NEBIUS_API_KEY"),
    )

    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=[
            {
            "role": "system",
            "content": "You are a chemistry expert. Add jokes about cats to your responses from time to time."
            },
            {
            "role": "user",
            "content": "Hello!"
            },
            {
            "role": "assistant",
            "content": "Hello! How can I assist you with chemistry today? And did you hear about the cat who became a chemist? She had nine lives, but she only needed one formula!"
            }
        ],
        max_tokens=100,
        temperature=1,
        top_p=1,
        top_k=50,
        n=1,
        stream=false,
        stream_options=null,
        stop=null,
        presence_penalty=0,
        frequency_penalty=0,
        logit_bias=null,
        logprobs=false,
        top_logprobs=null,
        user=null,
        extra_body={
            "guided_json": {"type": "object", "properties": {...}}
        },
        response_format={
            "type": "json_object"
        }
    )

    st.write(completion.to_json())  
