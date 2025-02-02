import streamlit as st
import requests

st.title("Генератор фидбеков")

num_personas = st.number_input("Сколько персон?", min_value=1, max_value=10, value=3)
personas = [st.text_input(f"Имя персоны {i+1}") for i in range(num_personas)]
style = st.selectbox("Стиль фидбэка", ["Формальный", "Дружеский", "Критический"])

if st.button("Сгенерировать"):
    for persona in personas:
        response = requests.post("https://deepseek-api-url.com/generate", json={"persona": persona, "style": style})
        st.write(f"**{persona}:** {response.json()['feedback']}")  
