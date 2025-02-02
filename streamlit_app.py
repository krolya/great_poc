import streamlit as st
import json
from openai import OpenAI
from pyairtable import Api

def OpenAIChat(promt):
    client = OpenAI(
        base_url="https://api.studio.nebius.ai/v1/",
        api_key=st.secrets.NEBIUS_API_KEY,
    )

    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=[
            {
            "role": "user",
            "content": promt
            }
        ],
        max_tokens=1000,
        temperature=1,
        #top_p=1,
        #top_k=50,
        #n=1,
        #stream=False,
        #stream_options=null,
        #stop=null,
        #presence_penalty=0,
        #frequency_penalty=0,
        #logit_bias=null,
        #logprobs=False,
        #top_logprobs=null,
        #user=null,
        #extra_body={
        #    "guided_json": {"type": "object", "properties": {...}}
        #},
        response_format={
            "type": "json_object"
        }
    )
    
    return json.loads(completion.choices[0].message.content)

def upload_to_airtable(data):
    api = Api(st.secrets.AIRTABLE_API_KEY)
    table = api.table(st.secrets.AIRTABLE_BASE_ID, st.secrets.AIRTABLE_TABLE_NAME)
    
    records = [{"fields": person} for person in data["characters"]]
    response = table.batch_create(records)
    return len(response)

with st.form("persona_form"):
    num_people = st.number_input("Количество персон", min_value=1, max_value=20, value=1)
    submitted = st.form_submit_button("Сгенерировать")

if submitted:
    prompt = f"""Сгенерируй JSON-объект со случайными персонажами по следующим правилам:

    Правила:
    {{
        "count": {num_people},
        "rules": {{
            "fields": {{
                "name": "случайное имя по полу",
                "gender": "мужской/женский",
                "marital status id": {{
                    "М": ["1-Холост", "2-В браке", "3-Гражданский брак", "4-Разведен", "5-Вдовец"],
                    "Ж": ["1-Незамужем", "2-В браке", "3-Гражданский брак", "4-Разведена", "5-Вдова"]
                }},
                "children": "число ≥0 с учетом возраста",
                "income": "15-1500 тыс.руб",
                "age": "3-99 с логикой",
                "interests": "3-5 уникальных ID из 7-111",
                "description": "1-2 предложения с характером и интересами"
            }},
            "logic_checks": [
                "Возраст >= 18 если есть дети",
                "Вдовцы/вдовы >= 50 лет",
                "Уникальные категории интересов",
                "Согласованность статуса и описания"
            ]
        }}
    }}
    
    Верни структурированный JSON в формате: {{"characters": [{{..., ...]}}}}"""


    promt = "Say hello!"


    client = OpenAI(
        base_url="https://api.studio.nebius.ai/v1/",
        api_key=st.secrets.NEBIUS_API_KEY,
    )

    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=[
            {
            "role": "user",
            "content": promt
            }
        ],
        max_tokens=100,
        temperature=1,
        #top_p=1,
        #top_k=50,
        #n=1,
        #stream=False,
        #stream_options=null,
        #stop=null,
        #presence_penalty=0,
        #frequency_penalty=0,
        #logit_bias=null,
        #logprobs=False,
        #top_logprobs=null,
        #user=null,
        #extra_body={
        #    "guided_json": {"type": "object", "properties": {...}}
        #},
        response_format={
            "type": "json_object"
        }
    )

    st.write(completion.choices[0].message.content)

    '''
    try:
        #with st.spinner("Генерация персонажей..."):
        generated_data = OpenAIChat(prompt)
        st.write(generated_data)
        
        st.success("Персонажи успешно сгенерированы!")
        st.json(generated_data)

        #with st.spinner("Загрузка в Airtable..."):
        uploaded_count = upload_to_airtable(generated_data)
        
        st.success(f"Успешно загружено {uploaded_count} записей в Airtable!")
    
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
        st.stop()
    '''
