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
        model="deepseek-ai/DeepSeek-V3",
        
        messages=[
            {
            "role": "user",
            "content": promt
            }
        ],
        #max_tokens=100,
        #temperature=1,
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
    
    return completion.choices[0].message.content

def upload_to_airtable(data):
    api = Api(st.secrets.AIRTABLE_API_TOKEN)
    table = api.table(st.secrets.AIRTABLE_BASE_ID, st.secrets.AIRTABLE_TABLE_ID)
    
    #records = [{"fields": person} for person in data["records"]]]
    records = json.loads(data)
    #st.write(records)
    #st.write(len(records))
    #for person in records["records"]:
    #    st.write(person)
    #    response = table.create(person)
    #    st.write(response.json())

    response = table.batch_create(records["records"])
    st.write(response)
    return len(response.json()["records"])

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
                "Name": "случайное имя по полу",
                "Gender": "мужской/женский",
                "Marital status id": "1 случайный ID из списка ["recB5tgjv9b2o2aPs", "recCO6SsPJva6VXmU", "recmMt8VCjPpC9O4I", "recEetmWrUmBJEh2I", "recTqqBXrv4PfjNDW"]",
                "Children": "число ≥0 с учетом возраста",
                "Income": "15-1500 тыс.руб",
                "Age": "3-99 с логикой",
                "Interests": "2 уникальных ID из списка ["rec0ExsZkszeavFi9", "recWPIWQQuEfQUfhC", "rec8DycBdP3nfIYAG", "recEdaM1lQmsr3Au8", "rec7kDVeNmJSleNKD"]",
                "Description": "1-2 предложения с характером и интересами"
            }},
            "logic_checks": [
                "Возраст >= 18 если есть дети",
                "Вдовцы/вдовы >= 50 лет",
                "Уникальные категории интересов",
                "Согласованность статуса и описания"
            ]
        }}
    }}
    
    Верни структурированный JSON в формате и только его без каких либо пояснений или чего-либо еще в ответе: 
    
    {{
        "records":[
        {{
            "Name": "Иван Петров",
            "Gender": "Мужской",
            "Marital status ID": [
            "recB5tgjv9b2o2aPs"
            ],
            "Children": 0,
            "Income": 120000,
            "Age": 28,
            "Interests": [
            "rec0ExsZkszeavFi9"
            ]
        }},
        {{
            "Name": "Анна Смирнова",
            "Gender": "Женский",
            "Marital status ID": [
            "recCO6SsPJva6VXmU"
            ],
            "Children": 2,
            "Income": 80000,
            "Age": 35,
            "Interests": [
            "rechEsuoL1qsAFAiP"
            ]
        }}
    ]
    }}'
    
    
    
    """


    with st.spinner("Генерация персонажей..."):
        generated_data = OpenAIChat(prompt)
        st.write(generated_data)
    
    st.success("Персонажи успешно сгенерированы!")
    
    with st.spinner("Загрузка в Airtable..."):
        uploaded_count = upload_to_airtable(generated_data)
    
    st.success(f"Успешно загружено {uploaded_count} записей в Airtable!")
    
