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

def GeneratePerson():
    prompt = f"""Сгенерируй JSON-объект со случайными персонажами по следующим правилам:

    Правила:
    {{
        "count": 5,
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

# Настройка страницы
st.set_page_config(page_title="Более нормальный человек", layout="wide")
    

# Добавляем CSS для задания высоты заголовка (примерно 5% от высоты экрана)
#st.markdown("""
#<style>
#    .header {
#        height: 5vh;
#        display: flex;
#        align-items: center;
#        justify-content: center;
#    }
#</style>
#""", unsafe_allow_html=True)

# Верхняя часть — заголовок
#st.markdown('<div class="header"><h1>Генерация персон</h1></div>', unsafe_allow_html=True)

# Разбиваем остальную область на две колонки с пропорцией 30:70
col_left, col_right = st.columns([3, 7])

# Левая колонка: "Целевая аудитория" и фильтры
with col_left:
    st.header("Целевая аудитория")

    with st.expander("Основные настройки", expanded=True):

        # 5.0. Слайдер для выбора количества персон для генерации
        number_of_persons = st.slider("Количество персон для генерации", min_value=0, max_value=10000, value=20)

        # 5.1. Слайдер для выбора соотношения мужчин и женщин (0-100%)
        gender_ratio = st.slider("Процент мужчин в выборке (%)", min_value=0, max_value=100, value=50)

        # 5.2. Двойной слайдер для выбора диапазона возраста
        age_range = st.slider("Возраст", min_value=4, max_value=100, value=(18, 60))

        # 5.3. Доход
        income_options = ["Низкий", "Низкий плюс"," Средний", "Средний плюс","Высокий","Высокий плюс"]
        income_selected = st.multiselect("Выберите группу доходов", options=income_options,
                                            default=income_options)

    # 5.3. Фильтр «Образование»
    with st.expander("Образование", expanded=True):
        education_options = ["Среднее", "Неоконченное высшее", "Высшее"]
        education_selected = st.multiselect("Выберите образование", options=education_options,
                                            default=education_options)

    # 5.4. Фильтр «Регион проживания»
    all_regions = [
        "Москва",
        "Московская область",
        "Санкт-Петербург",
        "Новосибирская область",
        "Свердловская область",
        "Краснодарский край",
        "Республика Татарстан",
        "Челябинская область",
        "Самарская область",
        "Оренбургская область"
    ]
    with st.expander("Регион проживания", expanded=True):
        # Кнопки для выбора/снятия всех флажков
        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button("Выбрать все", key="select_all_regions"):
            for region in all_regions:
                st.session_state[f"region_{region}"] = True
        if col_btn2.button("Снять все", key="deselect_all_regions"):
            for region in all_regions:
                st.session_state[f"region_{region}"] = False

        selected_regions = []
        for region in all_regions:
            # По умолчанию Москва и Московская область включены
            default = True if region in ["Москва", "Московская область"] else False
            checked = st.checkbox(
                region,
                value=st.session_state.get(f"region_{region}", default),
                key=f"region_{region}"
            )
            if checked:
                selected_regions.append(region)

        # 5.5. Фильтр «Размер населенного пункта»
        st.markdown("#### Размер населенного пункта")
        city_size_options = [
            "До 100 0000 человек",
            "От 100 000 до 500 000",
            "от 500 000 до 1 000 000",
            "свыше 1 000 000"
        ]
        city_size_selected = st.multiselect("Выберите размер населенного пункта", options=city_size_options,
                                            default=city_size_options)

    with st.expander("Семейное положение", expanded=True):

        # 5.8. Фильтр «Семейное положение»
        marital_options = ["В браке", "Разведен(-а)", "В отношениях", "Одинок (-а)"]
        marital_selected = st.multiselect("Выберите семейное положение", options=marital_options,
                                        default=marital_options)
        # 5.6. Двойной слайдер для выбора диапазона количества детей
        children_count = st.slider("Количество детей", min_value=0, max_value=10, value=(0, 3))

        # 5.7. Двойной слайдер для выбора диапазона возраста детей
        children_age = st.slider("Возраст детей", min_value=0, max_value=18, value=(0, 18))
  
    
    # 5.9. Поле для ввода тэгов
    tags = st.text_input("Тэги", placeholder="Введите тэги через запятую")

# Правая колонка: "Генерация"
with col_right:
    st.header("Генерация")
    #st.write("Здесь можно разместить настройки генерации или результаты.")

    # Например, можно добавить кнопку для запуска генерации
    if st.button("Сгенерировать"):
        st.info("Генерация началась...")
        GeneratePerson()
    
    
    
