import streamlit as st
import json
import datetime

from openai import OpenAI
from pyairtable import Api
#from github import Github


#глобальные переменные
ad_description = ""
free_question = ""
message = ""
tags = ""  
children_age = (0, 18)
children_count = (0, 3)
marital_selected = ["В браке", "Разведен(-а)", "В отношениях", "Одинок (-а)"]
city_size_selected = ["До 100 0000 человек", "От 100 000 до 500 000", "От 500 000 до 1 000 000", "Свыше 1 000 000"]
selected_regions = ["Москва", "Московская область"]
education_selected = ["Среднее", "Неоконченное высшее", "Высшее"]
income_selected = ["Низкий", "Низкий плюс"," Средний", "Средний плюс","Высокий","Высокий плюс"]
age_range = (18, 60)
gender_ratio = 50
model_name = "deepseek-ai/DeepSeek-V3"
debug = False
generation_id = ""

#функции
def OpenAIChat(promt):

    if "deepseek" not in model_name.lower():
        client = OpenAI(
            api_key=st.secrets.OPENAI_API_KEY,
        )
    else:
        client = OpenAI(
            base_url="https://api.studio.nebius.ai/v1/",
            api_key=st.secrets.NEBIUS_API_KEY,
        )

    #st.write(model_name)
    #st.info("Запускаем чат...")

    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role":"user","content": promt}],
        response_format={"type": "json_object"}
    )
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
   
    return completion.choices[0].message.content

def upload_to_airtable(data):
    api = Api(st.secrets.AIRTABLE_API_TOKEN)
    table = api.table(st.secrets.AIRTABLE_BASE_ID, st.secrets.AIRTABLE_TABLE_ID)
    
    #records = [{"fields": person} for person in data["records"]]]
    st.info("Загружаем данные в Airtable...")
    #st.write(data)
    records = json.loads(data)
    #st.write(records)
    #st.write(len(records))
    #for person in records["records"]:
    #    st.write(person)
    #    response = table.create(person)
    #    st.write(response.json())

    response = table.batch_create(records["records"])
    if st.session_state.debug: st.write(response)
    return len(response)

def GeneratePerson():

    generation_id = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    prompt = f"""
    Ты специальный сервис по созданию персонажей. Твоя задача сгенерировать JSON-объект с случайными персонажами по следующим правилам:

    Количество персонажей: {number_of_persons}
    Процент мужчин в выборке должно быть следующим: {gender_ratio}% 
    Имя персонажа: случайное имя в зависимости от пола
    Возраст: от {age_range[0]} до {age_range[1]} лет, если есть дети, то всегда >= 18 лет
    Доход: случайный доход из списка {income_selected}
    Образование: случайное образование из списка {education_selected}
    Регион проживания: случайный регион из списка {selected_regions}
    Размер населенного пункта: случайный размер из списка {city_size_selected}, работает только для областей, если выбраны Москва или Санкт-Петербург, то всегда "Свыше 1 000 000"
    Семейное положение: случайное образование из списка {marital_selected}
    Количество детей: от {children_count[0]} до {children_count[1]}
    Возраст детей: от {children_age[0]} до {children_age[1]}
    
    Верни структурированный JSON в формате и только его без каких либо пояснений или чего-либо еще в ответе по следующему шаблону, напротив каждой строки будет дано пояснение после знака#:
    
    {{
        "records":[ #эта строка будет всегда присутствовать, независимости от количества персонажей
        {{
            "Name": "Иван Петров", #случайное имя в зависимости от пола
            "Gender": "Мужской", #случайный пол
            "Marital status": "В браке", #случайное семейное положение из списка {marital_selected}, очень важно, чтобы значение было ровно как в списке, т.е. "Одинок (-а)", а не "Одинок" или "Одинока"
            "Income": "Средний", #случайный доход из списка {income_selected}
            "Age": 28, #случайный возраст от {age_range[0]} до {age_range[1]} лет
            "Children": 0, #случайное количество детей от {children_count[0]} до {children_count[1]}
            "Children age 1": 0 #случайный возраст ребенка от {children_age[0]} до {children_age[1]}, если количество детей = 1
            "Children age 2": 0 #случайный возраст ребенка от {children_age[0]} до {children_age[1]}, если количество детей = 2 
            "Children age 3": 0 #случайный возраст ребенка от {children_age[0]} до {children_age[1]}, если количество детей = 3
            "Children age 4": 0 #случайный возраст ребенка от {children_age[0]} до {children_age[1]}, если количество детей = 4
            "Children age 5": 0 #случайный возраст ребенка от {children_age[0]} до {children_age[1]}, если количество детей = 5
            "Region": "Москва", #случайный регион из списка {selected_regions}
            "City size": "Свыше 1 000 000", #случайный размер из списка {city_size_selected}
            "Education": "Среднее", #случайное образование из списка {education_selected}
            "Generation ID": "{generation_id}", #уникальный идентификатор генерации
            "Generation model": "{model_name}", #модель генерации
            "Description": "Здесь нужно дать полное описание персоны с учетом всех параметров выше и расширь его каким-то дополнительным описанием", #полное описание персонажа
        }},
        {{
        #описание следующего персонажа, может быть сколько угодно пока не равно количеству персонажей
        }}
    }}
    """

    # GitHub API не работает в Streamlit
    # Подключаемся к GitHub
    #g = Github(st.secrets.GITHUB_API_TOKEN)

    # Получаем репозиторий
    #repo = g.get_repo("krolya/great_poc")

    # Получаем файл из репозитория
    #prompt = repo.get_contents("person_generation.promt").decoded_content.decode("utf-8")

    # Заменяем переменные в шаблоне ОЧЕНЬ ОЧЕНЬ НЕБЕЗОПАСНЫМ способом
    #formatted_prompt = eval(f'f"""{prompt}"""')


    print("Содержимое файла:", file_content)
    if st.session_state.debug: st.write(prompt)

    with st.spinner("Генерация персонажей..."):
        #st.write(generation_id)
        generated_data = OpenAIChat(prompt)
        if st.session_state.debug: st.write(generated_data)
    
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

# Создаем три вкладки
tab1, tab2, tab3 = st.tabs(["Генерация персон", "Аналитика", "Настройки"])

with tab1:

    # 1. Заголовок
    # Разбиваем остальную область на две колонки с пропорцией 30:70
    col_left, col_right = st.columns([3, 7])

    # Левая колонка: "Целевая аудитория" и фильтры
    with col_left:
        st.header("Целевая аудитория")

        with st.expander("Основные настройки", expanded=True):

            # 5.0. Слайдер для выбора количества персон для генерации
            number_of_persons = st.slider("Количество персон для генерации", min_value=0, max_value=100, value=20)

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
                "От 500 000 до 1 000 000",
                "Свыше 1 000 000"
            ]
            city_size_selected = st.multiselect("Выберите размер населенного пункта", options=city_size_options,
                                                default=city_size_options)

        with st.expander("Семейное положение", expanded=True):

            # 5.8. Фильтр «Семейное положение»
            marital_options = ["В браке", "Разведен(-а)", "В отношениях", "Одинок (-а)"]
            marital_selected = st.multiselect("Выберите семейное положение", options=marital_options,
                                            default=marital_options)
            # 5.6. Двойной слайдер для выбора диапазона количества детей
            children_count = st.slider("Количество детей", min_value=0, max_value=5, value=(0, 3))

            # 5.7. Двойной слайдер для выбора диапазона возраста детей
            children_age = st.slider("Возраст детей", min_value=0, max_value=18, value=(0, 18))
    
        
        # 5.9. Поле для ввода тэгов
        tags = st.text_input("Тэги", placeholder="Введите тэги через запятую")

        

    # Правая колонка: "Генерация"
    with col_right:
        st.header("Генерация персон")

        #st.write("Здесь можно разместить настройки генерации или результаты.")

        

        # 5.12 Выберите модель
        model_name = st.selectbox("Выберите модель", ["deepseek-ai/DeepSeek-V3",
                                                "deepseek-ai/DeepSeek-R1",
                                                "meta-llama/Llama-3.3-70B-Instruct",
                                                "gpt-4o",
                                                "o1",
                                                "o1-mini"])

        # Например, можно добавить кнопку для запуска генерации
        if st.button("Сгенерировать"):
            st.info("Генерация началась...")
            GeneratePerson()

with tab2:
    col_left, col_right = st.columns([3, 7])
    with col_left:
        st.header("Фильтры")

        with st.expander("Основные настройки", expanded=True):

            # 5.0. Слайдер для выбора количества персон для генерации
            number_of_persons = st.slider("Количество персон для анализа", min_value=0, max_value=100, value=20)

            if st.button("Сгенерировать"):
                st.info("Генерация началась...")
                GeneratePerson()

    with col_right:
        st.header("Анализ рекламы")

        # 5.10. Поле для ввода сообщения для проверки
        ad_description = st.text_input("Описание рекламы", placeholder="Введите максимально полное описание рекламы")

        # 5.10. Поле для ввода сообщения для проверки
        message = st.text_input("Целевое сообщение рекламы", placeholder="Введите основной месседж для проверки")

        # 5.11. Поле для ввода сообщения для проверки
        free_question = st.text_input("Введите свободный вопрос", placeholder="Введите свободный вопрос, который вы хотите задать персоне")

with tab3:
    debug = st.checkbox("Выводить отладочную информацию", value=False, key="debug")

    
