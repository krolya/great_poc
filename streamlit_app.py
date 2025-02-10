"""
Расширенный код с комментариями и подгрузкой промтов из GitHub репозитория.
Теперь чтение данных из Airtable вынесено в отдельную функцию.
"""

import streamlit as st
import json
import datetime
import requests  # добавляем requests для работы с GitHub Raw

from openai import OpenAI
from pyairtable import Api

# Глобальные переменные.
# Хранятся настройки и параметры по умолчанию, которые управляют процессом генерации и анализа.
ad_description = ""
free_question = ""
message = ""
tags = ""
children_age = (0, 18)
children_count = (0, 3)
marital_selected = ["В браке", "Разведен(-а)", "В отношениях", "Одинок (-a)"]
city_size_selected = ["До 100 0000 человек", "От 100 000 до 500 000", "От 500 000 до 1 000 000", "Свыше 1 000 000"]
selected_regions = ["Москва", "Московская область"]
education_selected = ["Среднее", "Неоконченное высшее", "Высшее"]
income_selected = ["Низкий", "Низкий плюс"," Средний", "Средний плюс","Высокий","Высокий плюс"]
age_range = (18, 60)
gender_ratio = 50
model_name = "gpt-4o"
generation_id = ""
number_of_persons = 20
number_of_persons_analysis = 20


def get_file_from_github(file_path: str) -> str:
    """
    Функция, которая выгружает содержимое файла из GitHub репозитория в виде текста.
    Использует библиотеку requests и секретный токен GitHub (st.secrets.GITHUB_API_TOKEN).

    :param file_path: относительный путь к файлу в репозитории (например, "person_generation.promt")
    :return: текстовое содержимое файла
    """
    # Сформируем URL-адрес для raw-контента
    url = f"https://raw.githubusercontent.com/krolya/great_poc/main/{file_path}"
    
    # Добавим заголовок с токеном
    headers = {"Authorization": f"Bearer {st.secrets.GITHUB_API_TOKEN}"}

    response = requests.get(url, headers=headers)
    response.raise_for_status()  # выбросит исключение, если не 2xx
    return response.text


def openai_chat(system_prompt: str, user_prompt: str) -> str:
    """
    Обертка для обращения к OpenAI API.

    Принимает два параметра:
     - system_prompt: текст, отправляемый как системное сообщение (role="system")
     - user_prompt: текст, отправляемый как пользовательское сообщение (role="user")
    Возвращает строковый ответ от модели, предполагается что это JSON.
    """
    global model_name

    # В зависимости от имени модели, выбираем разные настройки клиента
    if "deepseek" not in model_name.lower():
        client = OpenAI(
            api_key=st.secrets.OPENAI_API_KEY,
        )
    else:
        client = OpenAI(
            base_url="https://api.studio.nebius.ai/v1/",
            api_key=st.secrets.NEBIUS_API_KEY,
        )

    # Формируем список сообщений: system + user
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Делаем запрос к модели
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format={"type": "json_object"}
    )

    # Если включен режим отладки, выведем ответ модели
    if st.session_state.debug:
        st.info("Что вернул OpenAI:")
        st.write(completion.choices[0].message.content)

    # Возвращаем содержимое первого варианта
    return completion.choices[0].message.content


def upload_to_airtable(data, table_name="Personas") -> int:
    """
    Загрузка сгенерированных данных в Airtable.

    :param data: JSON-строка (предположительно, от модели)
    :param table_name: имя таблицы в Airtable
    :return: кол-во записей, успешно загруженных
    """
    api = Api(st.secrets.AIRTABLE_API_TOKEN)
    table = api.table(st.secrets.AIRTABLE_BASE_ID, table_name)
    st.info("Загружаем данные в Airtable...")

    # Преобразуем строку JSON в python-объект
    records = json.loads(data)
    # Batch create - заливаем все записи одним махом
    response = table.batch_create(records["records"])

    if st.session_state.debug:
        st.write(response)

    return len(response)


def read_from_airtable(table_id: str, page_size: int = 20, max_records: int = 1000):
    """
    Функция для чтения данных из указанной таблицы Airtable.
    Возвращает генератор (yield) групп записей.

    :param table_id: ID (или имя) таблицы в Airtable
    :param page_size: сколько записей читать за один вызов
    :param max_records: максимальное число записей
    :return: генератор, который по очереди возвращает блоки (списки) записей.
    """
    api = Api(st.secrets.AIRTABLE_API_TOKEN)
    table = api.table(st.secrets.AIRTABLE_BASE_ID, table_id)
    yield from table.iterate(page_size=page_size, max_records=max_records)


def generate_person():
    """
    Функция, генерирующая персонажей:
    1) Подгружает person_generation.promt из репозитория.
    2) Формирует system_prompt и user_prompt.
    3) Вызывает openai_chat.
    4) Загружает результат в Airtable.
    """
    global generation_id
    generation_id = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

     # system_prompt задает общий контекст
    system_prompt = get_file_from_github("person_generation_system.promt")

    # Подгружаем шаблон промта для генерации персон
    person_prompt_template = get_file_from_github("person_generation.promt")

    # Заменяем плейсхолдеры
    user_prompt = (person_prompt_template
        .replace("{number_of_persons}", str(number_of_persons))
        .replace("{gender_ratio}", str(gender_ratio))
        .replace("{age_min}", str(age_range[0]))
        .replace("{age_max}", str(age_range[1]))
        .replace("{income_selected}", str(income_selected))
        .replace("{education_selected}", str(education_selected))
        .replace("{selected_regions}", str(selected_regions))
        .replace("{city_size_selected}", str(city_size_selected))
        .replace("{marital_selected}", str(marital_selected))
        .replace("{children_min}", str(children_count[0]))
        .replace("{children_max}", str(children_count[1]))
        .replace("{children_age_min}", str(children_age[0]))
        .replace("{children_age_max}", str(children_age[1]))
        .replace("{generation_id}", generation_id)
        .replace("{model_name}", model_name)
    )

    if st.session_state.debug:
        st.info("Преобразованный пользовательский промт (user_prompt):")
        st.write(user_prompt)

    # Генерация
    with st.spinner("Генерация персонажей..."):
        generated_data = openai_chat(system_prompt, user_prompt)
        if st.session_state.debug:
            st.info("Generated data:")
            st.write(generated_data)

    st.success("Персонажи успешно сгенерированы!")

    # Загрузка в Airtable
    with st.spinner("Загрузка в Airtable..."):
        uploaded_count = upload_to_airtable(generated_data)

    st.success(f"Успешно загружено {uploaded_count} записей в Airtable!")


def analyze_ad():
    """
    Функция для аналитики рекламы:
    1) Подгружает ad_analysis.promt из GitHub.
    2) Использует read_from_airtable, чтобы получить записи персон.
    3) Для каждой записи формирует user_prompt + system_prompt.
    4) Результат сохраняется в таблице Responses.
    """
    st.write("Анализ рекламы")
    response_test_id = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

    # Общий системный контекст
    system_prompt = get_file_from_github("ad_analysis_system.promt")

    # Подгружаем шаблон аналитики
    ad_analysis_template = get_file_from_github("ad_analysis.promt")

    # Запускаем чтение персон из Airtable
    with st.spinner("Генерация ответов..."):
        for records in read_from_airtable(
            table_id=st.secrets.AIRTABLE_TABLE_ID,
            page_size=number_of_persons_analysis,
            max_records=1000
        ):
            for record in records:

                if st.session_state.debug:
                    st.info("Record:")
                    st.write(record)

                # Достанем нужные поля
                description = record["fields"].get("Description", "")
                name = record["fields"].get("Name", "")
                age = record["fields"].get("Age", "")
                region = record["fields"].get("Region", "")
                city_size = record["fields"].get("City size", "")
                children = record["fields"].get("Children", 0)
                income = record["fields"].get("Income", "")
                marital_status = record["fields"].get("Marital status", "")
                education = record["fields"].get("Education", "")
                children_age_1 = record["fields"].get("Children age 1", 0)
                children_age_2 = record["fields"].get("Children age 2", 0)
                children_age_3 = record["fields"].get("Children age 3", 0)
                children_age_4 = record["fields"].get("Children age 4", 0)
                children_age_5 = record["fields"].get("Children age 5", 0)
                record_id = record.get("id", "")

                # Формируем user_prompt на основе шаблона
                user_prompt = (
                    ad_analysis_template
                    .replace("{description}", description)
                    .replace("{name}", name)
                    .replace("{age}", str(age))
                    .replace("{region}", region)
                    .replace("{city_size}", city_size)
                    .replace("{children}", str(children))
                    .replace("{income}", income)
                    .replace("{marital_status}", marital_status)
                    .replace("{education}", education)
                    .replace("{children_age_1}", str(children_age_1))
                    .replace("{children_age_2}", str(children_age_2))
                    .replace("{children_age_3}", str(children_age_3))
                    .replace("{children_age_4}", str(children_age_4))
                    .replace("{children_age_5}", str(children_age_5))
                    .replace("{record_id}", record_id)
                    .replace("{response_test_id}", response_test_id)
                    .replace("{ad_description}", ad_description)
                    .replace("{message}", message)
                    .replace("{free_question}", free_question)
                )

                if st.session_state.debug:
                    st.info("Сформированный user_prompt для анализа рекламы")
                    st.write(user_prompt)

                # Запрос к модели
                generated_data = openai_chat(system_prompt, user_prompt)

                if st.session_state.debug:
                    st.info("Generated data:")
                    st.write(generated_data)

                # Сохраняем в таблицу "Responses"
                upload_to_airtable(generated_data, "Responses")

    st.success("Анализ успешно завершен!")


def show_generation_tab():
    """
    Отрисовывает на правой колонке вкладки "Генерация персон" кнопку для запуска генерации
    и селектор выбора модели.
    """
    global number_of_persons, gender_ratio, age_range, income_selected, education_selected, selected_regions
    global city_size_selected, marital_selected, children_count, children_age, tags, model_name

    st.header("Генерация персон")

    # Выбор модели
    model_name = st.selectbox(
        "Выберите модель",
        [
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1",
            "meta-llama/Llama-3.3-70B-Instruct",
            "gpt-4o",
            "o1",
            "o1-mini"
        ],
        index=3
    )

    if st.button("Сгенерировать"):
        st.info("Генерация началась...")
        generate_person()


def show_analysis_tab():
    """
    Отрисовывает на правой колонке вкладки "Аналитика" поля для ввода описания рекламы,
    целевого сообщения, свободного вопроса и кнопку запуска анализа.
    """
    global number_of_persons_analysis, ad_description, message, free_question

    st.header("Анализ рекламы")

    # Поля для ввода описания рекламы, ключевого сообщения и свободного вопроса
    ad_description = st.text_input("Описание рекламы", placeholder="Введите максимально полное описание рекламы")
    message = st.text_input("Целевое сообщение рекламы", placeholder="Введите основной месседж для проверки")
    free_question = st.text_input(
        "Введите свободный вопрос", placeholder="Введите свободный вопрос, который вы хотите задать персоне"
    )

    if st.button("Анализировать"):
        st.info("Анализ начался...")
        analyze_ad()


def show_filters_tab_generation():
    """
    Отрисовывает на левой колонке вкладки "Генерация персон" фильтры для настройки целевой аудитории.
    """
    global number_of_persons, gender_ratio, age_range, income_selected, education_selected
    global selected_regions, city_size_selected, marital_selected, children_count, children_age, tags

    st.header("Целевая аудитория")

    with st.expander("Основные настройки", expanded=True):
        number_of_persons = st.slider("Количество персон для генерации", min_value=0, max_value=100, value=20)
        gender_ratio = st.slider("Процент мужчин в выборке (%)", min_value=0, max_value=100, value=50)
        age_range = st.slider("Возраст", min_value=4, max_value=100, value=(18, 60))
        income_options = ["Низкий", "Низкий плюс"," Средний", "Средний плюс","Высокий","Высокий плюс"]
        income_selected = st.multiselect("Выберите группу доходов", options=income_options, default=income_options)

    with st.expander("Образование", expanded=True):
        education_options = ["Среднее", "Неоконченное высшее", "Высшее"]
        education_selected = st.multiselect(
            "Выберите образование", options=education_options, default=education_options
        )

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
        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button("Выбрать все", key="select_all_regions"):
            for region in all_regions:
                st.session_state[f"region_{region}"] = True
        if col_btn2.button("Снять все", key="deselect_all_regions"):
            for region in all_regions:
                st.session_state[f"region_{region}"] = False

        selected_regions = []
        for region in all_regions:
            default = True if region in ["Москва", "Московская область"] else False
            checked = st.checkbox(
                region,
                value=st.session_state.get(f"region_{region}", default),
                key=f"region_{region}"
            )
            if checked:
                selected_regions.append(region)

        st.markdown("#### Размер населенного пункта")
        city_size_options = [
            "До 100 0000 человек",
            "От 100 000 до 500 000",
            "От 500 000 до 1 000 000",
            "Свыше 1 000 000"
        ]
        city_size_selected = st.multiselect(
            "Выберите размер населенного пункта", 
            options=city_size_options,
            default=city_size_options
        )

    with st.expander("Семейное положение", expanded=True):
        marital_options = ["В браке", "Разведен(-а)", "В отношениях", "Одинок (-a)"]
        marital_selected = st.multiselect(
            "Выберите семейное положение", options=marital_options, default=marital_options
        )
        children_count = st.slider("Количество детей", min_value=0, max_value=5, value=(0, 3))
        children_age = st.slider("Возраст детей", min_value=0, max_value=18, value=(0, 18))

    tags = st.text_input("Тэги", placeholder="Введите тэги через запятую")


def show_filters_tab_analysis():
    """
    Отрисовывает на левой колонке вкладки "Аналитика" фильтры для анализа рекламы.
    """
    global number_of_persons_analysis

    st.header("Фильтры")
    with st.expander("Основные настройки", expanded=True):
        number_of_persons_analysis = st.slider(
            "Количество персон для анализа", min_value=0, max_value=100, value=20
        )


def main():
    """
    Основная функция:
    1) Настраивает страницу,
    2) Создает вкладки ("Генерация персон", "Аналитика", "Настройки"),
    3) В каждой вкладке отображает две колонки: слева - фильтры, справа - основная кнопка/форма.
    """

    # Настройка страницы
    st.set_page_config(page_title="Более нормальный человек", layout="wide")

    if "debug" not in st.session_state:
        st.session_state.debug = False

    tab1, tab2, tab3 = st.tabs(["Генерация персон", "Аналитика", "Настройки"])

    with tab1:
        col_left, col_right = st.columns([3, 7])
        with col_left:
            show_filters_tab_generation()
        with col_right:
            show_generation_tab()

    with tab2:
        col_left, col_right = st.columns([3, 7])
        with col_left:
            show_filters_tab_analysis()
        with col_right:
            show_analysis_tab()

    with tab3:
        st.checkbox("Выводить отладочную информацию", key="debug")


if __name__ == "__main__":
    main()
