"""
1) Вынесен запрос к Airtable (all_records) в отдельную функцию.
2) Добавлены фильтры по возрасту детей (children_age_1..5), чтобы соответствовать analysis_children_age.
"""

import streamlit as st
import json
import datetime
import requests

from openai import OpenAI
from pyairtable import Api
from pyairtable.formulas import AND, OR, EQ, GTE, LTE, Field

# -------------------
# Глобальные переменные.
# -------------------
ad_description = ""
free_question = ""
message = ""
tags = ""

# Фильтры (используются в генерации и анализе)
children_age = (0, 18)
children_count = (0, 3)
marital_selected = ["В браке", "Разведен(-а)", "В отношениях", "Одинок (-a)"]
city_size_selected = ["До 100 0000 человек", "От 100 000 до 500 000", "От 500 000 до 1 000 000", "Свыше 1 000 000"]
selected_regions = ["Москва", "Московская область"]
education_selected = ["Среднее", "Неоконченное высшее", "Высшее"]
income_selected = ["Низкий", "Низкий плюс"," Средний", "Средний плюс","Высокий","Высокий плюс"]
age_range = (18, 60)
gender_ratio = 50

# Модель и общее управление
model_name = "gpt-4o"
generation_id = ""

# Количество для генерации
number_of_persons = 20
# Количество для анализа
number_of_persons_analysis = 20

# Фильтры для анализа (дублируют логику генерации, могут отличаться по умолчанию)
analysis_age_range = (18, 60)
analysis_income_selected = ["Низкий", "Низкий плюс"," Средний", "Средний плюс","Высокий","Высокий плюс"]
analysis_education_selected = ["Среднее", "Неоконченное высшее", "Высшее"]
analysis_selected_regions = ["Москва", "Московская область"]
analysis_city_size_selected = ["До 100 0000 человек", "От 100 000 до 500 000", "От 500 000 до 1 000 000", "Свыше 1 000 000"]
analysis_marital_selected = ["В браке", "Разведен(-а)", "В отношениях", "Одинок (-a)"]
analysis_children_count = (0, 3)
analysis_children_age = (0, 18)


# --------------------------------------------------
# Функции для GitHub, OpenAI, Airtable
# --------------------------------------------------

def get_file_from_github(file_path: str) -> str:
    """
    Выгружает содержимое файла из GitHub (Raw) по токену.
    """
    url = f"https://raw.githubusercontent.com/krolya/great_poc/main/{file_path}"
    headers = {"Authorization": f"Bearer {st.secrets.GITHUB_API_TOKEN}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text


def openai_chat(system_prompt: str, user_prompt: str) -> str:
    """
    Обертка для обращения к OpenAI API (2 сообщения: system + user)
    """
    global model_name

    if "deepseek" not in model_name.lower():
        client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)
    else:
        client = OpenAI(
            base_url="https://api.studio.nebius.ai/v1/",
            api_key=st.secrets.NEBIUS_API_KEY,
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format={"type": "json_object"}
    )

    if st.session_state.debug:
        st.info("Что вернул OpenAI:")
        st.write(completion.choices[0].message.content)

    return completion.choices[0].message.content


def upload_to_airtable(data, table_name="Personas") -> int:
    """
    Загрузка (JSON-строка) в Airtable
    """
    api = Api(st.secrets.AIRTABLE_API_TOKEN)
    table = api.table(st.secrets.AIRTABLE_BASE_ID, table_name)
    st.info("Загружаем данные в Airtable...")

    records = json.loads(data)
    response = table.batch_create(records["records"])

    if st.session_state.debug:
        st.write(response)

    return len(response)


def fetch_analysis_records(formula: str, page_size=100, max_records=1000):
    """
    Функция для чтения записей из Airtable (таблица из настроек) с учётом formula.
    Возвращает список (не генератор).
    """
    api = Api(st.secrets.AIRTABLE_API_TOKEN)
    table = api.table(st.secrets.AIRTABLE_BASE_ID, st.secrets.AIRTABLE_TABLE_ID)
    # table.all(...) вернёт список (все записи)
    all_records = table.all(
        page_size=page_size,
        max_records=max_records,
        formula=formula
    )
    return all_records

# --------------------------------------------------
# Построение FORMULA для анализа
# --------------------------------------------------

def build_analysis_formula() -> str:
    """
    Формирует Airtable Formula, учитывая:
    - Возраст (analysis_age_range)
    - Доход (analysis_income_selected)
    - Образование (analysis_education_selected)
    - Регион (analysis_selected_regions)
    - Размер города (analysis_city_size_selected)
    - Семейное положение (analysis_marital_selected)
    - Кол-во детей (analysis_children_count)
    - Возраст детей (analysis_children_age) -- проверим поля Children age 1..5.
    """
    conds = []

    # 1) Возраст: Age >= X AND Age <= Y
    conds.append(GTE(Field("Age"), analysis_age_range[0]))
    conds.append(LTE(Field("Age"), analysis_age_range[1]))

    # 2) Income (OR)
    if analysis_income_selected:
        sub_conds = []
        for inc_val in analysis_income_selected:
            sub_conds.append(EQ(Field("Income"), inc_val))
        conds.append(OR(*sub_conds))

    # 3) Education (OR)
    if analysis_education_selected:
        sub_conds = []
        for edu_val in analysis_education_selected:
            sub_conds.append(EQ(Field("Education"), edu_val))
        conds.append(OR(*sub_conds))

    # 4) Region (OR)
    if analysis_selected_regions:
        sub_conds = []
        for reg_val in analysis_selected_regions:
            sub_conds.append(EQ(Field("Region"), reg_val))
        conds.append(OR(*sub_conds))

    # 5) City size (OR)
    if analysis_city_size_selected:
        sub_conds = []
        for city_val in analysis_city_size_selected:
            sub_conds.append(EQ(Field("City size"), city_val))
        conds.append(OR(*sub_conds))

    # 6) Marital status (OR)
    if analysis_marital_selected:
        sub_conds = []
        for m_val in analysis_marital_selected:
            sub_conds.append(EQ(Field("Marital status"), m_val))
        conds.append(OR(*sub_conds))

    # 7) Children count
    conds.append(GTE(Field("Children"), analysis_children_count[0]))
    conds.append(LTE(Field("Children"), analysis_children_count[1]))

    # 8) Возраст детей
    # Предположим, нужно чтобы все поля Children age i (1..5), если > 0,
    # укладывались в analysis_children_age.
    # Т.е. for each i in 1..5: if field > 0 -> field in [min, max].

    for i in range(1, 6):
        field_name = f"Children age {i}"
        # Логика: если значение > 0, хотим age_min <= значение <= age_max.
        # Cформируем AND( > 0, >= min, <= max ).
        # Но чтобы объединить для всех, делаем conds.append(...)

        # AND( OR(EQ(Field,0), (Field >= x AND Field <= y)) )
        # => если 0 (нет ребёнка), это не дисквалифицирует.
        # Иначе проверяем диапазон.

        child_in_range = AND(
            GTE(Field(field_name), analysis_children_age[0]),
            LTE(Field(field_name), analysis_children_age[1])
        )
        # Если =0, пропускаем. Если >0, проверяем child_in_range.
        # => OR(EQ(Field, 0), child_in_range)
        conds.append(
            OR(
                EQ(Field(field_name), 0),
                child_in_range
            )
        )

    formula_obj = AND(*conds)
    return str(formula_obj)

# -------------------
# Логика генерации
# -------------------
def generate_person():
    global generation_id
    generation_id = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

    # Шаблоны
    system_prompt = get_file_from_github("person_generation_system.promt")
    person_prompt_template = get_file_from_github("person_generation.promt")

    user_prompt = (
        person_prompt_template
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

    with st.spinner("Генерация персонажей..."):
        generated_data = openai_chat(system_prompt, user_prompt)
        if st.session_state.debug:
            st.info("Generated data:")
            st.write(generated_data)

    st.success("Персонажи успешно сгенерированы!")

    with st.spinner("Загрузка в Airtable..."):
        uploaded_count = upload_to_airtable(generated_data)

    st.success(f"Успешно загружено {uploaded_count} записей в Airtable!")


# -------------------
# Логика анализа
# -------------------
def analyze_ad():
    st.write("Анализ рекламы")
    response_test_id = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

    system_prompt = get_file_from_github("ad_analysis_system.promt")
    ad_analysis_template = get_file_from_github("ad_analysis.promt")

    # Считываем файлы (макс 10)
    uploaded_files = st.session_state.get("analysis_uploaded_files", [])
    files_text = ""
    for fdict in uploaded_files:
        filename = fdict["filename"]
        content = fdict["content"]
        files_text += f"\n---\nФайл: {filename}\nСодержимое:\n{content}\n"

    # Построим формулу и загрузим записи через отдельную функцию
    formula = build_analysis_formula()
    all_records = fetch_analysis_records(formula, page_size=100, max_records=1000)

    analyzed_count = 0

    with st.spinner("Генерация ответов..."):
        for record in all_records:
            if analyzed_count >= number_of_persons_analysis:
                break

            analyzed_count += 1
            if st.session_state.debug:
                st.info(f"Analyzing record #{analyzed_count}:")
                st.write(record)

            rfields = record["fields"]
            description = rfields.get("Description", "")
            name = rfields.get("Name", "")
            age = rfields.get("Age", 0)
            region = rfields.get("Region", "")
            city_size = rfields.get("City size", "")
            children = rfields.get("Children", 0)
            income = rfields.get("Income", "")
            marital_status = rfields.get("Marital status", "")
            education = rfields.get("Education", "")
            children_age_1 = rfields.get("Children age 1", 0)
            children_age_2 = rfields.get("Children age 2", 0)
            children_age_3 = rfields.get("Children age 3", 0)
            children_age_4 = rfields.get("Children age 4", 0)
            children_age_5 = rfields.get("Children age 5", 0)
            record_id = record.get("id", "")

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

            # Добавим файлы
            if files_text:
                user_prompt += f"\n\nДополнительные файлы:{files_text}"

            if st.session_state.debug:
                st.info("Сформированный user_prompt для анализа рекламы")
                st.write(user_prompt)

            # Запрос
            generated_data = openai_chat(system_prompt, user_prompt)

            if st.session_state.debug:
                st.info("Generated data:")
                st.write(generated_data)

            # Сохраняем
            upload_to_airtable(generated_data, "Responses")

    st.success("Анализ успешно завершен!")


# -------------------
# Отрисовка вкладок
# -------------------
def show_generation_tab():
    global number_of_persons, gender_ratio, age_range, income_selected, education_selected
    global selected_regions, city_size_selected, marital_selected, children_count, children_age, tags, model_name

    st.header("Генерация персон")

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
    global number_of_persons_analysis, ad_description, message, free_question

    st.header("Анализ рекламы")

    ad_description = st.text_input("Описание рекламы", placeholder="Введите максимально полное описание рекламы")
    message = st.text_input("Целевое сообщение рекламы", placeholder="Введите основной месседж для проверки")
    free_question = st.text_input(
        "Введите свободный вопрос", placeholder="Введите свободный вопрос, который вы хотите задать персоне"
    )

    uploaded_files = st.file_uploader("Добавить до 10 файлов", accept_multiple_files=True)

    final_files = []
    if uploaded_files:
        for i, f in enumerate(uploaded_files):
            if i >= 10:
                break
            try:
                content = f.read().decode("utf-8")
            except:
                content = "Не удалось декодировать файл в UTF-8"
            final_files.append({"filename": f.name, "content": content})

    st.session_state["analysis_uploaded_files"] = final_files

    if st.button("Анализировать"):
        st.info("Анализ начался...")
        analyze_ad()


def show_filters_tab_generation():
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
        if col_btn1.button("Выбрать все", key="select_all_regions_gen"):
            for region in all_regions:
                st.session_state[f"region_{region}"] = True
        if col_btn2.button("Снять все", key="deselect_all_regions_gen"):
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
    global number_of_persons_analysis
    global analysis_age_range, analysis_income_selected, analysis_education_selected
    global analysis_selected_regions, analysis_city_size_selected, analysis_marital_selected
    global analysis_children_count, analysis_children_age

    st.header("Фильтры")
    with st.expander("Основные настройки", expanded=True):
        number_of_persons_analysis = st.slider(
            "Количество персон для анализа", min_value=0, max_value=100, value=20
        )

    with st.expander("Настройки фильтров (как при генерации)", expanded=True):
        analysis_age_range = st.slider("Возраст", min_value=4, max_value=100, value=(18, 60))
        income_options = ["Низкий", "Низкий плюс"," Средний", "Средний плюс","Высокий","Высокий плюс"]
        analysis_income_selected = st.multiselect("Доход", options=income_options, default=income_options)

        education_options = ["Среднее", "Неоконченное высшее", "Высшее"]
        analysis_education_selected = st.multiselect(
            "Образование", options=education_options, default=education_options
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

        st.markdown("##### Регион")
        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button("Выбрать все", key="select_all_regions_analysis"):
            for region in all_regions:
                st.session_state[f"analysis_region_{region}"] = True
        if col_btn2.button("Снять все", key="deselect_all_regions_analysis"):
            for region in all_regions:
                st.session_state[f"analysis_region_{region}"] = False

        local_selected_regions = []
        for region in all_regions:
            default = True if region in ["Москва", "Московская область"] else False
            checked = st.checkbox(
                region,
                value=st.session_state.get(f"analysis_region_{region}", default),
                key=f"analysis_region_{region}"
            )
            if checked:
                local_selected_regions.append(region)
        analysis_selected_regions = local_selected_regions

        st.markdown("##### Размер населенного пункта")
        city_size_options = [
            "До 100 0000 человек",
            "От 100 000 до 500 000",
            "От 500 000 до 1 000 000",
            "Свыше 1 000 000"
        ]
        analysis_city_size_selected = st.multiselect(
            "Размер города",
            options=city_size_options,
            default=city_size_options
        )

        st.markdown("##### Семейное положение")
        marital_options = ["В браке", "Разведен(-а)", "В отношениях", "Одинок (-a)"]
        analysis_marital_selected = st.multiselect(
            "Семейное положение", options=marital_options, default=marital_options
        )
        analysis_children_count = st.slider("Количество детей", min_value=0, max_value=5, value=(0, 3))
        analysis_children_age = st.slider("Возраст детей", min_value=0, max_value=18, value=(0, 18))


# -------------------
# Основная функция
# -------------------
def main():
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
