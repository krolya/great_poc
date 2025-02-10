"""
Добавляем возможность редактировать system/user промты прямо на вкладках (генерация, анализ),
если включен флаг отладки (debug). Тогда текст загружается из GitHub, заполняется в поля,
и берётся при генерации/анализе из этих текстовых полей.
"""

import streamlit as st
import json
import datetime
import requests

from openai import OpenAI
from pyairtable import Api
from pyairtable.formulas import AND, OR, EQ, GTE, LTE, Field

# -------------------
# Глобальные переменные
# -------------------
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

model_name = "gpt-4o"
generation_id = ""
number_of_persons = 20
number_of_persons_analysis = 20

analysis_age_range = (18, 60)
analysis_income_selected = ["Низкий", "Низкий плюс"," Средний", "Средний плюс","Высокий","Высокий плюс"]
analysis_education_selected = ["Среднее", "Неоконченное высшее", "Высшее"]
analysis_selected_regions = ["Москва", "Московская область"]
analysis_city_size_selected = ["До 100 0000 человек", "От 100 000 до 500 000", "От 500 000 до 1 000 000", "Свыше 1 000 000"]
analysis_marital_selected = ["В браке", "Разведен(-а)", "В отношениях", "Одинок (-а)"]
analysis_children_count = (0, 3)
analysis_children_age = (0, 18)

# -------------------
# Функция загрузки файла из GitHub
# -------------------
def get_file_from_github(file_path: str) -> str:
    url = f"https://raw.githubusercontent.com/krolya/great_poc/main/{file_path}"
    headers = {"Authorization": f"Bearer {st.secrets.GITHUB_API_TOKEN}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text

# -------------------
# Универсальная функция parse_prompt
# -------------------
def parse_prompt(text: str, placeholders: dict) -> str:
    for key, val in placeholders.items():
        text = text.replace(f"{{{key}}}", str(val))
    return text

# -------------------
# OpenAI / Airtable
# -------------------
def openai_chat(system_prompt: str, user_prompt: str) -> str:
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
    api = Api(st.secrets.AIRTABLE_API_TOKEN)
    table = api.table(st.secrets.AIRTABLE_BASE_ID, table_name)
    st.info("Загружаем данные в Airtable...")

    records = json.loads(data)
    response = table.batch_create(records["records"])

    if st.session_state.debug:
        st.write(response)

    return len(response)


def fetch_analysis_records(formula: str, page_size=100, max_records=1000):
    api = Api(st.secrets.AIRTABLE_API_TOKEN)
    table = api.table(st.secrets.AIRTABLE_BASE_ID, st.secrets.AIRTABLE_TABLE_ID)
    return table.all(
        page_size=page_size,
        max_records=max_records,
        formula=formula
    )

# -------------------
# Формула отбора персон
# -------------------
from pyairtable.formulas import AND, OR, EQ, GTE, LTE, Field

def build_analysis_formula() -> str:
    conds = []

    conds.append(GTE(Field("Age"), analysis_age_range[0]))
    conds.append(LTE(Field("Age"), analysis_age_range[1]))

    if analysis_income_selected:
        sub_conds = []
        for inc_val in analysis_income_selected:
            sub_conds.append(EQ(Field("Income"), inc_val))
        conds.append(OR(*sub_conds))

    if analysis_education_selected:
        sub_conds = []
        for edu_val in analysis_education_selected:
            sub_conds.append(EQ(Field("Education"), edu_val))
        conds.append(OR(*sub_conds))

    if analysis_selected_regions:
        sub_conds = []
        for reg_val in analysis_selected_regions:
            sub_conds.append(EQ(Field("Region"), reg_val))
        conds.append(OR(*sub_conds))

    if analysis_city_size_selected:
        sub_conds = []
        for city_val in analysis_city_size_selected:
            sub_conds.append(EQ(Field("City size"), city_val))
        conds.append(OR(*sub_conds))

    if analysis_marital_selected:
        sub_conds = []
        for m_val in analysis_marital_selected:
            sub_conds.append(EQ(Field("Marital status"), m_val))
        conds.append(OR(*sub_conds))

    conds.append(GTE(Field("Children"), analysis_children_count[0]))
    conds.append(LTE(Field("Children"), analysis_children_count[1]))

    for i in range(1, 6):
        field_name = f"Children age {i}"
        child_in_range = AND(
            GTE(Field(field_name), analysis_children_age[0]),
            LTE(Field(field_name), analysis_children_age[1])
        )
        conds.append(
            OR(
                EQ(Field(field_name), 0),
                child_in_range
            )
        )

    formula_obj = AND(*conds)
    return str(formula_obj)

# -------------------
# Генерация
# -------------------
def generate_person():
    global generation_id
    generation_id = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

    # Если debug, берём промты из text_area
    if st.session_state.debug:
        system_prompt_raw = st.session_state.get("gen_system_prompt", "")
        user_prompt_raw = st.session_state.get("gen_user_prompt", "")
    else:
        # Загружаем из GitHub
        system_prompt_raw = get_file_from_github("person_generation_system.promt")
        user_prompt_raw = get_file_from_github("person_generation.promt")

    gen_placeholders = {
        "model_name": model_name,
        "generation_id": generation_id,
        "number_of_persons": number_of_persons,
        "gender_ratio": gender_ratio,
        "age_min": age_range[0],
        "age_max": age_range[1],
        "income_selected": income_selected,
        "education_selected": education_selected,
        "selected_regions": selected_regions,
        "city_size_selected": city_size_selected,
        "marital_selected": marital_selected,
        "children_min": children_count[0],
        "children_max": children_count[1],
        "children_age_min": children_age[0],
        "children_age_max": children_age[1]
    }

    system_prompt = parse_prompt(system_prompt_raw, gen_placeholders)
    user_prompt = parse_prompt(user_prompt_raw, gen_placeholders)

    if st.session_state.debug:
        st.info("System prompt (генерация):")
        st.write(system_prompt)
        st.info("User prompt (генерация):")
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
# Анализ
# -------------------
def analyze_ad():
    st.write("Анализ рекламы")
    response_test_id = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

    if st.session_state.debug:
        system_prompt_raw = st.session_state.get("analysis_system_prompt", "")
        user_prompt_raw = st.session_state.get("analysis_user_prompt", "")
    else:
        system_prompt_raw = get_file_from_github("ad_analysis_system.promt")
        user_prompt_raw = get_file_from_github("ad_analysis.promt")

    analysis_static = {
        "model_name": model_name,
        "ad_description": ad_description,
        "message": message,
        "free_question": free_question
    }

    uploaded_files = st.session_state.get("analysis_uploaded_files", [])
    files_text = ""
    for fdict in uploaded_files:
        filename = fdict["filename"]
        content = fdict["content"]
        files_text += f"\n---\nФайл: {filename}\nСодержимое:\n{content}\n"

    formula = build_analysis_formula()
    all_records = fetch_analysis_records(formula, page_size=100, max_records=1000)

    analyzed_count = 0

    with st.spinner("Генерация ответов..."):
        for record in all_records:
            if analyzed_count >= number_of_persons_analysis:
                break

            analyzed_count += 1
            rfields = record["fields"]

            dynamic_part = {
                "response_test_id": response_test_id,
                "record_id": record.get("id", ""),
                "description": rfields.get("Description", ""),
                "name": rfields.get("Name", ""),
                "age": rfields.get("Age", 0),
                "region": rfields.get("Region", ""),
                "city_size": rfields.get("City size", ""),
                "children": rfields.get("Children", 0),
                "income": rfields.get("Income", ""),
                "marital_status": rfields.get("Marital status", ""),
                "education": rfields.get("Education", ""),
                "children_age_1": rfields.get("Children age 1", 0),
                "children_age_2": rfields.get("Children age 2", 0),
                "children_age_3": rfields.get("Children age 3", 0),
                "children_age_4": rfields.get("Children age 4", 0),
                "children_age_5": rfields.get("Children age 5", 0)
            }

            placeholders = {**analysis_static, **dynamic_part}

            system_prompt = parse_prompt(system_prompt_raw, placeholders)
            user_prompt = parse_prompt(user_prompt_raw, placeholders)

            if files_text:
                user_prompt += f"\n\nДополнительные файлы:{files_text}"

            if st.session_state.debug:
                st.info(f"Analyzing record #{analyzed_count}")
                st.write(record)
                st.info("System prompt (анализ):")
                st.write(system_prompt)
                st.info("User prompt (анализ):")
                st.write(user_prompt)

            generated_data = openai_chat(system_prompt, user_prompt)

            if st.session_state.debug:
                st.info("Generated data:")
                st.write(generated_data)

            upload_to_airtable(generated_data, "Responses")

    st.success("Анализ успешно завершен!")

# -------------------
# UI вкладок
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
        index=3,
        key="model_select_generation"
    )

    # Если debug, загружаем из GitHub и показываем текстовые поля
    if st.session_state.debug:
        st.subheader("Отладочные промты (генерация)")
        # грузим из GitHub
        system_prompt_raw = get_file_from_github("person_generation_system.promt")
        user_prompt_raw = get_file_from_github("person_generation.promt")
        # показываем в text_area
        st.session_state["gen_system_prompt"] = st.text_area(
            "System prompt (генерация)", 
            value=system_prompt_raw,
            key="gen_system_prompt_textarea"
        )
        st.session_state["gen_user_prompt"] = st.text_area(
            "User prompt (генерация)", 
            value=user_prompt_raw,
            key="gen_user_prompt_textarea"
        )

    if st.button("Сгенерировать", key="generate_button"):
        st.info("Генерация началась...")
        generate_person()


def show_analysis_tab():
    global number_of_persons_analysis, ad_description, message, free_question

    st.subheader("Отбор аудитории")
    if st.button("Отобрать персоны", key="select_persons_button"):
        formula = build_analysis_formula()
        records = fetch_analysis_records(formula, page_size=100, max_records=1000)
        data_for_table = []
        for r in records:
            fields = r["fields"]
            data_for_table.append({
                "ID": r.get("id", ""),
                "Имя": fields.get("Name", ""),
                "Возраст": fields.get("Age", 0),
                "Регион": fields.get("Region", ""),
                "Доход": fields.get("Income", ""),
                "Образование": fields.get("Education", ""),
                "Дети": fields.get("Children", 0)
            })
        st.write(f"Найдено {len(data_for_table)} персон:")
        st.dataframe(data_for_table)

    st.subheader("Анализ рекламы")

    ad_description = st.text_input("Описание рекламы", placeholder="Введите максимально полное описание рекламы", key="ad_description_input")
    message = st.text_input("Целевое сообщение рекламы", placeholder="Введите основной месседж для проверки", key="ad_message_input")
    free_question = st.text_input(
        "Введите свободный вопрос", placeholder="Введите свободный вопрос, который вы хотите задать персоне",
        key="ad_freeq_input"
    )

    # Если debug, загружаем из GitHub и показываем поля
    if st.session_state.debug:
        st.subheader("Отладочные промты (анализ)")
        system_prompt_raw = get_file_from_github("ad_analysis_system.promt")
        user_prompt_raw = get_file_from_github("ad_analysis.promt")
        st.session_state["analysis_system_prompt"] = st.text_area(
            "System prompt (анализ)", 
            value=system_prompt_raw,
            key="analysis_system_prompt_textarea"
        )
        st.session_state["analysis_user_prompt"] = st.text_area(
            "User prompt (анализ)", 
            value=user_prompt_raw,
            key="analysis_user_prompt_textarea"
        )

    uploaded_files = st.file_uploader("Добавить до 10 файлов", accept_multiple_files=True, key="analysis_uploader")

    final_files = []
    if uploaded_files:
        for i, f in enumerate(uploaded_files):
            if i >= 10:
                break
            content_bytes = f.read()
            import base64
            encoded = base64.b64encode(content_bytes).decode("utf-8")
            mime_type = f.type if f.type else "application/octet-stream"
            data_url = f"data:{mime_type};base64,{encoded}"
            final_files.append({"filename": f.name, "content": data_url})

    st.session_state["analysis_uploaded_files"] = final_files

    if st.button("Анализировать", key="analyze_button"):
        st.info("Анализ начался...")
        analyze_ad()



def show_filters_tab_generation():
    global number_of_persons, gender_ratio, age_range, income_selected, education_selected
    global selected_regions, city_size_selected, marital_selected, children_count, children_age, tags

    st.header("Целевая аудитория")

    with st.expander("Основные настройки", expanded=True):
        number_of_persons = st.slider(
            "Количество персон для генерации", 
            min_value=0, max_value=100, value=20,
            key="slider_num_persons_gen"
        )
        gender_ratio = st.slider(
            "Процент мужчин в выборке (%)", 
            min_value=0, max_value=100, value=50,
            key="slider_gender_ratio_gen"
        )
        age_range = st.slider(
            "Возраст", 
            min_value=4, max_value=100, value=(18, 60),
            key="slider_age_range_gen"
        )
        income_options = ["Низкий", "Низкий плюс"," Средний", "Средний плюс","Высокий","Высокий плюс"]
        income_selected = st.multiselect(
            "Выберите группу доходов", 
            options=income_options, 
            default=income_options,
            key="multiselect_income_gen"
        )

    with st.expander("Образование", expanded=True):
        education_options = ["Среднее", "Неоконченное высшее", "Высшее"]
        education_selected = st.multiselect(
            "Выберите образование", 
            options=education_options, 
            default=education_options,
            key="multiselect_edu_gen"
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

        temp_selected = []
        for region in all_regions:
            default = True if region in ["Москва", "Московская область"] else False
            checked = st.checkbox(
                region,
                value=st.session_state.get(f"region_{region}", default),
                key=f"checkbox_gen_{region}"
            )
            if checked:
                temp_selected.append(region)
        selected_regions = temp_selected

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
            default=city_size_options,
            key="multiselect_city_size_gen"
        )

    with st.expander("Семейное положение", expanded=True):
        marital_options = ["В браке", "Разведен(-а)", "В отношениях", "Одинок (-а)"]
        marital_selected = st.multiselect(
            "Выберите семейное положение", 
            options=marital_options, 
            default=marital_options,
            key="multiselect_marital_gen"
        )
        children_count = st.slider(
            "Количество детей", 
            min_value=0, max_value=5, value=(0, 3),
            key="slider_children_count_gen"
        )
        children_age = st.slider(
            "Возраст детей", 
            min_value=0, max_value=18, value=(0, 18),
            key="slider_children_age_gen"
        )

    tags = st.text_input("Тэги", placeholder="Введите тэги через запятую", key="tags_gen")


def show_filters_tab_analysis():
    global number_of_persons_analysis
    global analysis_age_range, analysis_income_selected, analysis_education_selected
    global analysis_selected_regions, analysis_city_size_selected, analysis_marital_selected
    global analysis_children_count, analysis_children_age

    st.header("Фильтры")
    with st.expander("Основные настройки", expanded=True):
        number_of_persons_analysis = st.slider(
            "Количество персон для анализа", 
            0, 100, 20, 
            key="slider_num_persons_analysis"
        )

    with st.expander("Настройки фильтров (как при генерации)", expanded=True):
        analysis_age_range = st.slider(
            "Возраст", 
            4, 100, (18, 60),
            key="slider_age_range_analysis"
        )
        income_options = ["Низкий", "Низкий плюс"," Средний", "Средний плюс","Высокий","Высокий плюс"]
        analysis_income_selected = st.multiselect(
            "Доход", 
            options=income_options, 
            default=income_options,
            key="multiselect_income_analysis"
        )

        education_options = ["Среднее", "Неоконченное высшее", "Высшее"]
        analysis_education_selected = st.multiselect(
            "Образование", 
            options=education_options, 
            default=education_options,
            key="multiselect_edu_analysis"
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
                key=f"checkbox_analysis_{region}"
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
            default=city_size_options,
            key="multiselect_city_size_analysis"
        )

        st.markdown("##### Семейное положение")
        marital_options = ["В браке", "Разведен(-а)", "В отношениях", "Одинок (-а)"]
        analysis_marital_selected = st.multiselect(
            "Семейное положение", 
            options=marital_options, 
            default=marital_options,
            key="multiselect_marital_analysis"
        )
        analysis_children_count = st.slider(
            "Количество детей", 
            0, 5, (0, 3),
            key="slider_children_count_analysis"
        )
        analysis_children_age = st.slider(
            "Возраст детей", 
            0, 18, (0, 18),
            key="slider_children_age_analysis"
        )



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
