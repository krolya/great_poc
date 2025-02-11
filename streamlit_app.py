import streamlit as st
import json
import datetime
import requests
import itertools

from openai import OpenAI
from pyairtable import Api
from pyairtable.formulas import AND, OR, EQ, GTE, LTE, Field

# -------------------
# Глобальные переменные
# -------------------
ad_name = ""  # добавляем новую переменную для Ad name
ad_description = ""
audio_description = ""
metadata_description = ""

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
analysis_gender_selected = ["Мужской", "Женский"]  # добавляем фильтр по полу

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
def openai_chat(system_prompt: str, user_prompt: str, file_messages=None) -> str:
    global model_name

    if "deepseek" not in model_name.lower():
        client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)
    else:
        client = OpenAI(
            base_url="https://api.studio.nebius.ai/v1/",
            api_key=st.secrets.NEBIUS_API_KEY,
        )

    # Формируем сообщения для запроса
    messages = [{"role": "system", "content": system_prompt}]

    if file_messages:
        # Формируем составное сообщение для пользователя: сначала текст, затем файлы
        user_content = [{"type": "text", "text": user_prompt}] + file_messages
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": user_prompt})

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

    # Добавляем условие по полу
    if analysis_gender_selected:
        sub_conds = []
        for g_val in analysis_gender_selected:
            sub_conds.append(EQ(Field("Gender"), g_val))
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
        "ad_name": ad_name,
        "ad_description": ad_description,
        "audio_description": audio_description,
        "metadata_description": metadata_description,
        "message": message,
        "free_question": free_question
    }

    uploaded_files = st.session_state.get("analysis_uploaded_files", [])
    file_messages = []
    for fdict in uploaded_files:
        file_messages.append({
            "type": "image_url",
            "image_url": {"url": fdict["content"]}
        })

    # Используем отобранные персоны из датафрейма/таблицы, сохранённые в session_state под ключом "selected_persons"
    persons = st.session_state.get("selected_persons", [])
    if not persons:
        st.error("Нет отобранных персон. Пожалуйста, отберите персоны сначала.")
        return

    analyzed_count = 0

    with st.spinner("Генерация ответов..."):
        for record in persons:
            if analyzed_count >= number_of_persons_analysis:
                break

            analyzed_count += 1
            # Предполагается, что record — это словарь с данными персоны, полученными ранее для отображения
            dynamic_part = {
                "response_test_id": response_test_id,
                "record_id": record.get("ID", ""),
                "description": record.get("Description", ""),
                "name": record.get("Name", ""),
                "age": record.get("Age", 0),
                "region": record.get("Region", ""),
                "city_size": record.get("City size", ""),
                "children": record.get("Children", 0),
                "income": record.get("Income", ""),
                "marital_status": record.get("Marital status", ""),
                "education": record.get("Education", ""),
                "children_age_1": record.get("Children age 1", 0),
                "children_age_2": record.get("Children age 2", 0),
                "children_age_3": record.get("Children age 3", 0),
                "children_age_4": record.get("Children age 4", 0),
                "children_age_5": record.get("Children age 5", 0)
            }

            placeholders = {**analysis_static, **dynamic_part}

            system_prompt = parse_prompt(system_prompt_raw, placeholders)
            user_prompt = parse_prompt(user_prompt_raw, placeholders)

            if st.session_state.debug:
                st.info(f"Analyzing record #{analyzed_count}")
                st.write(record)
                st.info("System prompt (анализ):")
                st.write(system_prompt)
                st.info("User prompt (анализ):")
                st.write(user_prompt)

            generated_data = openai_chat(system_prompt, user_prompt, file_messages=file_messages)

            if st.session_state.debug:
                st.info("Generated data:")
                st.write(generated_data)

            upload_to_airtable(generated_data, "Responses")

    st.success("Анализ успешно завершен!")

# -------------------
# Вывод
# -------------------
def fetch_airtable_records(table_name: str, formula) -> list:
    api = Api(st.secrets.AIRTABLE_API_TOKEN)
    table = api.table(st.secrets.AIRTABLE_BASE_ID, table_name)
    return table.all(formula=formula)

def display_responses(selected_ad_name, selected_response_test_ids):
    """
    Отображаем ответы по выбранному Ad name и списку response_test_ids.
    Если ответы ещё не загружены в сессию, делаем запрос.
    Иначе берём из st.session_state.
    """

    if not selected_ad_name or not selected_response_test_ids:
        st.error("Пожалуйста, выберите название рекламы и тестовые ID ответа")
        return

    # Проверяем, загружали ли мы уже ответы для текущего набора фильтров
    # Чтобы различать наборы фильтров, можно завести ключ вида ("response_data", selected_ad_name, tuple(selected_response_test_ids))
    filter_key = f"resp_data_{selected_ad_name}_{','.join(sorted(selected_response_test_ids))}"

    # Если уже есть данные в сессии для этих же фильтров, берём их
    # Иначе делаем запрос и сохраняем
    if filter_key not in st.session_state:
        responses_formula = AND(
            EQ(Field("Ad name"), selected_ad_name),
            OR(*[EQ(Field("Response test ID"), rt_id) for rt_id in selected_response_test_ids])
        )
        response_records = fetch_airtable_records("Responses", responses_formula)

        response_data = []
        for r in response_records:
            fields = r.get("fields", {})
            persona_description = fields.get("Persona description", "")
            if isinstance(persona_description, list):
                persona_description = "; ".join(persona_description)

            response_entry = {
                "ID": r.get("id", ""),
                "Ad name": fields.get("Ad name", ""),
                "Response": fields.get("Response", ""),
                "Response clarity score": fields.get("Response clarity score", 0),
                "Response clarity description": fields.get("Response clarity description", ""),
                "Response likeability score": fields.get("Response likeability score", 0),
                "Response likeability description": fields.get("Response likeability description", ""),
                "Response trust score": fields.get("Response trust score", 0),
                "Response trust description": fields.get("Response trust description", ""),
                "Response diversity score": fields.get("Response diversity score", 0),
                "Response diversity description": fields.get("Response diversity description", ""),
                "Response message score": fields.get("Response message score", 0),
                "Response message description": fields.get("Response message description", ""),
                "Response free question 1": fields.get("Response free question 1", ""),
                "Response description": fields.get("Response description", ""),
                "Persona description": persona_description
            }
            response_data.append(response_entry)

        # Сохраняем результат в session_state для этих фильтров
        st.session_state[filter_key] = response_data

        # Сбросим индекс (чтобы начинать с первого ответа)
        st.session_state["current_response_index"] = 0
    else:
        response_data = st.session_state[filter_key]

    # Если пусто, выводим предупреждение
    if not response_data:
        st.warning("Нет данных по этим фильтрам.")
        return

    # Показ таблицы ответов
    with st.expander("Показать таблицу ответов"):
        num_rows = len(response_data)
        row_height = 30
        header_height = 40
        total_height = max(num_rows, 10) * row_height + header_height
        st.dataframe(response_data, height=total_height)

    # Устанавливаем индекс, если его ещё нет
    if "current_response_index" not in st.session_state:
        st.session_state["current_response_index"] = 0

    nav_placeholder = st.empty()
    detail_placeholder = st.empty()

    # Функция для обновления текущего ответа
    def update_response(delta):
        st.session_state.current_response_index = (
            st.session_state.current_response_index + delta
        ) % len(response_data)

    # Панель навигации
    nav_cols = nav_placeholder.columns([1, 2, 1])
    nav_cols[0].button(
        "⬅️ Предыдущий",
        on_click=update_response,
        args=(-1,),
        key="prev_btn"
    )
    nav_cols[1].write(
        f"Ответ {st.session_state.current_response_index + 1} из {len(response_data)}"
    )
    nav_cols[2].button(
        "Следующий ➡️",
        on_click=update_response,
        args=(1,),
        key="next_btn"
    )

    # Рисуем выбранный ответ
    current_index = st.session_state.current_response_index
    current_response = response_data[current_index]
    details_html = f"""
    <div style="border: 2px solid #ddd; padding: 10px; margin-top: 10px;">
      <p><b>Описание персоны:</b> {current_response.get("Persona description", "")}</p>
      <p><b>Ответ персоны:</b> {current_response.get("Response", "")}</p>
      <p><b>Понятность:</b> {current_response.get("Response clarity score", 0)} / {current_response.get("Response clarity description", "")}</p>
      <p><b>Лайкабилити:</b> {current_response.get("Response likeability score", 0)} / {current_response.get("Response likeability description", "")}</p>
      <p><b>Доверие:</b> {current_response.get("Response trust score", 0)} / {current_response.get("Response trust description", "")}</p>
      <p><b>Отличие:</b> {current_response.get("Response diversity score", 0)} / {current_response.get("Response diversity description", "")}</p>
      <p><b>Месседж:</b> {current_response.get("Response message score", 0)} / {current_response.get("Response message description", "")}</p>
    </div>
    """
    detail_placeholder.markdown(details_html, unsafe_allow_html=True)


# -------------------
# Функция получения данных из Airtable
# -------------------
def fetch_distinct_values(table_name, field_name, filter_by=None):
    api = Api(st.secrets.AIRTABLE_API_TOKEN)
    table = api.table(st.secrets.AIRTABLE_BASE_ID, table_name)
    formula = None
    if filter_by:
        formula = EQ(Field("Ad name"), filter_by)
    records = table.all(formula=formula)
    return list(set(record["fields"].get(field_name, "") for record in records if field_name in record["fields"]))

# -------------------
# UI вкладок
# -------------------
def show_response_analysis_tab():
    st.header("Анализ ответов")

    col_left, col_right = st.columns([3, 7])
    
    with col_left:
        st.subheader("Фильтры")
        ad_names = fetch_distinct_values("Responses", "Ad name")
        selected_ad_name = st.selectbox("Ad name", ad_names, key="ad_name_filter")
        
        response_test_ids = []
        if selected_ad_name:
            response_test_ids = fetch_distinct_values("Responses", "Response test ID", filter_by=selected_ad_name)
        selected_response_test_ids = st.multiselect("Response test ID", response_test_ids, key="response_test_id_filter")
    
    with col_right:
        st.subheader("Анализ ответов")

        # 1. Кнопка, чтобы пользователь включил «Показ» ответов
        if st.button("Показать", key="show_responses_button"):
            # Запоминаем в сессии, что пользователь хочет видеть ответы
            st.session_state["show_responses"] = True

        # 2. Если в сессии записано, что показываем, то вызываем display_responses
        if st.session_state.get("show_responses", False):
            display_responses(selected_ad_name, selected_response_test_ids)



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
    global number_of_persons_analysis, ad_name, ad_description, audio_description, metadata_description, message, free_question

    st.subheader("Отбор аудитории")
    if st.button("Отобрать персоны", key="select_persons_button"):
        formula = build_analysis_formula()
        records = fetch_analysis_records(formula, page_size=100, max_records=1000)
        data_for_table = []
        for r in records:
            fields = r.get("fields", {})
            # Формируем запись, включающую ID и все поля из записи
            record_data = {"ID": r.get("id", "")}
            record_data.update(fields)
            data_for_table.append(record_data)
        st.write(f"Найдено {len(data_for_table)} персон:")
        st.dataframe(data_for_table)
        st.session_state["selected_persons"] = data_for_table

    st.subheader("Анализ рекламы")
    ad_name = st.text_input("Название рекламы", placeholder="Введите название рекламы", key="ad_name_input")  # новое поле
    ad_description = st.text_input("Описание рекламы", placeholder="Введите максимально полное описание рекламы", key="ad_description_input")
    audio_description = st.text_input("Аудио описание", placeholder="Введите аудио описание рекламы", key="ad_audio_description_input")
    metadata_description = st.text_input("Метаданные", placeholder="Введите метаданные рекламы", key="ad_metadata_description_input")
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
    global analysis_children_count, analysis_children_age, analysis_gender_selected

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

        st.markdown("##### Пол")
        gender_options = ["Мужской", "Женский"]
        analysis_gender_selected = st.multiselect(
            "Пол",
            options=gender_options,
            default=gender_options,
            key="multiselect_gender_analysis"
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

    tab1, tab2, tab3, tab4 = st.tabs(["Генерация персон", "Аналитика", "Анализ ответов", "Настройки"])

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
        show_response_analysis_tab()

    with tab4:
        st.checkbox("Выводить отладочную информацию", key="debug")

if __name__ == "__main__":
    main()