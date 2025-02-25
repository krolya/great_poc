import streamlit as st
import json
import datetime
import requests
import itertools
import base64

from openai import OpenAI
from pyairtable import Api
from pyairtable.formulas import AND, OR, EQ, GTE, LTE, Field
from concurrent.futures import ThreadPoolExecutor

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
analysis_tags_selected = []

# -------------------
# Функции загрузки/выгрузки схемы Airtable из GitHub
# -------------------
def get_airtable_schema():
    api = Api(st.secrets.AIRTABLE_API_TOKEN)
    base_id = st.secrets.AIRTABLE_BASE_ID
    schema = api.base(base_id).schema()
    
    # Преобразуем объект schema в сериализуемый JSON
    schema_dict = {
        "tables": []
    }
    
    for table in schema.tables:
        table_dict = {
            "name": table.name,
            "fields": []
        }
        
        for field in table.fields:
            field_dict = {"name": field.name, "type": field.type}
            
            if hasattr(field, "options") and isinstance(field.options, dict):
                field_dict["options"] = field.options
            
            table_dict["fields"].append(field_dict)
        
        schema_dict["tables"].append(table_dict)
    
    return schema_dict

def upload_schema_to_airtable(schema):
    api = Api(st.secrets.AIRTABLE_API_TOKEN)
    base_id = st.secrets.AIRTABLE_BASE_ID
    
    for table in schema.get("tables", []):
        api.table(base_id, table['name']).create(table)

# -------------------
# Функция загрузки файла из GitHub
# -------------------
def get_file_from_github(file_path: str) -> str:
    url = f"https://raw.githubusercontent.com/krolya/great_poc/main/{file_path}"
    headers = {"Authorization": f"Bearer {st.secrets.GITHUB_API_TOKEN}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text

def save_file_to_github(content: dict, file_path: str):
    url = f"https://api.github.com/repos/krolya/great_poc/contents/{file_path}"
    headers = {
        "Authorization": f"Bearer {st.secrets.GITHUB_API_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    auth_check = requests.get("https://api.github.com/user", headers=headers)
    if auth_check.status_code != 200:
        st.error("Ошибка: Недействительный GitHub токен или недостаточно прав!")
        return
    
    response = requests.get(url, headers=headers)
    sha = response.json().get("sha", "") if response.status_code == 200 else ""
    
    json_content = json.dumps(content, indent=4, ensure_ascii=False)
    base64_content = base64.b64encode(json_content.encode("utf-8")).decode("utf-8")
    
    data = {
        "message": "Upload Airtable schema",
        "content": base64_content,
    }
    if sha:
        data["sha"] = sha  # Добавляем SHA для обновления файла
    
    response = requests.put(url, headers=headers, json=data)
    if response.status_code not in [200, 201]:
        st.error(f"Ошибка при загрузке на GitHub: {response.status_code}, {response.text}")
        return
    
    st.success("Файл успешно загружен на GitHub")

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

# -------------------
# Функция загрузки новых тэгов в Airtable
# -------------------
def ensure_tags_exist(api, base_id, table_name, tags):
    """
    Проверяет существование тэгов в Airtable и добавляет новые, если их нет.
    """
    table = api.table(base_id, table_name)
    existing_tags = fetch_distinct_values(table_name, "Tags")
    new_tags = set(tags) - set(existing_tags)
    
    if new_tags:
        table.batch_create([{"fields": {"Tags": [tag]}} for tag in new_tags])
    
    return list(set(tags))  # Возвращает объединенный список старых и новых тэгов

def upload_to_airtable(data, table_name="Personas") -> int:
    api = Api(st.secrets.AIRTABLE_API_TOKEN)
    table = api.table(st.secrets.AIRTABLE_BASE_ID, table_name)
    st.info("Загружаем данные в Airtable...")

     # Убеждаемся, что все тэги существуют
    #ensure_tags_exist(api, st.secrets.AIRTABLE_BASE_ID, table_name, tags)

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

    # Добавляем фильтр по тегам: если указаны тэги, то каждая запись должна содержать все выбранные
    if analysis_tags_selected:
        tag_conditions = []
        for tag in analysis_tags_selected:
            tag_conditions.append(f"FIND('{tag}', {{Tags}}) > 0")
        conds.append("AND(" + ", ".join(tag_conditions) + ")")
    
    formula_obj = AND(*conds)
    return str(formula_obj)

# -------------------
# Генерация
# -------------------
def generate_person():
    global generation_id
    generation_id = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

    if st.session_state.debug:
        system_prompt_raw = st.session_state.get("gen_system_prompt", "")
        user_prompt_raw = st.session_state.get("gen_user_prompt", "")
    else:
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
        "children_age_max": children_age[1],
        "generation_instruction": generation_instruction,
        "tags": tags
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
# --- Bare mode parallel analysis functions ---
def openai_chat_bare(system_prompt: str, user_prompt: str, openai_api_key: str, nebius_api_key: str, json_schema: dict, file_messages=None, debug=False) -> str:
    """
    Bare mode версия функции для запроса OpenAI с использованием предоставленной json_schema.
    Секреты для OpenAI передаются напрямую через параметры.
    """
    from openai import OpenAI

    global model_name
    if "deepseek" not in model_name.lower():
        client = OpenAI(api_key=openai_api_key)
    else:
        client = OpenAI(
            base_url="https://api.studio.nebius.ai/v1/",
            api_key=nebius_api_key,
        )

    messages = [{"role": "system", "content": system_prompt}]
    if file_messages:
        user_content = [{"type": "text", "text": user_prompt}] + file_messages
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": user_prompt})

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": json_schema
        }
    )

    if debug:
        print("OpenAI response:", completion.choices[0].message.content)

    return completion.choices[0].message.content

def upload_to_airtable_bare(data: str, airtable_api_token: str, airtable_base_id: str, table_name: str = "Responses", debug: bool = False) -> int:
    """
    Bare mode версия функции загрузки данных в Airtable.
    Секреты передаются через параметры.
    """
    import json
    from pyairtable import Api

    api = Api(airtable_api_token)
    table = api.table(airtable_base_id, table_name)
    
    if debug:
        print("Uploading data to Airtable...")
    
    records = json.loads(data)
    response = table.batch_create(records["records"])
    
    if debug:
        print("Airtable response:", response)
    
    return len(response)

def analyze_ad_chunk(start_index, end_index, response_test_id: str, persons: list,
                       system_prompt_raw: str, user_prompt_raw: str, file_messages, 
                       analysis_static: dict, json_schema: dict,
                       openai_api_key: str, nebius_api_key: str,
                       airtable_api_token: str, airtable_base_id: str, debug: bool = False) -> int:
    """
    Обрабатывает срез записей (от start_index до end_index) без использования Streamlit UI.
    Для каждой записи:
      - формирует промпты с подстановкой динамических и статических параметров,
      - получает сгенерированный ответ через openai_chat_bare (использующий json_schema),
      - загружает данные в Airtable через upload_to_airtable_bare.
    Возвращает число обработанных записей.
    """
    processed_count = 0
    for idx, record in enumerate(persons[start_index:end_index], start=start_index):
        dynamic_part = {
            "response_test_id": response_test_id,
            "record_id": record.get("Record ID", ""),
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

        generated_data = openai_chat_bare(
            system_prompt,
            user_prompt,
            openai_api_key,
            nebius_api_key,
            json_schema,
            file_messages=file_messages,
            debug=debug
        )
        
        upload_to_airtable_bare(
            generated_data,
            airtable_api_token,
            airtable_base_id,
            table_name="Responses",
            debug=debug
        )
        
        processed_count += 1
    if debug:
        print(f"Chunk processed: {processed_count} records from index {start_index} to {end_index - 1}")
    return processed_count

def parallel_analyze_ad(num_threads):
    # Извлекаем список персон из st.session_state
    persons = st.session_state.get("selected_persons", [])
    if not persons:
        st.error("Нет отобранных персон. Пожалуйста, отберите персоны сначала.")
        return

    total_persons = len(persons)
    if num_threads < 1:
        st.error("Количество потоков должно быть не менее 1")
        return

    # Скачиваем JSON-схему один раз
    import json
    ad_analysis_schema_str = get_file_from_github("ad_analysis.json")
    json_schema = json.loads(ad_analysis_schema_str)

    # Определяем число персон для анализа и начальный индекс
    number_of_persons_analysis = st.session_state.get("number_of_persons_analysis", 100)
    start_index = st.session_state.get("analysis_start_index", 0)
    available = len(persons) - start_index
    total_to_process = min(number_of_persons_analysis, available)

    # Готовим параметры для анализа
    debug = st.session_state.get("debug", False)
    response_test_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    if debug:
        system_prompt_raw = st.session_state.get("analysis_system_prompt", "")
        user_prompt_raw = st.session_state.get("analysis_user_prompt", "")
    else:
        system_prompt_raw = get_file_from_github("ad_analysis_system.promt")
        user_prompt_raw = get_file_from_github("ad_analysis.promt")

    uploaded_files = st.session_state.get("analysis_uploaded_files", [])
    file_messages = []
    for fdict in uploaded_files:
        file_messages.append({
            "type": "image_url",
            "image_url": {"url": fdict["content"]}
        })

    # Статические параметры для анализа
    analysis_static = {
        "model_name": model_name,
        "ad_name": ad_name,
        "ad_description": ad_description,
        "audio_description": audio_description,
        "metadata_description": metadata_description,
        "message": message,
        "free_question": free_question
    }

    # Распределяем записи между потоками
    chunk_size = total_to_process // num_threads
    remainder = total_to_process % num_threads

    processed_total = 0
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        current_start = start_index
        for i in range(num_threads):
            extra = 1 if i < remainder else 0
            current_end = current_start + chunk_size + extra
            futures.append(
                executor.submit(
                    analyze_ad_chunk,
                    current_start, current_end,
                    response_test_id,
                    persons,
                    system_prompt_raw,
                    user_prompt_raw,
                    file_messages,
                    analysis_static,
                    json_schema,
                    st.secrets.OPENAI_API_KEY,   # Секрет для OpenAI
                    st.secrets.NEBIUS_API_KEY,     # Секрет для Nebius (если используется)
                    st.secrets.AIRTABLE_API_TOKEN, # Секрет для Airtable
                    st.secrets.AIRTABLE_BASE_ID,   # Базовый ID Airtable
                    debug
                )
            )
            print(f"[DEBUG] Thread {i+1} processing records from {current_start} to {current_end - 1}")
            current_start = current_end

        # Суммируем число обработанных записей, полученных из потоков
        for future in futures:
            processed_total += future.result()

    st.success(f"Анализ успешно завершён! Всего обработано {processed_total} записей.")

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

    # Получаем начальный индекс для анализа
    start_index = st.session_state.get("analysis_start_index", 0)
    if start_index < 0 or start_index >= len(persons):
        st.error(f"Начальный индекс ({start_index}) вне диапазона (0 - {len(persons)-1}).")
        return

    analyzed_count = 0
    total_available = len(persons) - start_index
    total_to_process = min(number_of_persons_analysis, total_available)
    progress_placeholder = st.empty()

    with st.spinner("Генерация ответов..."):
        for idx, record in enumerate(persons[start_index:], start=start_index):
            if analyzed_count >= number_of_persons_analysis:
                break

            analyzed_count += 1
            progress_placeholder.write(f"Обработано {analyzed_count} из {total_to_process} (индекс: {idx} из {len(persons)})")

            # Формируем динамическую часть для подстановки в промпты
            dynamic_part = {
                "response_test_id": response_test_id,
                "record_id": record.get("Record ID", ""),
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
            st.info(f"Processed record #{analyzed_count} (индекс {idx})")

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
    filter_key = f"resp_data_{selected_ad_name}_{','.join(sorted(selected_response_test_ids))}"

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

        st.session_state[filter_key] = response_data
        st.session_state["current_response_index"] = 0
    else:
        response_data = st.session_state[filter_key]

    if not response_data:
        st.warning("Нет данных по этим фильтрам.")
        return

    with st.expander("Показать таблицу ответов"):
        num_rows = len(response_data)
        row_height = 30
        header_height = 40
        total_height = max(num_rows, 10) * row_height + header_height
        st.dataframe(response_data, height=total_height)

    if "current_response_index" not in st.session_state:
        st.session_state["current_response_index"] = 0

    nav_placeholder = st.empty()
    detail_placeholder = st.empty()

    def update_response(delta):
        st.session_state.current_response_index = (
            st.session_state.current_response_index + delta
        ) % len(response_data)

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
      <p><b>Response description:</b> {current_response.get("Response description", "")}</p>
      <p><b>Response free question 1:</b> {current_response.get("Response free question 1", "")}</p>
    </div>
    """
    detail_placeholder.markdown(details_html, unsafe_allow_html=True)
    
    st.markdown("### Сводная таблица по ответам")
    response_params = [
        ("Понятность", "Response clarity score"),
        ("Лайкабилити", "Response likeability score"),
        ("Доверие", "Response trust score"),
        ("Отличие", "Response diversity score"),
        ("Месседж", "Response message score")
    ]
    
    summary_data = []
    for param_name, field_key in response_params:
        cleaned_scores = []
        for resp in response_data:
            try:
                score = float(resp.get(field_key, 0))
                cleaned_scores.append(score)
            except (ValueError, TypeError):
                continue
        if cleaned_scores:
            min_val = min(cleaned_scores)
            avg_val = sum(cleaned_scores) / len(cleaned_scores)
            max_val = max(cleaned_scores)
        else:
            min_val = avg_val = max_val = 0
        
        summary_data.append({
            "Параметр ответа": param_name,
            "Минимальный скор среди ответов": min_val,
            "Среднее арифметическое ответов": avg_val,
            "Максимальный скор среди ответов": max_val
        })
    
    st.table(summary_data)


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

    unique_values = set()
    for record in records:
        field_value = record["fields"].get(field_name, "")
        if isinstance(field_value, list):  # Если это список (мультиселект)
            unique_values.update(field_value)  # Добавляем все элементы списка
        else:
            unique_values.add(field_value)  # Добавляем одиночное значение
    
    return list(unique_values)


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


def show_reports_tab():
    """
    Функция отображает закладку "Отчеты" с заголовком и списком ссылок.
    """
    st.header("Отчеты")
    st.markdown("[Общее сравнение](https://airtable.com/appUEREr4qQKijWHj/shrWpCNYxuVH85jST)")
    st.markdown("[Аудитория](https://airtable.com/appUEREr4qQKijWHj/shrpv5N42OvDsEjtE)")
    st.markdown("[Идея 1. Все очень](https://airtable.com/appUEREr4qQKijWHj/shrLtfMjdaDotTYd9)")
    st.markdown("[Идея 2. Всегда освежает](https://airtable.com/appUEREr4qQKijWHj/shr96HXkEdAhALDQQ)")
    st.markdown("[Идея 3. Впечатляет, освежает](https://airtable.com/appUEREr4qQKijWHj/shrrLUqrzb8A8vyre)")
    st.markdown("[Идея 4. Открывайте квас снова и снова](https://airtable.com/appUEREr4qQKijWHj/shrVJy2pGZkdHKLlL)")


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

    # Новое текстовое поле для инструкции генерации
    global generation_instruction
    generation_instruction = st.text_input(
        "Дополнительная инструкция для генерации персон ", 
        placeholder="Введите инструкцию для генерации, можно добавить напр. черты характера, особенности, итд", 
        key="generation_instruction"
    )

    if st.button("Сгенерировать", key="generate_button"):
        st.info("Генерация началась...")
        generate_person()


def show_analysis_tab():
    global number_of_persons_analysis, ad_name, ad_description, audio_description, metadata_description, message, free_question

    st.subheader("Отбор аудитории")
    if st.button("Отобрать персоны", key="select_persons_button"):
        formula = build_analysis_formula()

        if st.session_state.debug:
            st.write(formula)

        records = fetch_analysis_records(formula, page_size=100, max_records=number_of_persons_analysis)
        data_for_table = []
        for r in records:
            fields = r.get("fields", {})
            # Формируем запись, включающую ID и все поля из записи
            record_data = {"Record ID": r.get("id", "")}
            record_data.update(fields)
            data_for_table.append(record_data)
        st.write(f"Найдено {len(data_for_table)} персон:")
        st.dataframe(data_for_table)
        st.session_state["selected_persons"] = data_for_table

    st.subheader("Анализ рекламы")
    ad_name = st.text_input("Название рекламы", placeholder="Введите название рекламы", key="ad_name_input")  # новое поле
    ad_description = st.text_input("Описание рекламы", placeholder="Введите максимально полное описание рекламы", key="ad_description_input")
    audio_description = st.text_input("Основная идея для рекламного ролика", placeholder="Введите jсновную идею для рекламного ролика", key="ad_audio_description_input")
    metadata_description = st.text_input("Описание кадров", placeholder="Введите описание кадров", key="ad_metadata_description_input")
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


    # Новое числовое поле для указания начального индекса при анализе
    st.number_input("Начальный индекс для анализа", min_value=0, value=0, step=1, key="analysis_start_index")
    st.number_input("Количество потоков для анализа", min_value=0, value=1, step=1, key="analysis_num_threads")

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
        if st.session_state.get("analysis_num_threads", 1) <= 1:
            analyze_ad()
        else:
            parallel_analyze_ad(st.session_state["analysis_num_threads"])


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
        income_options = ["Низкий", "Низкий плюс","Средний", "Средний плюс","Высокий","Высокий плюс"]
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

    # Поле для тегов (будет использовано как placeholder для генерации)
    tags = st.text_input(
        "Тэги", 
        placeholder="Введите тэги через запятую", 
        key="tags_gen"
    )


def show_filters_tab_analysis():
    global number_of_persons_analysis, analysis_tags_selected, analysis_age_range, analysis_income_selected, analysis_education_selected
    global analysis_selected_regions, analysis_city_size_selected, analysis_marital_selected, analysis_children_count, analysis_children_age, analysis_gender_selected

    st.header("Фильтры")
    with st.expander("Основные настройки", expanded=True):
        number_of_persons_analysis = st.slider(
            "Количество персон для анализа", 
            0, 1000, 20, 
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
            "До 100 000 человек",
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

        # Новый multi-select для тэгов
        all_tags = fetch_distinct_values("Personas", "Tags")
        global analysis_tags_selected
        analysis_tags_selected = st.multiselect("Тэги", options=all_tags, key="tags_filter_analysis")


def main():
    st.set_page_config(page_title="Более нормальный человек", layout="wide")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Генерация персон", "Аналитика", "Анализ ответов","Отчеты", "Настройки"])

    if "debug" not in st.session_state:
        st.session_state.debug = False

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
        show_reports_tab()

    with tab5:
        st.checkbox("Выводить отладочную информацию", key="debug")

        if st.session_state.debug:
            st.subheader("Управление схемой Airtable")
            file_path = "airtable_schema.json"
            
            if st.button("Выгрузить схему Airtable на GitHub"):
                schema = get_airtable_schema()
                save_file_to_github(json.dumps(schema, indent=4, ensure_ascii=False), file_path)
                st.success("Схема успешно выгружена на GitHub")
            
            if st.button("Загрузить схему из GitHub в Airtable"):
                if st.confirm("ВНИМАНИЕ! ОПАСНОСТЬ! Вы уверены, что хотите загрузить схему из GitHub в Airtable? Это может перезаписать текущую структуру данных."):
                    schema_text = get_file_from_github(file_path)
                    schema = json.loads(schema_text)
                    upload_schema_to_airtable(schema)
                    st.success("Схема успешно загружена в Airtable")

if __name__ == "__main__":
    main()
