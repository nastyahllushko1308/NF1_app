import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Налаштування сторінки
st.set_page_config(
    page_title="CDSS Нейрофіброматоз 1 типу",
    page_icon="🧬",
    layout="centered"
)

# Заголовок та шапка системи
st.title("🧬 Інтелектуальна система підтримки прийняття клінічних рішень (CDSS)")
st.markdown("### Диференційна діагностика форми походження нейрофіброматозу 1 типу")
st.caption("Розробник: студентка магістратури Глушко А. О. (ВНУ імені Лесі Українки)")

st.info("ℹ️ **Інструкція лікаря:** Відзначте клінічні симптоми, виявлені у пацієнта під час комплексного обстеження.")

# Завантаження навченої моделі XGBoost
@st.cache_resource
def load_trained_model():
    model = XGBClassifier()
    model.load_model("model_xgb.json")
    return model

try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Помилка завантаження моделі: {e}")
    st.stop()

# Інтерфейс вибору ознак (розбивка на 2 колонки)
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 1. Системні маркери (ANOVA)")
    lisch = st.checkbox("Вузлики Ліша (Lisch Nodules)")
    pseudo = st.checkbox("Вроджений псевдоартроз")
    hyper = st.checkbox("Артеріальна гіпертензія")
    leukemia = st.checkbox("МДС / Лейкемія")

with col2:
    st.markdown("#### 2. Соматичні та нервові ознаки")
    delay = st.checkbox("Затримка розвитку")
    bone = st.checkbox("Дисплазія кісток")
    cnf = st.checkbox("Шкірні нейрофіброми (CNF)")
    cals = st.checkbox("Пігментні плями (CALS)")

# Формуємо вектор ознак у тому самому порядку, на якому навчався XGBoost
features_list = [
    int(lisch),
    int(pseudo),
    int(hyper),
    int(leukemia),
    int(delay),
    int(bone),
    int(cnf),
    int(cals)
]

features_names = [
    'Lisch_Nodules', 'Congenital_Pseudarthrosis', 'Hypertension', 
    'MDS_Leukemia', 'Developmental_Delay', 'Bone_Dysplasia', 
    'Cutaneous_Neurofibromas', 'Cafe_au_Lait_Spots'
]

# Кнопка запуску обчислення
if st.button("🧬 Розрахувати предикційний вердикт моделі", type="primary", use_container_width=True):
    # Перетворюємо у DataFrame, щоб передати точні назви колонок моделі
    input_df = pd.DataFrame([features_list], columns=features_names)
    
    # Отримуємо ймовірності від справжньої моделі XGBoost
    probabilities = model.predict_proba(input_df)[0]
    prob_familial = probabilities[1] * 100
    prob_sporadic = probabilities[0] * 100

    st.markdown("---")
    st.markdown("### 📊 Діагностичний висновок системи:")

    if prob_familial >= 50.0:
        st.error(f"**Рекомендований клас: СІМЕЙНА ФОРМА (Успадкована мутація гена NF1)**")
        st.metric(label="Розрахована ймовірність класу", value=f"{prob_familial:.2f}%")
        st.caption("⚠️ Модель зафіксувала стійкий симптомокомплекс, притаманний пацієнтам із обтяженим сімейним анамнезом.")
    else:
        st.success(f"**Рекомендований клас: СПОРАДИЧНИЙ ВИПАДОК (De novo мутація)**")
        st.metric(label="Розрахована ймовірність класу", value=f"{prob_sporadic:.2f}%")
        st.caption("🌱 Модель схиляється до первинного виникнення гермінальної мутації за відсутності аналогічних проявів у батьків.")

    st.warning("⚠️ **Увага!** Даний висновок є предикцією ансамблевого класифікатора і виконує роль довідкового «другого читача». Остаточний діагноз встановлюється консиліумом лікарів на основі молекулярно-генетичного аналізу.")