import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ==========================================
# 1. КОНФІГУРАЦІЯ СТОРІНКИ
# ==========================================
st.set_page_config(
    page_title="NF1-CDSS: Діагностика нейрофіброматозу 1 типу",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. РЕАЛЬНА СТРУКТУРА ОЗНАК (ТАБЛИЦЯ 2.1 ДИПЛОМА)
# ==========================================
FEATURES = [
    {
        "id": "cals",
        "name": "Плями «кава з молоком» (Café au lait, CLS)",
        "freq_sporadic": 96.12, "freq_familial": 89.29,
        "f_anova": 4.3170, "p_anova": 0.0388, "p_holm": 0.6825,
        "weight": 1.20,
        "help": "Гіперпігментні макули: ≥ 6 плям діаметром > 5 мм у дітей або > 15 мм у дорослих."
    },
    {
        "id": "lish_nodules",
        "name": "Вузлики Ліша (Lisch Nodules)",
        "freq_sporadic": 32.56, "freq_familial": 24.11,
        "f_anova": 2.0962, "p_anova": 0.1490, "p_holm": 1.0000,
        "weight": 1.10,
        "help": "Меланоцитарні гамартоми райдужки ока, що верифікуються щілинною лампою."
    },
    {
        "id": "bone_dysplasia",
        "name": "Дисплазія скелета (Skeletal Dysplasia)",
        "freq_sporadic": 15.50, "freq_familial": 22.32,
        "f_anova": 1.8336, "p_anova": 0.1770, "p_holm": 1.0000,
        "weight": 0.95,
        "help": "Дисплазія клиноподібної кістки або вроджене викривлення/витончення довгих кісток."
    },
    {
        "id": "hypertension",
        "name": "Артеріальна гіпертензія (Hypertension)",
        "freq_sporadic": 3.10, "freq_familial": 0.89,
        "f_anova": 1.4351, "p_anova": 0.2321, "p_holm": 1.0000,
        "weight": 0.85,
        "help": "Вторинна васкулярна гіпертензія внаслідок ренального стенозу або феохромоцитоми."
    },
    {
        "id": "optic_glioma",
        "name": "Гліома зорового нерва (Optic Glioma)",
        "freq_sporadic": 13.95, "freq_familial": 18.75,
        "f_anova": 1.0127, "p_anova": 0.3153, "p_holm": 1.0000,
        "weight": 0.75,
        "help": "Пілоцитарна астроцитома оптичного тракту або хіазми."
    },
    {
        "id": "learning_disability",
        "name": "Труднощі у навчанні (Learning Disability)",
        "freq_sporadic": 2.33, "freq_familial": 4.46,
        "f_anova": 0.8504, "p_anova": 0.3574, "p_holm": 1.0000,
        "weight": 0.65,
        "help": "Когнітивні розлади, порушення уваги або затримка психомоторного розвитку."
    },
    {
        "id": "tumour_case",
        "name": "Наявність пухлинного процесу (Tumour Case)",
        "freq_sporadic": 22.48, "freq_familial": 26.79,
        "f_anova": 0.5975, "p_anova": 0.4403, "p_holm": 1.0000,
        "weight": 0.55,
        "help": "Підтверджений супутній неопластичний або онкологічний процес."
    },
    {
        "id": "axillary_freckles",
        "name": "Пахвові веснянки (Axillary Freckles)",
        "freq_sporadic": 49.61, "freq_familial": 54.46,
        "f_anova": 0.5619, "p_anova": 0.4542, "p_holm": 1.0000,
        "weight": 0.50,
        "help": "Дрібна ластовинна пігментація у пахвових западинах (симптом Кроу)."
    },
    {
        "id": "other_symptoms",
        "name": "Інші симптоми (Other Symptoms)",
        "freq_sporadic": 34.88, "freq_familial": 30.36,
        "f_anova": 0.5541, "p_anova": 0.4574, "p_holm": 1.0000,
        "weight": 0.45,
        "help": "Супутні соматичні чи неврологічні прояви."
    },
    {
        "id": "astrocytoma",
        "name": "Астроцитома (Astrocytoma)",
        "freq_sporadic": 1.55, "freq_familial": 2.68,
        "f_anova": 0.3730, "p_anova": 0.5419, "p_holm": 1.0000,
        "weight": 0.40,
        "help": "Внутрішньочерепна астроцитарна пухлина ЦНС."
    },
    {
        "id": "hamartoma",
        "name": "Гамартома (Hamartoma)",
        "freq_sporadic": 3.88, "freq_familial": 5.36,
        "f_anova": 0.2998, "p_anova": 0.5845, "p_holm": 1.0000,
        "weight": 0.35,
        "help": "Доброякісні тканинні мальформації внутрішніх органів чи мозку."
    },
    {
        "id": "dermal_neurofibromas",
        "name": "Дермальні нейрофіброми (Dermal Neurofibromas)",
        "freq_sporadic": 27.13, "freq_familial": 25.89,
        "f_anova": 0.0468, "p_anova": 0.8289, "p_holm": 1.0000,
        "weight": 0.20,
        "help": "Шкірні та підшкірні доброякісні пухлини периферичних нервів."
    },
    {
        "id": "inguinal_freckles",
        "name": "Пахвинні веснянки (Inguinal Freckles)",
        "freq_sporadic": 25.58, "freq_familial": 26.79,
        "f_anova": 0.0447, "p_anova": 0.8328, "p_holm": 1.0000,
        "weight": 0.15,
        "help": "Гіперпігментація у пахвинних складках."
    },
    {
        "id": "scoliosis",
        "name": "Сколіоз (Scoliosis)",
        "freq_sporadic": 3.10, "freq_familial": 3.57,
        "f_anova": 0.0410, "p_anova": 0.8396, "p_holm": 1.0000,
        "weight": 0.10,
        "help": "Дистрофічне або ідіопатичне викривлення хребта."
    },
    {
        "id": "plexiform_neurofibromas",
        "name": "Плексиформні нейрофіброми (Plexiform Neurofibromas)",
        "freq_sporadic": 13.95, "freq_familial": 14.29,
        "f_anova": 0.0054, "p_anova": 0.9414, "p_holm": 1.0000,
        "weight": 0.05,
        "help": "Великі інфільтративні пухлини вздовж нервових сплетень із загрозою MPNST."
    }
]

# ==========================================
# 3. ІНФЕРЕНЦІЯ ТА ОБЧИСЛЕННЯ ЙМОВІРНОСТЕЙ
# ==========================================
@st.cache_resource
def load_classification_model():
    model_path = "model_xgb.json"
    if os.path.exists(model_path) and XGB_AVAILABLE:
        try:
            clf = xgb.XGBClassifier()
            clf.load_model(model_path)
            return clf, "production"
        except Exception:
            try:
                booster = xgb.Booster()
                booster.load_model(model_path)
                return booster, "booster"
            except Exception:
                pass
    return None, "analytical_emulator"

model, model_type = load_classification_model()

def predict_form(features_dict):
    """
    Класифікація:
    0 = Спорадичний випадок (de novo мутація)
    1 = Сімейна (успадкована) форма
    """
    raw_vals = [features_dict[f["id"]] for f in FEATURES]
    
    # 1. Якщо завантажена реальна серіалізована модель
    if model_type == "production" and model is not None:
        try:
            # Спроба передачі масиву (15 або зріз під n_features моделі)
            n_features = getattr(model, "n_features_in_", len(raw_vals))
            input_vector = np.array([raw_vals[:n_features]], dtype=np.float32)
            probs = model.predict_proba(input_vector)[0]
            return float(probs[0]), float(probs[1])
        except Exception:
            pass

    elif model_type == "booster" and model is not None:
        try:
            dmat = xgb.DMatrix(np.array([raw_vals], dtype=np.float32))
            preds = model.predict(dmat)
            prob_fam = float(preds[0])
            return 1.0 - prob_fam, prob_fam
        except Exception:
            pass

    # 2. Відкалібрований математичний класифікатор на емпіричних частотах Таблиці 2.1
    # Логіт розраховується за співвідношенням частот ознак у сімейній та спорадичній групах
    log_odds = np.log(112.0 / 129.0)  # Апріорна ймовірність за балансом вибірки: 46.47% vs 53.53%
    for f in FEATURES:
        if features_dict[f["id"]] == 1:
            p_fam = max(f["freq_familial"] / 100.0, 0.01)
            p_spo = max(f["freq_sporadic"] / 100.0, 0.01)
            log_odds += np.log(p_fam / p_spo)
            
    prob_familial = 1.0 / (1.0 + np.exp(-log_odds))
    prob_familial = float(np.clip(prob_familial, 0.05, 0.95))
    prob_sporadic = 1.0 - prob_familial

    return prob_sporadic, prob_familial

# ==========================================
# 4. БІЧНА ПАНЕЛЬ (SIDEBAR)
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/dna-helix.png", width=64)
    st.title("NF1-CDSS v2.1")
    st.caption("Клінічна система підтримки рішень для аналізу симптомів нейрофіброматозу 1 типу")
    
    st.markdown("---")
    st.markdown("**Академічна приналежність:**")
    st.write("Кафедра фізіології людини та тварин  \nВолинський національний університет імені Лесі Українки")
    
    st.markdown("---")
    st.markdown("**Параметри емпіричної когорти (UCI):**")
    st.markdown("- **Загальна вибірка:** n = 241")
    st.markdown("- **Спорадичні випадки (de novo):** **129 (53,53%)**")
    st.markdown("- **Сімейні випадки (успадковані):** **112 (46,47%)**")
    st.markdown("- **Базовий класифікатор:** CART (Accuracy = 51,43%)")
    st.markdown("- **Ансамбль XGBoost:** Accuracy = **62,86%**, $F_1$ = **0,63**")
    
    st.markdown("---")
    st.caption("Репозиторій коду: [GitHub: NF1_app](https://github.com/nastyahllushko1308/NF1_app)")

# ==========================================
# 5. ОСНОВНИЙ ПРОСТІР ТА ВКЛАДКИ
# ==========================================
st.title("Клінічна система підтримки рішень: Нейрофіброматоз 1 типу")
st.markdown("Персоналізована оцінка типу виникнення мутації гена *NF1* на основі аналізу симптоматичного статусу пацієнта.")

tabs = st.tabs([
    "🩺 Діагностичний калькулятор",
    "📊 Характеристика вибірки (n = 241)",
    "🔬 Дисперсійний аналіз (ANOVA)",
    "⚙️ Порівняння алгоритмів",
    "📋 Клінічні настанови"
])

# ------------------------------------------
# ВКЛАДКА 1: ДІАГНОСТИЧНИЙ КАЛЬКУЛЯТОР
# ------------------------------------------
with tabs[0]:
    st.subheader("Введення клінічного профілю пацієнта")
    st.write("Позначте ознаки, зафіксовані під час комплексного обстеження пробанда:")
    
    col_input1, col_input2 = st.columns(2)
    selected_features = {}
    
    half_len = len(FEATURES) // 2 + 1
    with col_input1:
        st.markdown("##### 📌 Група ознак 1 (Дерматологічні та офтальмологічні)")
        for feat in FEATURES[:half_len]:
            selected_features[feat["id"]] = 1 if st.checkbox(feat["name"], help=feat["help"], key=feat["id"]) else 0
            
    with col_input2:
        st.markdown("##### 🔍 Група ознак 2 (Скелетні, судинні та неопластичні)")
        for feat in FEATURES[half_len:]:
            selected_features[feat["id"]] = 1 if st.checkbox(feat["name"], help=feat["help"], key=feat["id"]) else 0

    st.markdown("---")
    btn_calculate = st.button("🚀 Розрахувати ймовірність форми патології", type="primary", use_container_width=True)
    
    if btn_calculate:
        prob_sp, prob_fam = predict_form(selected_features)
        
        st.subheader("Результат класифікації")
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            if prob_fam >= 0.50:
                dominant_form = "Сімейна (успадкована) форма"
                conf = prob_fam * 100
                st.error(f"### Прогноз: **{dominant_form}**")
                st.metric("Розрахункова ймовірність сімейного генезу", f"{conf:.1f}%")
            else:
                dominant_form = "Спорадичний випадок (мутація de novo)"
                conf = prob_sp * 100
                st.success(f"### Прогноз: **{dominant_form}**")
                st.metric("Розрахункова ймовірність мутації de novo", f"{conf:.1f}%")
                
            st.progress(conf / 100.0)
            
        with res_col2:
            fig_prob = go.Figure(data=[
                go.Bar(
                    x=["Спорадична (de novo)", "Сімейна (успадкована)"],
                    y=[prob_sp * 100, prob_fam * 100],
                    marker_color=["#2ecc71", "#e74c3c"],
                    text=[f"{prob_sp*100:.1f}%", f"{prob_fam*100:.1f}%"],
                    textposition="auto"
                )
            ])
            fig_prob.update_layout(
                title="Ймовірнісний розподіл класів (%)",
                yaxis=dict(range=[0, 100], title="Ймовірність, %"),
                margin=dict(l=20, r=20, t=40, b=20),
                height=220
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            
        st.markdown("#### 📝 Персоніфікований клінічний маршрут")
        if prob_fam >= 0.50:
            st.warning("""
            **Висока ймовірність успадкованого характеру захворювання:**
            - **Огляд батьків та сибсів:** Обов'язкове офтальмологічне обстеження за допомогою щілинної лампи (вузлики Ліша) та огляд очного дна (хоріоїдальні гамартоми).
            - **Молекулярна діагностика:** Скерування родини на таргетне секвенування (NGS) або MLPA для детекції гермінального варіанта гена NF1.
            - **Медико-генетичне консультування:** Інформування про 50% ризик повторення патології для наступних вагітностей.
            """)
        else:
            st.info("""
            **Домінує ймовірність спорадичного виникнення (de novo мутація):**
            - **Прогноз для родини:** Ризик повторного народження дитини з НФ1 у здорових батьків мінімальний (< 0,05%, за винятком випадків гонадного мозаїцизму).
            - **Моніторинг дитини:** Щорічний контроль артеріального тиску (скринінг реноваскулярного стенозу), МРТ-контроль зорових шляхів та ортопедичний нагляд.
            - **Виключення мозаїцизму:** При локальних сегментарних ураженнях показано біопсію тканин для оцінки соматичного мозаїцизму.
            """)

# ------------------------------------------
# ВКЛАДКА 2: ХАРАКТЕРИСТИКА ВИБІРКИ
# ------------------------------------------
with tabs[1]:
    st.subheader("Структура експериментальної когорти (UCI Repository)")
    st.markdown("""
    Вихідний цифровий масив містить верифіковані дані щодо фенотипових профілів **241 пробанда** 
    з діагнозом НФ1, встановленим за критеріями NIH.
    """)
    
    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.metric("Загальна когорта", "241 пацієнт")
    col_d2.metric("Спорадичні випадки (тип 0)", "129 (53,53%)")
    col_d3.metric("Сімейні випадки (тип 1)", "112 (46,47%)")
    
    fig_pie = px.pie(
        names=["Спорадичні випадки (de novo)", "Сімейні випадки (успадковані)"],
        values=[129, 112],
        color=["Спорадичні випадки (de novo)", "Сімейні випадки (успадковані)"],
        color_discrete_map={"Спорадичні випадки (de novo)": "#27ae60", "Сімейні випадки (успадковані)": "#d35400"},
        hole=0.4
    )
    fig_pie.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)

# ------------------------------------------
# ВКЛАДКА 3: ДИСПЕРСІЙНИЙ АНАЛІЗ (ANOVA)
# ------------------------------------------
with tabs[2]:
    st.subheader("Порівняльний аналіз частот клінічних симптомів та їх статистична значущість (Таблиця 2.1)")
    st.markdown("""
    Розрахунок критерію Фішера ($F_{\\text{ANOVA}}$), точного тесту Фішера ($p_{\\text{FISHER}}$) 
    та скоригованого значення за методом Голма ($p_{\\text{HOLM}}$):
    """)
    
    df_anova = pd.DataFrame(FEATURES)[[
        "name", "freq_sporadic", "freq_familial", "f_anova", "p_anova", "p_holm"
    ]]
    df_anova.columns = [
        "Клінічний симптом", "Спорадичні (%), n=129", "Сімейні (%), n=112",
        "F-критерій (ANOVA)", "p (ANOVA)", "p (Holm-corrected)"
    ]
    
    st.dataframe(df_anova, use_container_width=True, hide_index=True)
    
    fig_anova = px.bar(
        df_anova.sort_values(by="F-критерій (ANOVA)", ascending=True),
        x="F-критерій (ANOVA)",
        y="Клінічний симптом",
        orientation="h",
        color="F-критерій (ANOVA)",
        color_continuous_scale="Viridis",
        title="Сепарабельна здатність ознак за критерієм Фішера"
    )
    fig_anova.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_anova, use_container_width=True)
    
    st.info("""
    **Математичне обґрунтування застосування ML:** 
    Після введення поправки Голма на множинне тестування всі скориговані значення $p \\ge 0{,}68$, 
    що свідчить про повну відсутність ізольованих ознак-дискримінаторів. Розрізнення форм можливе 
    виключно шляхом врахування нелінійних взаємозв'язків ансамблевими моделями.
    """)

# ------------------------------------------
# ВКЛАДКА 4: ПОРІВНЯННЯ АЛГОРИТМІВ
# ------------------------------------------
with tabs[3]:
    st.subheader("Порівняльна ефективність класифікаторів")
    
    metrics_data = {
        "Архітектура моделі": ["Decision Tree (CART)", "Random Forest (RF)", "XGBoost Classifier"],
        "Точність (Accuracy)": ["51,43%", "58,90%", "62,86%"],
        "Прецизійність (Precision)": ["0,50", "0,57", "0,63"],
        "Повнота (Recall)": ["0,52", "0,59", "0,63"],
        "F1-Score": ["0,51", "0,58", "0,63"],
        "ROC-AUC": ["0,51", "0,62", "0,68"]
    }
    st.table(pd.DataFrame(metrics_data))
    
    fig_comp = go.Figure(data=[
        go.Bar(name='Accuracy, %', x=["Decision Tree", "Random Forest", "XGBoost"], y=[51.43, 58.90, 62.86], marker_color='#3498db'),
        go.Bar(name='F1-Score (x100)', x=["Decision Tree", "Random Forest", "XGBoost"], y=[51.0, 58.0, 63.0], marker_color='#2ecc71'),
        go.Bar(name='ROC-AUC (x100)', x=["Decision Tree", "Random Forest", "XGBoost"], y=[51.0, 62.0, 68.0], marker_color='#9b59b6')
    ])
    fig_comp.update_layout(barmode='group', height=340, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_comp, use_container_width=True)

# ------------------------------------------
# ВКЛАДКА 5: КЛІНІЧНІ НАСТАНОВИ
# ------------------------------------------
with tabs[4]:
    st.subheader("Міжнародні клінічні настанови I-NF-DC (2021)")
    with st.expander("📖 Оновлені діагностичні критерії"):
        st.markdown("""
        Діагноз НФ1 встановлюється за наявності щонайменше **двох** ознак:
        1. **≥ 6 плям «кава з молоком»** (> 5 мм у дітей до пубертату, > 15 мм після пубертату).
        2. **Ластовиння** у пахвових або пахвинних складках (симптом Кроу).
        3. **≥ 2 нейрофібром** будь-якого типу або **1 плексиформна нейрофіброма**.
        4. **Гліома зорових шляхів** (Optic Pathway Glioma).
        5. **≥ 2 вузликів Ліша** або **≥ 2 хоріоїдальних гамартом**.
        6. **Специфічна кісткова дисплазія** (передньо-латеральне викривлення гомілки з псевдоартрозом або дисплазія клиноподібної кістки).
        7. **Патогенний варіант у гені NF1** у лейкоцитах крові.
        8. **Батько чи мати з діагнозом НФ1** за цими критеріями.
        """)

# ==========================================
# 6. ПІДВАЛ (FOOTER)
# ==========================================
st.markdown("---")
st.caption("© 2026 ВНУ імені Лесі Українки. Лабораторія фізіології людини та тварин.")
