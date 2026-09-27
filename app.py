import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="توقع إعادة دخول مرضى السكري", page_icon="🏥", layout="centered")


# ============================================================
# دالة تجميع أكواد ICD-9 - لازم تكون كود عادي (مش pickle)
# ============================================================
def map_icd9(code):
    if code is None or code == '':
        return 'Other'
    try:
        code = float(code)
    except (ValueError, TypeError):
        return 'Other'
    if 390 <= code <= 459 or code == 785:
        return 'Circulatory'
    elif 460 <= code <= 519 or code == 786:
        return 'Respiratory'
    elif 520 <= code <= 579 or code == 787:
        return 'Digestive'
    elif 250 <= code < 251:
        return 'Diabetes'
    elif 800 <= code <= 999:
        return 'Injury'
    elif 710 <= code <= 739:
        return 'Musculoskeletal'
    elif 580 <= code <= 629 or code == 788:
        return 'Genitourinary'
    elif 140 <= code <= 239:
        return 'Neoplasms'
    else:
        return 'Other'


# ============================================================
# تحميل كل ملفات pickle (مرة واحدة بس، مع cache)
# ============================================================
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("rf_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    onehot_encoders = pickle.load(open("onehot_encoders.pkl", "rb"))
    label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
    ordinal_encoders = pickle.load(open("ordinal_encoders.pkl", "rb"))
    age_mapping = pickle.load(open("age_mapping.pkl", "rb"))
    heavy_skew_cols = pickle.load(open("heavy_skew_cols.pkl", "rb"))
    moderate_cols = pickle.load(open("moderate_cols.pkl", "rb"))
    second_line_drugs = pickle.load(open("second_line_drugs.pkl", "rb"))
    final_columns = pickle.load(open("final_columns.pkl", "rb"))
    return dict(model=model, scaler=scaler, onehot_encoders=onehot_encoders,
                label_encoders=label_encoders, ordinal_encoders=ordinal_encoders,
                age_mapping=age_mapping, heavy_skew_cols=heavy_skew_cols,
                moderate_cols=moderate_cols, second_line_drugs=second_line_drugs,
                final_columns=final_columns)


artifacts = load_artifacts()

DRUG_OPTIONS = ['No', 'Down', 'Steady', 'Up']
DIAG_OPTIONS = ['Circulatory', 'Diabetes', 'Digestive', 'Genitourinary', 'Injury',
                'Musculoskeletal', 'Neoplasms', 'Other', 'Respiratory']


# ============================================================
# دالة معالجة مريض واحد - نفس منطق التدريب بالظبط
# ============================================================
def preprocess_single_patient(input_dict, a):
    df = pd.DataFrame([input_dict])

    # أدوية الخط الثاني
    present = [c for c in a['second_line_drugs'] if c in df.columns]
    df['num_second_line_meds'] = (df[present] == 'Steady').sum(axis=1)
    df['on_second_line_med'] = (df['num_second_line_meds'] > 0).astype(int)
    df = df.drop(columns=present, errors='ignore')

    # تجميع ICD-9 (المستخدم بيختار الفئة مباشرة من الفورم، فمش محتاجين نطبق الدالة هنا
    # إلا لو الفورم بياخد كود خام - هنا الفورم بياخد الفئة مباشرة)

    # العمر
    df['age'] = df['age'].map(a['age_mapping'])

    # LabelEncoder مع حماية من فئة غير معروفة
    for col, le in a['label_encoders'].items():
        val = str(df[col].iloc[0])
        df[col] = le.transform([val])[0] if val in le.classes_ else -1

    # OrdinalEncoder (الأدوية + التحاليل) - كل واحد له encoder منفصل في الـ dict
    for col, oe in a['ordinal_encoders'].items():
        df[[col]] = oe.transform(df[[col]].astype(str))

    # num_meds_changed (بعد الـ Ordinal Encoding: Down=1, Up=3)
    drug_cols = [c for c in ['metformin', 'repaglinide', 'glimepiride', 'glipizide',
                              'glyburide', 'pioglitazone', 'rosiglitazone', 'insulin'] if c in df.columns]
    df['num_meds_changed'] = (df[drug_cols].isin([1, 3])).sum(axis=1)

    # OneHotEncoder - كل عمود بيرجع df منفصل وبينضاف
    for col, ohe in a['onehot_encoders'].items():
        encoded = ohe.transform(df[[col]].astype(str)).toarray()
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(), index=df.index)
        df = pd.concat([df, encoded_df], axis=1)
        df = df.drop(columns=[col])

    # log1p على الأعمدة شديدة الالتواء
    for col in a['heavy_skew_cols']:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # RobustScaler على الأعمدة المعتدلة (تشمل شديدة الالتواء بعد الـ log)
    df[a['moderate_cols']] = a['scaler'].transform(df[a['moderate_cols']])

    # رتب الأعمدة بالظبط زي وقت التدريب
    df = df.reindex(columns=a['final_columns'], fill_value=0)

    return df


# ============================================================
# واجهة المستخدم
# ============================================================
st.title("🏥 توقع خطر إعادة دخول مريض السكري")
st.caption("أداة مساعدة للتقدير الأولي - لا تغني عن التقييم الطبي المباشر")

with st.form("patient_form"):
    st.subheader("البيانات الديموغرافية والإقامة")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("النوع", ["Female", "Male"])
        age_bucket = st.selectbox("الفئة العمرية", list(artifacts['age_mapping'].keys()), index=6)
        race = st.selectbox("العرق", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "Unknown"])
    with col2:
        time_in_hospital = st.number_input("مدة الإقامة (أيام)", 1, 30, 4)
        admission_type_id = st.number_input("نوع الدخول (ID)", min_value=1, max_value=8, value=1,
                                              help="1=Emergency, 2=Urgent, 3=Elective - راجع IDS_mapping.csv")
        discharge_disposition_id = st.number_input("جهة الخروج (ID)", min_value=1, max_value=30, value=1,
                                                      help="1=Home - راجع IDS_mapping.csv")
        admission_source_id = st.number_input("مصدر الدخول (ID)", min_value=1, max_value=25, value=1,
                                                help="راجع IDS_mapping.csv")

    st.subheader("الإجراءات والتشخيص")
    col3, col4 = st.columns(2)
    with col3:
        num_lab_procedures = st.number_input("عدد الفحوصات المعملية", 0, 150, 40)
        num_procedures = st.number_input("عدد الإجراءات", 0, 10, 1)
        num_medications = st.number_input("عدد الأدوية", 0, 100, 15)
        number_diagnoses = st.number_input("عدد التشخيصات", 1, 20, 7)
    with col4:
        number_outpatient = st.number_input("زيارات عيادات خارجية سابقة", 0, 50, 0)
        number_emergency = st.number_input("زيارات طوارئ سابقة", 0, 50, 0)
        number_inpatient = st.number_input("مرات دخول سابقة", 0, 50, 0)

    diag_1 = st.selectbox("التشخيص الأساسي", DIAG_OPTIONS)
    diag_2 = st.selectbox("التشخيص الثانوي", DIAG_OPTIONS)
    diag_3 = st.selectbox("التشخيص الإضافي", DIAG_OPTIONS)

    st.subheader("الأدوية ونتائج التحاليل")
    col5, col6 = st.columns(2)
    with col5:
        metformin = st.selectbox("Metformin", DRUG_OPTIONS)
        repaglinide = st.selectbox("Repaglinide", DRUG_OPTIONS)
        glimepiride = st.selectbox("Glimepiride", DRUG_OPTIONS)
        glipizide = st.selectbox("Glipizide", DRUG_OPTIONS)
    with col6:
        glyburide = st.selectbox("Glyburide", DRUG_OPTIONS)
        pioglitazone = st.selectbox("Pioglitazone", DRUG_OPTIONS)
        rosiglitazone = st.selectbox("Rosiglitazone", DRUG_OPTIONS)
        insulin = st.selectbox("Insulin", DRUG_OPTIONS)

    col7, col8 = st.columns(2)
    with col7:
        nateglinide = st.selectbox("Nateglinide (خط ثاني)", ["No", "Steady"])
        chlorpropamide = st.selectbox("Chlorpropamide (خط ثاني)", ["No", "Steady"])
    with col8:
        acarbose = st.selectbox("Acarbose (خط ثاني)", ["No", "Steady"])
        glyburide_metformin = st.selectbox("Glyburide-metformin (خط ثاني)", ["No", "Steady"])

    change = st.selectbox("تغيير في العلاج بهذه الزيارة؟", ["No", "Ch"])
    diabetesMed = st.selectbox("يأخذ دواء سكر حاليًا؟", ["No", "Yes"])

    submitted = st.form_submit_button("توقع خطر إعادة الدخول")

if submitted:
    patient_input = {
        'gender': gender, 'age': age_bucket, 'race': race,
        'time_in_hospital': time_in_hospital,
        'admission_type_id': admission_type_id,
        'discharge_disposition_id': discharge_disposition_id,
        'admission_source_id': admission_source_id,
        'num_lab_procedures': num_lab_procedures, 'num_procedures': num_procedures,
        'num_medications': num_medications, 'number_diagnoses': number_diagnoses,
        'number_outpatient': number_outpatient, 'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'diag_1': diag_1, 'diag_2': diag_2, 'diag_3': diag_3,
        'metformin': metformin, 'repaglinide': repaglinide, 'glimepiride': glimepiride,
        'glipizide': glipizide, 'glyburide': glyburide, 'pioglitazone': pioglitazone,
        'rosiglitazone': rosiglitazone, 'insulin': insulin,
        'nateglinide': nateglinide, 'chlorpropamide': chlorpropamide,
        'acarbose': acarbose, 'glyburide-metformin': glyburide_metformin,
        'change': change, 'diabetesMed': diabetesMed,
    }

    X_new = preprocess_single_patient(patient_input, artifacts)
    prediction = artifacts['model'].predict(X_new)[0]
    probability = artifacts['model'].predict_proba(X_new)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ خطر مرتفع نسبيًا لإعادة الدخول خلال 30 يوم")
    else:
        st.success(f"✅ خطر منخفض نسبيًا لإعادة الدخول خلال 30 يوم")

    st.metric("احتمالية إعادة الدخول خلال 30 يوم", f"{probability*100:.1f}%")
    st.progress(min(float(probability), 1.0))

    st.caption("⚠️ هذه أداة مساعدة للتقدير الأولي فقط، ولا تغني عن التقييم الطبي المباشر من الطبيب المختص.")