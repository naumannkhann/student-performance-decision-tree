
import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_files():
    model = joblib.load("student_performance_model.pkl")
    extra_encoder = joblib.load("extra_support_encoder.pkl")
    family_encoder = joblib.load("family_support_encoder.pkl")
    higher_encoder = joblib.load("higher_education_encoder.pkl")
    internet_encoder = joblib.load("internet_access_encoder.pkl")
    result_encoder = joblib.load("student_result_encoder.pkl")

    return (
        model,
        extra_encoder,
        family_encoder,
        higher_encoder,
        internet_encoder,
        result_encoder
    )

(
    model,
    extra_encoder,
    family_encoder,
    higher_encoder,
    internet_encoder,
    result_encoder
) = load_files()

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Prediction System")

st.write(
    "Enter the student's academic and lifestyle details "
    "to predict whether the student is likely to pass or fail."
)

with st.form("student_form"):

    weekly_studytime = st.number_input(
        "Weekly Study Time",
        min_value=1,
        max_value=4,
        value=2,
        step=1
    )

    failures = st.number_input(
        "Previous Failures",
        min_value=0,
        max_value=4,
        value=0,
        step=1
    )

    extra_edu_supp = st.selectbox(
        "Extra Educational Support",
        extra_encoder.classes_.tolist()
    )

    family_edu_supp = st.selectbox(
        "Family Educational Support",
        family_encoder.classes_.tolist()
    )

    interested_higher = st.selectbox(
        "Interested in Higher Education",
        higher_encoder.classes_.tolist()
    )

    internet_access = st.selectbox(
        "Internet Access",
        internet_encoder.classes_.tolist()
    )

    freetime_after_school = st.number_input(
        "Free Time After School",
        min_value=1,
        max_value=5,
        value=3,
        step=1
    )

    goout_with_friends = st.number_input(
        "Going Out With Friends",
        min_value=1,
        max_value=5,
        value=3,
        step=1
    )

    absences = st.number_input(
        "Absences",
        min_value=0,
        value=0,
        step=1
    )

    G1 = st.number_input(
        "First Period Grade (G1)",
        min_value=0,
        max_value=20,
        value=10,
        step=1
    )

    G2 = st.number_input(
        "Second Period Grade (G2)",
        min_value=0,
        max_value=20,
        value=10,
        step=1
    )

    submitted = st.form_submit_button("Predict Student Result")

if submitted:

    try:
        input_data = pd.DataFrame([{
            "weekly_studytime": weekly_studytime,
            "failures": failures,
            "extra_edu_supp": extra_encoder.transform(
                [extra_edu_supp]
            )[0],
            "family_edu_supp": family_encoder.transform(
                [family_edu_supp]
            )[0],
            "Interested_in_higher_edu": higher_encoder.transform(
                [interested_higher]
            )[0],
            "internet_access": internet_encoder.transform(
                [internet_access]
            )[0],
            "freetime_after_school": freetime_after_school,
            "goout_with_friends": goout_with_friends,
            "absences": absences,
            "G1": G1,
            "G2": G2
        }])

        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[0]

        result = result_encoder.inverse_transform(
            prediction
        )[0]

        classes = list(result_encoder.classes_)
        fail_index = classes.index("Fail")
        pass_index = classes.index("Pass")

        if result == "Pass":
            st.success(f"Student Result: {result}")
        else:
            st.error(f"Student Result: {result}")

        st.write(
            f"Pass Probability: "
            f"{probabilities[pass_index] * 100:.2f}%"
        )

        st.write(
            f"Fail Probability: "
            f"{probabilities[fail_index] * 100:.2f}%"
        )

    except Exception as error:
        st.error("Prediction could not be completed.")
        st.exception(error)
