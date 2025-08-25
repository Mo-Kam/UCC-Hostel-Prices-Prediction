import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# Load trained model and sample input
# =========================

model = joblib.load('Model/best_hostel_rent_model.pkl')
sample_input = joblib.load('Model/sample_input.pkl')


# =========================
# Streamlit Page Config
# =========================
st.set_page_config(page_title="UCC Hostel Rent Predictor", layout="centered")

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Home", "Prediction"])

# =========================
# Home Page
# =========================
if page == "Home":
    st.title("🏠 University of Cape Coast Hostel Rent Prediction")
    st.markdown("""
    This app predicts **hostel annual rent prices** at the University of Cape Coast using 
    advanced machine learning models.  
    It was developed as part of a postgraduate research project.  

    ### Author  
    **Mohammed Kamalidin**  
    MSc. Data Management and Analysis  

    ---
    Use the **Prediction** page from the sidebar to try it out!
    """)

# =========================
# Prediction Page
# =========================
elif page == "Prediction":
    st.title("🔮 Hostel Rent Prediction")

    st.markdown("Fill in the details below to predict the hostel rent (in GHS).")

    # ---- Input fields ----
    room_size = st.number_input("Room Size (sqm)", min_value=5.0, max_value=50.0, value=20.0)
    distance = st.number_input("Distance to Campus (minutes)", min_value=1.0, max_value=120.0, value=15.0)
    roommates = st.number_input("Number of Roommates", min_value=1, max_value=6, value=2)
    amenities = st.number_input("Number of Amenities", min_value=0, max_value=15, value=5)

    avg_rent_nearby = st.number_input("Average Rent Nearby (GHS)", min_value=500.0, max_value=10000.0, value=3000.0)
    required_deposit = st.number_input("Required Deposit (GHS)", min_value=0.0, max_value=10000.0, value=500.0)
    recent_rent_increase = st.number_input("Recent Rent Increase (GHS)", min_value=0.0, max_value=5000.0, value=200.0)

    gender = st.selectbox("Gender", ["Male","Female"])
    age_group = st.selectbox("Age Group", ["18-20","21-23","24+"])
    study_level = st.selectbox("Level of Study", ["Undergraduate","Postgraduate"])
    lecture_location = st.text_input("Lecture Location (e.g., Science, Business)")
    faculty = st.text_input("Faculty")
    offcampus_duration = st.selectbox("Years Off Campus", ["<1 year","1-2 years","3+ years"])
    room_category = st.selectbox("Room Category", ["Standard","Deluxe","Suite"])
    hostel_location = st.text_input("Hostel Location (e.g., Amamoma, Apewosika)")
    commute_mode = st.selectbox("Commute Mode", ["Walking","Shuttle","Taxi","Bike"])

    # Boolean features (Yes/No)
    yes_no_cols = [
        "includes_water","includes_electricity","includes_waste_disposal",
        "has_running_water","has_extra_storage","has_wifi_internet","has_study_area",
        "has_security_services","has_generator_backup_power","furnished_bed",
        "furnished_table","furnished_chairs","has_access_controls","has_janitorial_services"
    ]

    bool_inputs = {}
    st.markdown("### Amenities (Yes/No)")
    for col in yes_no_cols:
        bool_inputs[col] = 1 if st.checkbox(col.replace("_"," ").title()) else 0

    # ---- Prediction ----
    if st.button("Predict Rent"):
        # Prepare input as dataframe
        input_dict = {
            "room_size_sqm": room_size,
            "distance_minutes": distance,
            "num_roommates": roommates,
            "num_amenities": amenities,
            "log_avg_rent_nearby": np.log1p(avg_rent_nearby),
            "log_required_deposit": np.log1p(required_deposit),
            "log_recent_rent_increase": np.log1p(recent_rent_increase),
            "gender": gender,
            "age_group": age_group,
            "level_of_study": study_level,
            "lecture_location": lecture_location,
            "faculty": faculty,
            "off_campus_duration": offcampus_duration,
            "room_category": room_category,
            "hostel_location": hostel_location,
            "commute_mode": commute_mode
        }
        input_dict.update(bool_inputs)

        input_df = pd.DataFrame([input_dict])

        # Predict (model expects log target → convert back with expm1)
        pred_log = model.predict(input_df)[0]
        pred_rent = np.expm1(pred_log)

        st.success(f"Predicted Annual Rent: **GHS {pred_rent:,.2f}**")
