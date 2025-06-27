# app.py

import streamlit as st
import requests
import time
from flask_api import start_flask_server  # You’ll run this separately

# Splash Screen
def splash_screen():
    st.image("static/logo.png", width=300)  # Add your logo to static/logo.png
    st.write("Welcome to AI Resume Sculptor")
    time.sleep(1)
    st.experimental_rerun()

# Login Page
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        st.session_state["logged_in"] = True
        st.experimental_rerun()

    st.write("----")
    st.button("Continue with Google (Simulated)")

# Profile Creation Page
def profile_page():
    st.title("Create Your Profile")

    name = st.text_input("Full Name")
    dob = st.date_input("Date of Birth")
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    email = st.text_input("Email")
    permanent_address = st.text_area("Permanent Address")
    temporary_address = st.text_area("Temporary Address (Optional)")
    city = st.text_input("City")
    state = st.text_input("State")
    phone = st.text_input("Phone Number")
    qualification = st.text_input("Qualification")
    mother_tongue = st.text_input("Mother Tongue")
    languages_known = st.text_input("Languages Known (comma separated)")

    if st.button("Save Profile"):
        st.success("Profile Saved")
        st.session_state["profile_done"] = True
        st.experimental_rerun()

# Company Registration Page
def company_registration():
    st.title("Register Company")
    name = st.text_input("Company Name")
    location = st.text_input("Current Location")
    branch = st.text_input("Branch (Optional)")
    industry = st.text_input("Industry Type")

    if st.button("Register Company"):
        st.success("Company Registered")

# Dashboard
def dashboard():
    st.title("Dashboard")

    st.subheader("Upload Resume")
    resume = st.file_uploader("Upload your resume (PDF, DOCX, PNG)", type=['pdf', 'docx', 'png'])
    if resume is not None:
        st.success("Resume Uploaded")
        st.button("Analyze Resume")

    if st.button("Run ATS Filter"):
        st.write("Running ATS Filter...")

    if st.button("Run Soft Signal Analyzer"):
        st.write("Analyzing soft signals...")

    if st.button("Detect HR Rejection Reasons"):
        st.write("Checking possible rejection reasons...")

    if st.button("Resume Rewrite Assistant"):
        st.write("Suggesting improvements...")

    if st.button("Job Match Engine"):
        st.write("Matching with jobs...")

    if st.button("Mock HR Interview"):
        st.write("Starting mock interview session...")

    if st.button("Open Laks Chatbot"):
        st.write("Launching Laks...")

    if st.button("Interview Scheduler"):
        st.write("Schedule your interview...")

    if st.button("Interview Archive"):
        st.write("View past interviews...")

    if st.button("Feedback System"):
        st.write("See company feedback...")

# Main Routing Logic
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "profile_done" not in st.session_state:
        st.session_state["profile_done"] = False

    # Splash once, then clear
    if "splash_done" not in st.session_state:
        splash_screen()
        st.session_state["splash_done"] = True
    elif not st.session_state["logged_in"]:
        login_page()
    elif not st.session_state["profile_done"]:
        profile_page()
    else:
        dashboard()

if __name__ == '__main__':
    main()
