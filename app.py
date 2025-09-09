import streamlit as st
import pandas as pd
import subprocess
import os

USERS_FILE = "users.csv"
ATTENDANCE_FILE = "attendance.csv"

# Ensure files exist
if not os.path.exists(USERS_FILE):
    pd.DataFrame(columns=["ID", "Name"]).to_csv(USERS_FILE, index=False)
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["ID", "Name", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

def load_users():
    return pd.read_csv(USERS_FILE)

def load_attendance():
    return pd.read_csv(ATTENDANCE_FILE)

def save_user(user_id, user_name):
    df = load_users()
    if str(user_id) in df["ID"].astype(str).values:
        st.warning("⚠️ ID already exists!")
        return
    new_entry = pd.DataFrame([[user_id, user_name]], columns=["ID", "Name"])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(USERS_FILE, index=False)

def delete_user(user_id):
    df = load_users()
    df = df[df["ID"].astype(str) != str(user_id)]
    df.to_csv(USERS_FILE, index=False)

st.title("📸 Face Attendance System")

# Register student
st.subheader("➕ Register Student")
user_id = st.text_input("Enter Student ID")
user_name = st.text_input("Enter Student Name")
if st.button("Save Student"):
    if user_id and user_name:
        save_user(user_id, user_name)
        st.success(f"✅ {user_name} (ID: {user_id}) saved successfully!")
    else:
        st.warning("⚠️ Enter both ID and Name!")

# Capture images & auto-train
if st.button("📷 Capture Student Images"):
    if user_id and user_name:
        subprocess.run(["python", "capture.py", user_id, user_name])
        subprocess.run(["python", "trainer.py"])
        st.success("✅ Images captured and model trained!")
    else:
        st.warning("⚠️ Enter Student ID and Name first!")

# Registered students
st.subheader("👥 Registered Students")
users_df = load_users()
st.dataframe(users_df)

# Delete user
st.subheader("🗑️ Delete Student")
if not users_df.empty:
    selected_id = st.selectbox("Select ID to Delete", users_df["ID"])
    if st.button("Delete Student"):
        delete_user(selected_id)
        st.success(f"❌ Student with ID {selected_id} deleted.")
        users_df = load_users()
        st.dataframe(users_df)

# Take attendance
st.subheader("✅ Take Attendance")
if st.button("Start Attendance"):
    # Run detector.py
    process = subprocess.Popen(["python", "detector.py"], stdout=subprocess.PIPE, text=True)
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line and line != "Unknown":
            st.success(f"Attendance marked: {line}")
        elif line == "Unknown":
            st.warning("Unknown person detected")

    # After attendance, reload registered students and attendance
    st.subheader("👥 Registered Students (Updated)")
    users_df = load_users()
    st.dataframe(users_df)

    st.subheader("📜 Attendance Records")
    attendance_df = load_attendance()
    st.dataframe(attendance_df)
    st.download_button("⬇️ Download Attendance CSV", 
                       data=attendance_df.to_csv(index=False), 
                       file_name="attendance.csv")
