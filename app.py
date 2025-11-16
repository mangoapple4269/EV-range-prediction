import streamlit as st
import numpy as np
import joblib
from openai import OpenAI

# Load Model
model = joblib.load("model.pkl")

# Groq Client (NOT OpenAI company, only same API format)
client = OpenAI(
    api_key=st.secrets["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

# Page Settings
st.set_page_config(
    page_title="EV Range Predictor",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ EV Range Predictor & EV Assistant Chatbot")
st.write("Predict EV range and ask questions using the AI chatbot powered by Groq (free).")

# =============================
#     EV RANGE PREDICTION
# =============================
st.subheader("🔋 EV Range Prediction")

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        top_speed = st.number_input("Top Speed (km/h)", value=120)
        torque = st.number_input("Torque (Nm)", value=200)

    with col2:
        battery_capacity = st.number_input("Battery Capacity (kWh)", value=50.0)

    if st.button("🚗 Predict EV Range"):
        X = np.array([[top_speed, battery_capacity, torque]])
        pred = model.predict(X)[0]
        st.success(f"Estimated Range: **{pred:.2f} km**")


# =============================
#        CHATBOT SECTION
# =============================
st.subheader("🤖 Chat with EV Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_msg = st.text_input("Ask anything about EVs:")

if st.button("Send"):
    if user_msg.strip():
        st.session_state.messages.append(("You", user_msg))

        # Groq Chat Completion
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are a helpful EV expert assistant."},
                {"role": "user", "content": user_msg}
            ]
        )

        bot_reply = response.choices[0].message["content"]
        st.session_state.messages.append(("Bot", bot_reply))

# Display Chat Messages
with st.container():
    for speaker, msg in st.session_state.messages:
        if speaker == "You":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 Bot:** {msg}")

