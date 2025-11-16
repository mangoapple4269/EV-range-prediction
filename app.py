import streamlit as st
import numpy as np
import joblib
from openai import OpenAI

# Load Model
model = joblib.load("model.pkl")

# Groq Client
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

st.title("EV Range Predictor & EV Assistant Chatbot")
st.write("EV range prediction and chatbot assistance.")

# =============================
#     EV RANGE PREDICTION
# =============================
st.subheader("EV Range Prediction")

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        top_speed = st.number_input("Top Speed (km/h)", value=120)
        torque = st.number_input("Torque (Nm)", value=200)

    with col2:
        battery_capacity = st.number_input("Battery Capacity (kWh)", value=50.0)

    if st.button("Predict Range"):
        X = np.array([[top_speed, battery_capacity, torque]])
        pred = model.predict(X)[0]
        st.success(f"Estimated Range: {pred:.2f} km")

# =============================
#        CHATBOT SECTION
# =============================
st.subheader("Chat with EV Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_msg = st.text_input("Ask anything about EVs:")

if st.button("Send"):
    if user_msg.strip():
        st.session_state.messages.append(("You", user_msg))

        try:
            response = client.chat.completions.create(
                model="llama3-8b",
                messages=[
                    {"role": "system", "content": "You are a helpful EV expert assistant."},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.5,
                max_tokens=200
            )

            bot_reply = response.choices[0].message["content"]
            st.session_state.messages.append(("Bot", bot_reply))

        except Exception as e:
            st.error(f"Chatbot Error: {e}")

with st.container():
    for speaker, msg in st.session_state.messages:
        st.markdown(f"**{speaker}:** {msg}")
