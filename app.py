import requests
import streamlit as st

st.title("Smart Lost & Found")
st.write("Describe your lost item and we will find the best matches.")

text = st.text_input("Describe your lost item")

if st.button("Find matches"):
    if text.strip() == "":
        st.warning("Please type a description first.")
    else:
        try:
            response = requests.post("http://127.0.0.1:8000/match",
                                     json={"description": text})
            matches = response.json()["matches"]
            for m in matches:
                st.subheader(f"{m['description']}")
                st.write(f"Found at: {m['location']}  |  Match score: {m['score']}")
        except requests.exceptions.ConnectionError:
            st.error("API is not running. Start it with: uvicorn api:app --reload")