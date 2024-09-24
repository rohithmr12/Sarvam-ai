import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.title("Document Query System")

uploaded_file = "iesc111.pdf"

query = st.text_input("Enter your query:")

if st.button("Submit Query"):
    if query:
        payload = json.dumps({"query": query})
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{API_URL}/query", data=payload, headers=headers)
        if response.status_code == 200:
            result = response.json()["response"]
            st.write("Response:", result)
        else:
            st.error(f"Error querying document: {response.text}")
    else:
        st.warning("Please enter a query.")