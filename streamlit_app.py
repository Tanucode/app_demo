import streamlit as st

st.title('Streamlit App Example')
user_input = st.text_input("Enter some text:")
st.write(f"Predicted next word: {user_input}")
