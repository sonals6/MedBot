import streamlit as st

def main():
    st.title("MedBot")

    prompt = st.chat_input("Enter prompt:")

    if prompt:
        st.chat_message('user').markdown(prompt)

        response = "Hi! I am MedBot"
        st.chat_message('assistant').markdown(response)

if __name__ == "__main__":
    main()