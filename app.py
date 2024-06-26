from functions import rag
import streamlit as st

# set initial message
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there, how can I help you"}
    ]

if "messages" in st.session_state.keys():
    # display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# get user input
user_prompt = st.chat_input()

if user_prompt is not None:
    new_user_prompt = user_prompt
    st.session_state.messages.append({"role": "user", "content": new_user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            print("User question: {}".format(new_user_prompt))
            ai_response = rag(new_user_prompt)
            st.write(ai_response)

    new_ai_message = {"role": "user", "content": ai_response}
    st.session_state.messages.append(new_ai_message)