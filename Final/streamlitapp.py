from llmChain import LLMChain
from vectorDB import VectorStorage  

from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage,AIMessage


# Instantiate the LLMChain with your retriever and llm type
vdb = VectorStorage()
retriever = vdb.load_db_and_get_retriever("faiss_index")  # Your retriever instantiation goes here
llm_chain = LLMChain(retriever, llm_type="vertexai", model_name="text-bison", max_output_tokens=256)


import streamlit as st




with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?" ,"ref" :""}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"] + msg["ref"])

if prompt := st.chat_input():
    #if not openai_api_key:
    #    st.info("Please add your OpenAI API key to continue.")
    #    st.stop()

    #client = llm(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt , "ref":""})
    st.chat_message("user").write(prompt)
    #response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    
    #response = llm(prompt=prompt)

    vmessages = [] 
    vmessages.append(SystemMessage(content = "bot" ))
    for msg in st.session_state.messages[1:]:
        if (msg["role"] == "assistant"):
            vmessages.append(AIMessage(content = msg["content"] ))
        else:
            vmessages.append(HumanMessage(content = msg["content"]))


    # Define inputs for the chain
    inputs = {
        "question":  vmessages[-1].content,
        "chat_history": vmessages[:-1]
    }

    # Invoke the LLMChain
    result = llm_chain.invoke(inputs)
    print(result)
    msg = result['answer']
    print(msg)
    ref = ''
    ref = ref + '\n'
    for doc in result['docs']:
            ref = ref + " \n Reference [link](%s) " % doc.metadata['source']   
    st.chat_message("assistant").markdown(msg + ref)
    st.session_state.messages.append({"role": "assistant", "content": msg , "ref" : ref })
    