# chat-pykg/chain.py
import os
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar

import gradio as gr
from langchain import HuggingFaceHub, OpenAI
from langchain.agents import (AgentType, Tool, initialize_agent)
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.chains.base import Chain
from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT, QA_PROMPT)
from langchain.chat_models import ChatOpenAI
from langchain.memory import (ConversationBufferMemory,
                              ConversationBufferWindowMemory)
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers import SVMRetriever

from ingest import embedding_chooser
from tools import get_tools

stdouthandler = StdOutCallbackHandler()

def get_new_chain(vectorstores, vectorstore_radio, embedding_radio, model_selector, k_textbox, search_type_selector, max_tokens_textbox) -> Chain:
    if type(embedding_radio) == gr.Radio:
        embedding_radio = embedding_radio.value
    if type(vectorstore_radio) == gr.Radio:
        vectorstore_radio = vectorstore_radio.value
    embedding_function = embedding_chooser(embedding_radio)

    # Model Selection
    match model_selector:
        case 'gpt-3.5-turbo' | 'gpt-4':
            llm = ChatOpenAI(client = None, temperature=0.9, model=model_selector, verbose=True, max_tokens=int(max_tokens_textbox))
            doc_chain_llm = ChatOpenAI(client = None, streaming=True, callbacks=[stdouthandler], verbose=True, temperature=0.9, model=model_selector, max_tokens=int(max_tokens_textbox))
        case 'other':
            llm = HuggingFaceHub(client = None, repo_id="chavinlo/gpt4-x-alpaca")#, model_kwargs={"temperature":0, "max_length":64})
            doc_chain_llm = HuggingFaceHub(client = None, repo_id="chavinlo/gpt4-x-alpaca")
        case _:
            llm = ChatOpenAI(client = None, temperature=0.9, model=model_selector, verbose=True, max_tokens=int(max_tokens_textbox))
            doc_chain_llm = OpenAI(client = None, streaming=True, callbacks=[stdouthandler], verbose=True, temperature=0.9, model="gpt-3.5-turbo", max_tokens=int(max_tokens_textbox))
    
    g_api_key = os.environ.get("GOOGLE_API_KEY")
    
    tools = get_tools(g_api_key)
    # QA Chains
    for vectorstore in vectorstores:
        retriever = None
        if vectorstore_radio == 'Chroma':
            retriever = vectorstore.as_retriever(search_type=search_type_selector)
            retriever.search_kwargs = {"k":int(k_textbox)}
            if search_type_selector == 'mmr':
                retriever.search_kwargs = {"k":int(k_textbox), "fetch_k":4*int(k_textbox)}
        if vectorstore_radio == 'raw':
            if search_type_selector == 'svm':
                retriever = SVMRetriever.from_texts(vectorstore, embedding_function)
                retriever.k = int(k_textbox)
        # # QA Chain
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, verbose=True, input_key="question")
        
        tools.append(
            Tool(
            name = f'{vectorstore._collection.metadata}',
            description=f"Useful for when you need to answer questions about {vectorstore._collection.metadata}.",
            func=qa.run,
            verbose = True,
            )
        )


    # Memory
    memory = ConversationBufferMemory(memory_key="chat_history")

    # Agent
    ae = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory)#, callbacks=[stdouthandler])
    return ae



