# chatpykg/chain.py
import os
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar

import gradio as gr
from langchain import HuggingFaceHub, OpenAI
from langchain.agents import (AgentType, Tool, initialize_agent, ZeroShotAgent)
from langchain.callbacks import StdOutCallbackHandler, OpenAICallbackHandler
from langchain.chains import RetrievalQA
from langchain.chains.base import Chain
from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT, QA_PROMPT)
from langchain.chat_models import ChatOpenAI
from langchain.memory import (ConversationBufferMemory,
                              ConversationBufferWindowMemory)
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers import SVMRetriever
from langchain.callbacks.base import BaseCallbackHandler
from ingest import embedding_chooser
from tools import get_tools

callback_handler = StdOutCallbackHandler()
oai_callback_handler = OpenAICallbackHandler()

class CustomCallbackHandler(BaseCallbackHandler):
    """Custom Callback Handler that prints the whole prompt to std out."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the whole prompt."""
        for prompt in prompts:
            print(prompt)

def get_new_chain(vectorstores, vectorstore_radio, embedding_radio, model_selector, k_textbox, search_type_selector, max_tokens_textbox) -> Chain:
    if type(embedding_radio) == gr.Radio:
        embedding_radio = embedding_radio.value
    if type(vectorstore_radio) == gr.Radio:
        vectorstore_radio = vectorstore_radio.value
    embedding_function = embedding_chooser(embedding_radio)

    # Model Selection
    match model_selector:
        case 'gpt-3.5-turbo' | 'gpt-4':
            agent_llm = ChatOpenAI(client = None, temperature=0.8, model=model_selector, verbose=True, max_tokens=int(max_tokens_textbox), callbacks=[CustomCallbackHandler()])
            doc_llm = ChatOpenAI(client = None, temperature=0.0, model=model_selector, verbose=True, max_tokens=int(max_tokens_textbox),callbacks=[CustomCallbackHandler()])
        case 'other':
            agent_llm = HuggingFaceHub(client = None, repo_id="chavinlo/gpt4-x-alpaca")
            doc_llm = HuggingFaceHub(client = None, repo_id="chavinlo/gpt4-x-alpaca")#, model_kwargs={"temperature":0, "max_length":64})
        case _:
            agent_llm = ChatOpenAI(client = None, temperature=0.8, model=model_selector, verbose=True, max_tokens=int(max_tokens_textbox))
            doc_llm = ChatOpenAI(client = None, temperature=0.9, model=model_selector, verbose=True, max_tokens=int(max_tokens_textbox))
    
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
        qa = RetrievalQA.from_chain_type(llm=doc_llm, retriever=retriever, verbose=True, input_key="question")
        # qa.prompt = QA_PROMPT
        
        tools.append(
            Tool(
            name = f'{vectorstore._collection.metadata}',
            description=f"Useful for when you need to answer questions about code related to the {vectorstore._collection.metadata} package. Inputs should be a fully formed question.",
            func=qa.run,
            verbose = True,
            )
        )


    # Prompt

    PREFIX = """You are an agent called chatpykg and are an AI agent coded in python using langchain for the backend and gradio for the frontend. You are helpful for answering questions about programming with various packages and libraries. 
    Answer the following questions as best you can. If you are asked about a particular package, you can use the tools below to help you answer the question.
    You have access to the following tools:"""
    FORMAT_INSTRUCTIONS = """Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question"""
    SUFFIX = """Begin!

    Chat History:
    {chat_history}
    Question: {input}
    Thought:{agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=PREFIX,
        format_instructions=FORMAT_INSTRUCTIONS,
        suffix=SUFFIX,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )

    # Memory
    memory = ConversationBufferMemory(memory_key="chat_history")

    # Agent
    ae = initialize_agent(agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                          tools=tools,
                          llm=agent_llm,
                          verbose=True,
                          memory=memory,
                          callbacks=[callback_handler, oai_callback_handler]   
                        )
    return ae



