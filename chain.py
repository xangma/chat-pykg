# chat-pykg/chain.py
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar
import gradio as gr
from langchain.agents import Tool
from langchain import HuggingFaceHub
from langchain.agents import AgentType, initialize_agent, AgentExecutor, ConversationalChatAgent
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT, QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers import SVMRetriever
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.experimental import AutoGPT
from tools import get_tools
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from ingest import embedding_chooser

def get_new_chain(vectorstores, vectorstore_radio, embedding_radio, model_selector, k_textbox, search_type_selector, max_tokens_textbox) -> Chain:
    if type(embedding_radio) == gr.Radio:
        embedding_radio = embedding_radio.value
    if type(vectorstore_radio) == gr.Radio:
        vectorstore_radio = vectorstore_radio.value
    embedding_function = embedding_chooser(embedding_radio)
    # Prompt Templates
    qa_template = """You are called chat-pykg and are an AI assistant coded in python using langchain and gradio. You are very helpful for answering questions about programming with various open source packages and libraries.
                You are given snippets of code and information in the Context below, as well as a Question to give a Helpful answer to. 
                Due to data size limitations, the snippets of code in the Context have been specifically filtered/selected for their relevance from a document store containing code from one or many packages and libraries.
                Each of the code snippets is marked with '# source: package/filename' so you can attempt to establish where they are located in their package structure and gain more understanding of the code.
                Please provide a helpful answer in markdown to the Question.
                Do not make up any hyperlinks that are not in the Context.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.

                =========
                Context:{context}
                =========
                Question: {input}
                Helpful answer:"""
    QA_PROMPT.template = qa_template
    condense_question_template = """Given the following conversation and a Follow Up Input, rephrase the Follow Up Input to be a Standalone question.
    The Standalone question will be used for retrieving relevant source code and information from a document store, where each document is marked with '# source: package/filename'.
    Therefore, in your Standalone question you must try to include references to related code or sources that have been mentioned in the Follow Up Input or Chat History.
    =========
    Chat History:
    {chat_history}
    =========
    Follow Up Input: {input}
    Standalone question in markdown:"""
    CONDENSE_QUESTION_PROMPT.template = condense_question_template
    
    # Model Selection
    match model_selector:
        case ['gpt-4', 'gpt-3.5-turbo']:
            llm = ChatOpenAI(client = None, temperature=0.9, model_name=model_selector)
            doc_chain_llm = ChatOpenAI(client = None, streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0.9, model_name=model_selector, max_tokens=int(max_tokens_textbox))
        case ['other']:
            llm = HuggingFaceHub(client = None, repo_id="chavinlo/gpt4-x-alpaca")#, model_kwargs={"temperature":0, "max_length":64})
            doc_chain_llm = HuggingFaceHub(client = None, repo_id="chavinlo/gpt4-x-alpaca")
        case _:
            llm = ChatOpenAI(client = None, temperature=0.9, model_name="gpt-3.5-turbo")
            doc_chain_llm = ChatOpenAI(client = None, streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0.9, model_name="gpt-3.5-turbo", max_tokens=int(max_tokens_textbox))
    
    # Document Chain
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(doc_chain_llm, chain_type="stuff", prompt=QA_PROMPT)#, document_prompt = PromptTemplate(input_variables=["source", "page_content"], template="{source}\n{page_content}"))
    

    tools = get_tools()
    
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
        # QA Chain
        qa = ConversationalRetrievalChain(
            retriever=retriever, combine_docs_chain=doc_chain, question_generator=question_generator, verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
        
        tools.append(
            Tool(
            name = f'{vectorstore._collection.metadata} code/package QA Tool',
            func = qa.run,
            description=f"useful for when you need to answer questions about the {vectorstore._collection.metadata} code/package. Input should be a fully formed question."
            )
        )


    # Memory
    memory = ConversationBufferWindowMemory(input_key="input", output_key="output", k=5)

    # if embedding_radio == 'OpenAI':
    #     embedding_size = 1536
    # elif embedding_radio == 'HuggingFace':
    #     embedding_size = embedding_function.client.get_sentence_embedding_dimension()
    # else:
    #     embedding_size = 768
    # index = faiss.IndexFlatL2(embedding_size)
    # memory_vectorstore = FAISS(embedding_function.embed_query, index, InMemoryDocstore({}), {})
    # agent = AutoGPT.from_llm_and_tools(
    # ai_name="chat-pykg",
    # ai_role="Assistant",
    # tools=tools,
    # llm=ChatOpenAI(client = None, temperature=0),
    # memory=memory_vectorstore.as_retriever()
    # )
    # # Set verbose to be true
    # agent.chain.verbose = True

    agent_old = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

    llmch = LLMChain(llm=llm, prompt=QA_PROMPT)
    agent = ConversationalChatAgent(llm_chain = llmch)
    agente = AgentExecutor.from_agent_and_tools(agent, tools, memory=memory, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    agente.input_keys = ["input", "chat_history"]
    return agente

