import json
import os
import pathlib
from typing import Dict, List, Tuple
from langchain.chains.base import Chain
import os
import langchain
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from langchain import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.llm import LLMChain
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT

from abc import ABC
from typing import List, Optional, Any

import chromadb
from langchain.vectorstores import Chroma

def get_new_chain1(vectorstore, model_selector, k_textbox) -> Chain:
    max_tokens_dict = {'gpt-4': 2000, 'gpt-3.5-turbo': 1000}

    # These templates aren't used for the moment.
    _eg_template = """## Example:

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question: {answer}"""
    _prefix = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. You should assume that the question is related to PyCBC."""
    _suffix = """## Example:

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    template = """You are an AI assistant for various open source libraries.
You are given the following extracted parts of a long document and a question. Provide a conversational answer to the question.
You should only use hyperlinks that are explicitly listed as a source in the context. Do NOT make up a hyperlink that is not listed.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about the package documentation, politely inform them that you are tuned to only answer questions about the package documentationz.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""

    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    if model_selector in ['gpt-4', 'gpt-3.5-turbo']:
        llm = ChatOpenAI(client = None, temperature=0.7, model_name=model_selector)
        doc_chain_llm = ChatOpenAI(client = None, streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0.7, model_name=model_selector, max_tokens=1000)
    if model_selector == 'other':
        llm = HuggingFaceHub(repo_id="chavinlo/gpt4-x-alpaca")#, model_kwargs={"temperature":0, "max_length":64})
        doc_chain_llm = HuggingFaceHub(repo_id="chavinlo/gpt4-x-alpaca")
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(doc_chain_llm, chain_type="stuff", prompt=QA_PROMPT)

    # memory = ConversationKGMemory(llm=llm, input_key="question", output_key="answer")
    memory = ConversationBufferWindowMemory(input_key="question", output_key="answer", k=5)
    retriever = vectorstore.as_retriever()
    retriever.search_kwargs = {"k": int(k_textbox)}
    qa = ConversationalRetrievalChain(
        retriever=retriever, memory=memory, combine_docs_chain=doc_chain, question_generator=question_generator)

    return qa