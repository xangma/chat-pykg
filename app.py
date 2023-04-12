import datetime
import os
import gradio as gr
from abc import ABC
from typing import List, Optional, Any
import asyncio
import langchain
import chromadb
from chromadb.config import Settings
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, PythonCodeTextSplitter
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
import shutil
import random, string
from chain import get_new_chain1
from ingest import ingest_docs, CachedChroma

def randomword(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def set_chain_up(openai_api_key, model_selector, k_textbox, vectorstore, agent):
    if vectorstore == None: 
        return 'no_vectorstore'
    if vectorstore != None: 
        if model_selector in ["gpt-3.5-turbo", "gpt-4"]:
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
                qa_chain = get_new_chain1(vectorstore, model_selector, k_textbox)
                os.environ["OPENAI_API_KEY"] = ""
                return qa_chain
            else:
                return 'no_open_aikey'
        else:
            qa_chain = get_new_chain1(vectorstore, model_selector, k_textbox)
            return qa_chain

def get_vectorstore(chat_state, collection_textbox, vs_state):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = CachedChroma.from_documents_with_cache(persist_directory=".persisted_data", documents=None, embedding = embeddings, collection_name=collection_textbox)
    return vectorstore

def make_vectorstore(chat_state,collection_name, packagedocslist, vs_state):
    vectorstore = ingest_docs(collection_name, packagedocslist)
    return vectorstore

def delete_vs(chat_state, collection_textbox):
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=".persisted_data" # Optional, defaults to .chromadb/ in the current directory
    ))
    client.delete_collection(collection_textbox)

def delete_all_vs(chat_state):
    shutil.rmtree(".persisted_data")
    return "all_vs_deleted"

def get_all_vs_names(chat_state):
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=".persisted_data" # Optional, defaults to .chromadb/ in the current directory
    ))
    collection_names = [c.name for c in client.list_collections()]
    # print the collection names to the chatbot
    return collection_names, "all_collections"

def chat(inp, history, agent):
    history = history or []
    if type(agent) == str:
        if agent == 'no_open_aikey':
            history.append((inp, "Please paste your OpenAI key to use"))
            return history, history
        if agent == 'no_vectorstore':
            history.append((inp, "Please ingest some package docs to use"))
            return history, history
        if agent == 'all_collections' and inp != []:
            history.append(("", f"Current vectorstores: {inp}"))
            return history, history
        if agent == 'all_vs_deleted':
            history.append((inp, "All vectorstores deleted"))
            return history, history

    print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
    print("inp: " + inp)
    history = history or []
    output = agent({"question": inp, "chat_history": history})
    answer = output["answer"]
    history.append((inp, answer))
    print(history)
    return history, history

block = gr.Blocks(css=".gradio-container {background-color: system;}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>Package docs Assistant</center></h3>")

        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key (sk-...)",
            show_label=False,
            lines=1,
            type="password",
        )
        model_selector = gr.Dropdown(["gpt-3.5-turbo", "gpt-4", "other"], label="Model", show_label=True)
        model_selector.value = "gpt-3.5-turbo"
        k_textbox = gr.Textbox(
            placeholder="k: Number of search results to consider",
            label="Search Results k:",
            show_label=True,
            lines=1,
        )
        k_textbox.value = "10"
    chatbot = gr.Chatbot()
    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="What is this code?",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)
    with gr.Row():
        with gr.Column(scale=4):
            packagedocslist = gr.List(headers=['Package Docs URL'],row_count=5, label='Package docs URLs', show_label=True, interactive=True, max_cols=1, max_rows=5)
        with gr.Column(scale=1):
            randomname = randomword(5)
            collection_textbox = gr.Textbox(placeholder=randomname,
            label="Collection name:",
            show_label=True,
            lines=1,
        )
            collection_textbox.value = randomname
            get_vs_button = gr.Button(value="Get vectorstore", variant="secondary").style(full_width=False)
            make_vs_button = gr.Button(value="Make vectorstore", variant="secondary").style(full_width=False)
            delete_vs_button = gr.Button(value="Delete vectorstore", variant="secondary").style(full_width=False)
            delete_all_vs_button = gr.Button(value="Delete all vectorstores", variant="secondary").style(full_width=False)
            get_all_vs_names_button = gr.Button(value="Get all vectorstore names", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=[
            "What is this code and why hasn't the developer documented it?",
            "Where is this specific method in the source code and why is it broken?"
        ],
        inputs=message,
    )

    gr.HTML(
        """
    This simple application is an implementation of ChatGPT but over an external dataset.  
    The source code is split/broken down into many document objects using langchain's pythoncodetextsplitter, which apparently tries to keep whole functions etc. together. This means that each file in the source code is split into many smaller documents, and the k value is the number of documents to consider when searching for the most similar documents to the question. With gpt-3.5-turbo, k=10 seems to work well, but with gpt-4, k=20 seems to work better.  
    The model's memory is set to 5 messages, but I haven't tested with gpt-3.5-turbo yet to see if it works well. It seems to work well with gpt-4."""
    )

    gr.HTML(
        "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
    )

    history_state = gr.State()
    agent_state = gr.State()
    vs_state = gr.State()
    all_collections = gr.State()
    chat_state = gr.State()

    submit.click(chat, inputs=[message, history_state, agent_state], outputs=[chatbot, history_state])
    message.submit(chat, inputs=[message, history_state, agent_state], outputs=[chatbot, history_state])

    get_vs_button.click(get_vectorstore, inputs=[chat_state,collection_textbox, vs_state], outputs=[vs_state]).then(set_chain_up, inputs=[openai_api_key_textbox, model_selector, k_textbox, vs_state, agent_state], outputs=[agent_state])
    make_vs_button.click(make_vectorstore, inputs=[chat_state,collection_textbox, packagedocslist, vs_state], outputs=[vs_state], show_progress=True).then(set_chain_up, inputs=[openai_api_key_textbox, model_selector, k_textbox, vs_state, agent_state], outputs=[agent_state])
    delete_vs_button.click(delete_vs, inputs=[chat_state,collection_textbox], outputs=[])
    delete_all_vs_button.click(delete_all_vs, inputs=[chat_state], outputs=[chat_state]).then(chat, inputs=[all_collections, history_state, chat_state], outputs=[chatbot, history_state])
    get_all_vs_names_button.click(get_all_vs_names, inputs=[chat_state], outputs=[all_collections, chat_state]).then(chat, inputs=[all_collections, history_state, chat_state], outputs=[chatbot, history_state])

    #I need to also parse this code in the docstore so I can ask it to fix silly things like this below:
    openai_api_key_textbox.change(
        set_chain_up,
        inputs=[openai_api_key_textbox, model_selector, k_textbox, vs_state, agent_state],
        outputs=[agent_state],
    )
    model_selector.change(
        set_chain_up,
        inputs=[openai_api_key_textbox, model_selector, k_textbox, vs_state, agent_state],
        outputs=[agent_state],
    )
    k_textbox.change(
        set_chain_up,
        inputs=[openai_api_key_textbox, model_selector, k_textbox, vs_state, agent_state],
        outputs=[agent_state],
    )

block.launch(debug=True)
