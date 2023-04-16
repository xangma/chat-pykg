# chat-pykg/app.py
import datetime
import logging
import os
import random
import shutil
import string
import sys
from pathlib import Path
import numpy as np

import chromadb
import gradio as gr
from chromadb.config import Settings
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import SVMRetriever
from chain import get_new_chain1
from ingest import embedding_chooser, ingest_docs
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class LogTextboxHandler(logging.StreamHandler):
    def __init__(self, textbox):
        super().__init__()
        self.textbox = textbox

    def emit(self, record):
        log_entry = self.format(record)
        self.textbox.value += f"{log_entry}\n"

def toggle_log_textbox(log_textbox_state):
    toggle_visibility = not log_textbox_state
    log_textbox_state = not log_textbox_state
    return log_textbox_state,gr.update(visible=toggle_visibility)

def update_textbox(full_log):
    return gr.update(value=full_log)

def update_radio(radio):
    return gr.Radio.update(value=radio)

def change_tab():
    return gr.Tabs.update(selected=0)

def update_checkboxgroup(all_collections_state):
    new_options = [i for i in all_collections_state]
    return gr.CheckboxGroup.update(choices=new_options)

def update_log_textbox(full_log):
    return gr.Textbox.update(value=full_log)

def destroy_state(state):
    state = None
    return state

def clear_chat(chatbot, history):
    return [], []

def merge_collections(collection_load_names, vs_state, k_textbox, search_type_selector, vectorstore_radio, embedding_radio): 
    if type(embedding_radio) == gr.Radio:
        embedding_radio = embedding_radio.value
    persist_directory = os.path.join(".persisted_data", embedding_radio.replace(' ','_'))
    persist_directory_raw = Path('.persisted_data_raw')
    embedding_function = embedding_chooser(embedding_radio)
    merged_documents = [] 
    merged_embeddings = []
    merged_vectorstore = None
    if vectorstore_radio == 'Chroma':
        for collection_name in collection_load_names: 
            chroma_obj_get = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory,
                anonymized_telemetry = True
            ))
            if collection_name == '': 
                continue
            collection_obj = chroma_obj_get.get_collection(collection_name, embedding_function=embedding_function)
            collection = collection_obj.get(include=["metadatas", "documents", "embeddings"])
            for i in range(len(collection['documents'])):
                merged_documents.append(Document(page_content=collection['documents'][i], metadata = collection['metadatas'][i]))
                merged_embeddings.append(collection['embeddings'][i])
        merged_vectorstore = Chroma(collection_name="temp", embedding_function=embedding_function)
        merged_vectorstore.add_documents(documents=merged_documents, embeddings=merged_embeddings)
    if vectorstore_radio == 'raw':
        merged_vectorstore = []
        for collection_name in collection_load_names: 
            if collection_name == '':
                continue
            collection_path = persist_directory_raw / collection_name
            docarr = np.load(collection_path.as_posix() +'.npy', allow_pickle=True)
            merged_vectorstore.extend(docarr.tolist())
            # read every line and append to texts
            # for f in os.listdir(collection_path):
            #     with open(os.path.join(collection_path, f), "r") as f:
            #         merged_vectorstore.append(f.readlines())
    return merged_vectorstore

def set_chain_up(openai_api_key, model_selector, k_textbox, search_type_selector, max_tokens_textbox, vectorstore_radio, vectorstore, agent):
    if not agent or type(agent) == str: 
        if vectorstore != None:
            if model_selector in ["gpt-3.5-turbo", "gpt-4"]:
                if openai_api_key:
                    os.environ["OPENAI_API_KEY"] = openai_api_key
                    qa_chain = get_new_chain1(vectorstore, vectorstore_radio, model_selector, k_textbox, search_type_selector, max_tokens_textbox)
                    os.environ["OPENAI_API_KEY"] = ""
                    return qa_chain
                else:
                    return 'no_open_aikey'
            else:
                qa_chain = get_new_chain1(vectorstore, vectorstore_radio, model_selector, k_textbox, search_type_selector, max_tokens_textbox)
                return qa_chain
        else:
            return 'no_vectorstore'
    else:
        return agent

def delete_collection(all_collections_state, collections_viewer, select_vectorstore_radio, embedding_radio):
    if type(embedding_radio) == gr.Radio:
        embedding_radio = embedding_radio.value
    persist_directory = os.path.join(".persisted_data", embedding_radio.replace(' ','_'))
    persist_directory_raw = Path('.persisted_data_raw')
    if select_vectorstore_radio == 'Chroma':
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory # Optional, defaults to .chromadb/ in the current directory
        ))
        for collection in collections_viewer:
            try:
                client.delete_collection(collection)
                all_collections_state.remove(collection)
                collections_viewer.remove(collection)
            except Exception as e:
                logging.error(e)
    if select_vectorstore_radio == 'raw':
        for collection in collections_viewer:
            try:
                os.remove(os.path.join(persist_directory_raw.as_posix(), collection+'.npy' ))
                all_collections_state.remove(collection)
                collections_viewer.remove(collection)
            except Exception as e:
                logging.error(e)
    return all_collections_state, collections_viewer

def delete_all_collections(all_collections_state, select_vectorstore_radio, embedding_radio):
    if type(embedding_radio) == gr.Radio:
        embedding_radio = embedding_radio.value
    persist_directory = os.path.join(".persisted_data", embedding_radio.replace(' ','_'))
    persist_directory_raw = Path('.persisted_data_raw')
    if select_vectorstore_radio == 'Chroma':
        shutil.rmtree(persist_directory)
    if select_vectorstore_radio == 'raw':
        shutil.rmtree(persist_directory_raw)
    return []

def list_collections(all_collections_state, select_vectorstore_radio, embedding_radio):
    if type(embedding_radio) == gr.Radio:
        embedding_radio = embedding_radio.value
    persist_directory = os.path.join(".persisted_data", embedding_radio.replace(' ','_'))
    persist_directory_raw = Path('.persisted_data_raw')
    if select_vectorstore_radio == 'Chroma':
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory # Optional, defaults to .chromadb/ in the current directory
        ))
        collection_names = [[c.name][0] for c in client.list_collections()]
        return collection_names
    if select_vectorstore_radio == 'raw':
        if os.path.exists(persist_directory_raw):
            return [f.name.split('.npy')[0] for f in os.scandir(persist_directory_raw)]
    return []

def chat(inp, history, agent):
    history = history or []
    if type(agent) == str:
        if agent == 'no_open_aikey':
            history.append((inp, "Please paste your OpenAI key to use"))
            return history, history
        if agent == 'no_vectorstore':
            history.append((inp, "Please ingest some package docs to use"))
            return history, history
    else:
        print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
        print("inp: " + inp)
        history = history or []
        output = agent({"question": inp, "chat_history": history})
        answer = output["answer"]
        history.append((inp, answer))
        print(history)
    return history, history

block = gr.Blocks(title = "chat-pykg", analytics_enabled = False, css=".gradio-container {background-color: system;}")

with block:
    gr.Markdown("<h1><center>chat-pykg</center></h1>")
    with gr.Tabs() as tabs:
        with gr.TabItem("Chat", id=0):
            with gr.Row():
                openai_api_key_textbox = gr.Textbox(
                    placeholder="Paste your OpenAI API key (sk-...)",
                    show_label=False,
                    lines=1,
                    type="password",
                )
                model_selector = gr.Dropdown(
                    choices=["gpt-3.5-turbo", "gpt-4", "other"],
                    label="Model",
                    show_label=True,
                    value = "gpt-3.5-turbo"
                )
                k_textbox = gr.Textbox(
                    placeholder="k: Number of search results to consider",
                    label="Search Results k:",
                    show_label=True,
                    lines=1,
                    value="20",
                )
                search_type_selector = gr.Dropdown(
                    choices=["similarity", "mmr", "svm"],
                    label="Search Type",
                    show_label=True,
                    value = "similarity"
                )
                max_tokens_textbox = gr.Textbox(
                    placeholder="max_tokens: Maximum number of tokens to generate",
                    label="max_tokens",
                    show_label=True,
                    lines=1,
                    value="1000",
                )
            chatbot = gr.Chatbot()
            with gr.Row():
                clear_btn = gr.Button("Clear Chat", variant="secondary").style(full_width=False)
                message = gr.Textbox(
                    label="What's your question?",
                    placeholder="What is this code?",
                    lines=1,
                )
                submit = gr.Button(value="Send").style(full_width=False)
            gr.Examples(
                examples=[
                    "What does this code do?",
                    "I want to change the chat-pykg app to have a log viewer, where the user can see what python is doing in the background. How could I do that?",
                    "Hello, I want to allow chat-pykg to search the internet before answering, can you help me change the code to do that? Thanks.",
                ],
                inputs=message,
            )

            gr.HTML(
                """
            This simple application is an implementation of ChatGPT but over an external dataset.  
            The source code is split/broken down into many document objects using langchain's pythoncodetextsplitter, which apparently tries to keep whole functions etc. together. This means that each file in the source code is split into many smaller documents, and the k value is the number of documents to consider when searching for the most similar documents to the question. With gpt-3.5-turbo, k=10 seems to work well, but with gpt-4, k=20 seems to work better.  
            The model's memory is set to 5 messages, but I haven't tested with gpt-3.5-turbo yet to see if it works well. It seems to work well with gpt-4."""
            )
        with gr.TabItem("Repository Selector/Manager", id=1):
            with gr.Row():
                collections_viewer = gr.CheckboxGroup(choices=[], label='Repository Viewer', show_label=True)
            with gr.Row():
                load_collections_button = gr.Button(value="Load respositories to chat!", variant="secondary")#.style(full_width=False)
                get_all_collection_names_button = gr.Button(value="List all saved repositories", variant="secondary")#.style(full_width=False)
                delete_collections_button = gr.Button(value="Delete selected saved repositories", variant="secondary")#.style(full_width=False)
                delete_all_collections_button = gr.Button(value="Delete all saved repositories", variant="secondary")#.style(full_width=False)
            with gr.Row():
                select_embedding_radio = gr.Radio(
                    choices = ['Sentence Transformers', 'OpenAI'],
                    label="Embedding Options",
                    show_label=True,
                    value='Sentence Transformers'
                    )
                select_vectorstore_radio = gr.Radio(
                    choices = ['Chroma', 'raw'],
                    label="Vectorstore Options",
                    show_label=True,
                    value='Chroma'
                    )
        with gr.TabItem("Get New Repositories", id=2):
                with gr.Row():
                    all_collections_to_get = gr.List(headers=['Repository URL', 'Folders'], row_count=3, col_count=2, label='Repositories to get', show_label=True, interactive=True, max_cols=2, max_rows=3)
                    make_collections_button = gr.Button(value="Get new repositories", variant="secondary").style(full_width=False)
                with gr.Row():
                    chunk_size_textbox = gr.Textbox(
                        placeholder="Chunk size",
                        label="Chunk size",
                        show_label=True,
                        lines=1,
                        value="2000"
                    )
                    chunk_overlap_textbox = gr.Textbox(
                        placeholder="Chunk overlap",
                        label="Chunk overlap",
                        show_label=True,
                        lines=1,
                        value="200"
                    )
                    make_embedding_radio = gr.Radio(
                        choices = ['Sentence Transformers', 'OpenAI'],
                        label="Embedding Options",
                        show_label=True,
                        value='Sentence Transformers'
                        )
                    make_vectorstore_radio = gr.Radio(
                        choices = ['Chroma', 'raw'],
                        label="Vectorstore Options",
                        show_label=True,
                        value='Chroma'
                        )
                    
                with gr.Row():
                    gr.HTML('<center>See the <a href=https://python.langchain.com/en/latest/reference/modules/text_splitter.html>Langchain textsplitter docs</a></center>')

        history_state = gr.State()
        agent_state = gr.State()
        vs_state = gr.State()
        all_collections_state = gr.State()
        chat_state = gr.State()
        debug_state = gr.State()
        debug_state.value = False
        radio_state = gr.State()

        submit.click(set_chain_up, inputs=[openai_api_key_textbox, model_selector, k_textbox, search_type_selector, max_tokens_textbox, select_vectorstore_radio, vs_state, agent_state], outputs=[agent_state]).then(chat, inputs=[message, history_state, agent_state], outputs=[chatbot, history_state])
        message.submit(set_chain_up, inputs=[openai_api_key_textbox, model_selector, k_textbox, search_type_selector, max_tokens_textbox, select_vectorstore_radio, vs_state, agent_state], outputs=[agent_state]).then(chat, inputs=[message, history_state, agent_state], outputs=[chatbot, history_state])

        load_collections_button.click(merge_collections, inputs=[collections_viewer, vs_state, k_textbox, search_type_selector, select_vectorstore_radio, select_embedding_radio], outputs=[vs_state])#.then(change_tab, None, tabs) #.then(set_chain_up, inputs=[openai_api_key_textbox, model_selector, k_textbox, max_tokens_textbox, vs_state, agent_state], outputs=[agent_state])
        make_collections_button.click(ingest_docs, inputs=[all_collections_state, all_collections_to_get, chunk_size_textbox, chunk_overlap_textbox, select_vectorstore_radio, select_embedding_radio, debug_state], outputs=[all_collections_state, all_collections_to_get], show_progress=True).then(update_checkboxgroup, inputs = [all_collections_state], outputs = [collections_viewer])
        delete_collections_button.click(delete_collection, inputs=[all_collections_state, collections_viewer, select_vectorstore_radio, select_embedding_radio], outputs=[all_collections_state, collections_viewer]).then(update_checkboxgroup, inputs = [all_collections_state], outputs = [collections_viewer])
        delete_all_collections_button.click(delete_all_collections, inputs=[all_collections_state,select_vectorstore_radio, select_embedding_radio], outputs=[all_collections_state]).then(update_checkboxgroup, inputs = [all_collections_state], outputs = [collections_viewer])
        get_all_collection_names_button.click(list_collections, inputs=[all_collections_state,select_vectorstore_radio, select_embedding_radio], outputs=[all_collections_state]).then(update_checkboxgroup, inputs = [all_collections_state], outputs = [collections_viewer])
        clear_btn.click(clear_chat, inputs = [chatbot, history_state], outputs = [chatbot, history_state])

        make_embedding_radio.change(update_radio, inputs = make_embedding_radio, outputs = select_embedding_radio)
        select_embedding_radio.change(update_radio, inputs = select_embedding_radio, outputs = make_embedding_radio)
        make_vectorstore_radio.change(update_radio, inputs =make_vectorstore_radio, outputs = select_vectorstore_radio)
        select_vectorstore_radio.change(update_radio, inputs = select_vectorstore_radio, outputs = make_vectorstore_radio)

        # Whenever chain parameters change, destroy the agent. 
        input_list = [openai_api_key_textbox, model_selector, k_textbox, max_tokens_textbox, select_vectorstore_radio, make_embedding_radio]
        output_list = [agent_state]
        for input_item in input_list:
            input_item.change(
                destroy_state,
                inputs=output_list,
                outputs=output_list,
            )
        all_collections_state.value = list_collections(all_collections_state, select_vectorstore_radio, select_embedding_radio)
        block.load(update_checkboxgroup, inputs = all_collections_state, outputs = collections_viewer)
    log_textbox_handler = LogTextboxHandler(gr.TextArea(interactive=False, placeholder="Logs will appear here...", visible=False))
    log_textbox = log_textbox_handler.textbox
    logging.getLogger().addHandler(log_textbox_handler)
    log_textbox_visibility_state = gr.State()
    log_textbox_visibility_state.value = False
    log_toggle_button = gr.Button("Toggle Log", variant="secondary")
    log_toggle_button.click(toggle_log_textbox, inputs=[log_textbox_visibility_state], outputs=[log_textbox_visibility_state,log_textbox])

    gr.HTML(
        "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
    )
block.queue(concurrency_count=40)
block.launch(debug=True)
