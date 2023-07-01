# chat-pykg/app.py
import datetime
import logging
import os
import shutil
import sys
from pathlib import Path
import numpy as np
import gradio as gr
from chain import get_new_chain
from collections_manager import get_collections, delete_collection, list_collections, delete_all_collections
from ingest import ingest_docs
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

def set_chain_up(openai_api_key, google_api_key, google_cse_id, model_selector, k_textbox, search_type_selector, max_tokens_textbox, vectorstore_radio, embedding_radio, vectorstores, agent):
    if type(vectorstore_radio) == gr.Radio:
        vectorstore_radio = vectorstore_radio.value
    if not agent or type(agent) == str: 
        if model_selector in ["gpt-3.5-turbo", "gpt-4"]:
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
                os.environ["GOOGLE_API_KEY"] = google_api_key
                os.environ["GOOGLE_CSE_ID"] = google_cse_id
                qa_chain = get_new_chain(vectorstores, vectorstore_radio, embedding_radio, model_selector, k_textbox, search_type_selector, max_tokens_textbox)
                os.environ["OPENAI_API_KEY"] = ""
                os.environ["GOOGLE_API_KEY"] = ""
                os.environ["GOOGLE_CSE_ID"] = ""
                return qa_chain
            else:
                return 'no_open_aikey'
        else:
            qa_chain = get_new_chain(vectorstores, vectorstore_radio, embedding_radio, model_selector, k_textbox, search_type_selector, max_tokens_textbox)
            return qa_chain
    else:
        return agent

def chat(inp, history, agent):
    history = history or []
    if type(agent) == str:
        if agent == 'no_open_aikey':
            history.append((inp, "Please paste your OpenAI key to use"))
            return history, history
    else:
        print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
        print("inp: " + inp)
        history = history or []
        output = agent.run({"input": inp, "chat_history": history})
        answer = output
        history.append((inp, answer))
        print(history)
    return history, history

block = gr.Blocks(title = "chat-pykg", analytics_enabled = False, css=".gradio-container {background-color: system;}")

with block:
    gr.Markdown("<h1><center>chat-pykg</center></h1>")
    with gr.Tabs() as tabs:
        with gr.TabItem("Chat", id=0):
            with gr.Row():
                chatbot = gr.Chatbot()
            with gr.Row():
                clear_btn = gr.Button("Clear Chat", variant="secondary",scale=0)
                message = gr.Textbox(
                    label="What's your question?",
                    placeholder="What is this code?",
                    lines=1,
                )
                submit = gr.Button(value="Send",scale=0)
            gr.Examples(
                examples=[
                    "I want to change the chat-pykg app to have a log viewer, where the user can see what python is doing in the background. How could I do that?",
                    "Hello, I want to allow chat-pykg to search google before answering. In the langchain docs it says you can use a tool to do this: from langchain.agents import load_tools\ntools = load_tools([“google-search”]). How would I need to change get_new_chain1 function to use tools when it needs to as well as searching the vectorstore? Thanks!",
                    "Great, thanks. What if I want to add other tools in the future? Can you please change get_new_chain1 function to do that?"
                ],
                inputs=message,
            )
            with gr.Row():
                with gr.Column(scale=1):
                    model_selector = gr.Dropdown(
                        choices=["gpt-3.5-turbo", "gpt-4", "other"],
                        label="Model",
                        show_label=True,
                        value = "gpt-4"
                    )
                    k_textbox = gr.Textbox(
                        placeholder="k: Number of search results to consider",
                        label="Search Results k:",
                        show_label=True,
                        lines=1,
                        value="10",
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
                        value="500",
                    )
                with gr.Column(scale=1):
                    openai_api_key_textbox = gr.Textbox(
                        placeholder="Paste your OpenAI API key (sk-...)",
                        show_label=True,
                        lines=1,
                        type="password",
                        label="OpenAI API Key",
                    )
                    google_api_key_textbox = gr.Textbox(
                        placeholder="Paste your Google API key (AIza...)",
                        show_label=True,
                        lines=1,
                        type="password",
                        label="Google API Key",
                    )
                    google_cse_id_textbox = gr.Textbox(
                        placeholder="Paste your Google CSE ID (0123...)",
                        show_label=True,
                        lines=1,
                        type="password",
                        label="Google CSE ID",
                    )
            gr.Markdown(
            """
            This simple application is an implementation of ChatGPT but over an external dataset.  
            The source code is split/broken down into many document objects using langchain's pythoncodetextsplitter, which apparently tries to keep whole functions etc. together. This means that each file in the source code is split into many smaller documents, and the k value is the number of documents to consider when searching for the most similar documents to the question. With gpt-3.5-turbo, k=10 seems to work well, but with gpt-4, k=20 seems to work better.  
            """
            )
        with gr.TabItem("Repository Selector/Manager", id=1):
            with gr.Row():
                collections_viewer = gr.CheckboxGroup(choices=[], label='Repository Viewer', show_label=True)
            with gr.Row():
                load_collections_button = gr.Button(value="Load respositories to chat!", variant="secondary")#.style(scale=0)
                get_all_collection_names_button = gr.Button(value="List all saved repositories", variant="secondary")#.style(scale=0)
                delete_collections_button = gr.Button(value="Delete selected saved repositories", variant="secondary")#.style(scale=0)
                delete_all_collections_button = gr.Button(value="Delete all saved repositories", variant="secondary")#.style(scale=0)
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
                    with gr.Column():
                        all_collections_to_get = gr.List(headers=['Repository (organisation/repo_name)', 'Folders (folder1,folder2...)'], row_count=3, col_count=2, label='Repositories to get', show_label=True, interactive=True, max_cols=2, max_rows=3)
                        gr.Markdown(
                            """Folder syntax:  
                            - folder1,folder2,etc.,  
                            - Leave blank or put . to get all known extensions from folders,  
                            - '**/*.py' to get all python files in all directories.  """
                            )
                    with gr.Column():
                        make_collections_button = gr.Button(value="Get new repositories", variant="secondary",scale=0)
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
        vs_state.value = []
        all_collections_state = gr.State()
        chat_state = gr.State()
        debug_state = gr.State()
        debug_state.value = False
        radio_state = gr.State()

        submit.click(set_chain_up, inputs=[openai_api_key_textbox, google_api_key_textbox, google_cse_id_textbox, model_selector, k_textbox, search_type_selector, max_tokens_textbox, select_vectorstore_radio, select_embedding_radio, vs_state, agent_state], outputs=[agent_state]).then(chat, inputs=[message, history_state, agent_state], outputs=[chatbot, history_state])
        message.submit(set_chain_up, inputs=[openai_api_key_textbox, google_api_key_textbox, google_cse_id_textbox, model_selector, k_textbox, search_type_selector, max_tokens_textbox, select_vectorstore_radio, select_embedding_radio, vs_state, agent_state], outputs=[agent_state]).then(chat, inputs=[message, history_state, agent_state], outputs=[chatbot, history_state])

        load_collections_button.click(get_collections, inputs=[collections_viewer, vs_state, agent_state, k_textbox, search_type_selector, select_vectorstore_radio, select_embedding_radio], outputs=[vs_state, agent_state]).then(set_chain_up, inputs=[openai_api_key_textbox, google_api_key_textbox, google_cse_id_textbox, model_selector, k_textbox, search_type_selector, max_tokens_textbox, select_vectorstore_radio, select_embedding_radio,  vs_state, agent_state], outputs=[agent_state])
        make_collections_button.click(ingest_docs, inputs=[all_collections_state, all_collections_to_get, chunk_size_textbox, chunk_overlap_textbox, select_vectorstore_radio, select_embedding_radio, debug_state], outputs=[all_collections_state, all_collections_to_get], show_progress=True).then(update_checkboxgroup, inputs = [all_collections_state], outputs = [collections_viewer])
        delete_collections_button.click(delete_collection, inputs=[all_collections_state, collections_viewer, select_vectorstore_radio, select_embedding_radio], outputs=[all_collections_state, collections_viewer]).then(update_checkboxgroup, inputs = [all_collections_state], outputs = [collections_viewer])
        delete_all_collections_button.click(delete_all_collections, inputs=[all_collections_state,select_vectorstore_radio, select_embedding_radio], outputs=[all_collections_state]).then(update_checkboxgroup, inputs = [all_collections_state], outputs = [collections_viewer])
        get_all_collection_names_button.click(list_collections, inputs=[all_collections_state, select_vectorstore_radio, select_embedding_radio], outputs=[all_collections_state]).then(update_checkboxgroup, inputs = [all_collections_state], outputs = [collections_viewer])
        clear_btn.click(clear_chat, inputs = [chatbot, history_state], outputs = [chatbot, history_state])

        make_embedding_radio.change(update_radio, inputs = make_embedding_radio, outputs = select_embedding_radio)
        select_embedding_radio.change(update_radio, inputs = select_embedding_radio, outputs = make_embedding_radio)
        make_vectorstore_radio.change(update_radio, inputs =make_vectorstore_radio, outputs = select_vectorstore_radio)
        select_vectorstore_radio.change(update_radio, inputs = select_vectorstore_radio, outputs = make_vectorstore_radio)

        # Whenever chain parameters change, destroy the agent. 
        input_list = [openai_api_key_textbox, model_selector, k_textbox, search_type_selector, max_tokens_textbox, select_vectorstore_radio, make_embedding_radio]
        output_list = [agent_state]
        for input_item in input_list:
            input_item.change(
                destroy_state,
                inputs=output_list,
                outputs=output_list,
            )
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
    all_collections_state.value = list_collections(all_collections_state, select_vectorstore_radio, select_embedding_radio)
    block.load(update_checkboxgroup, inputs = all_collections_state, outputs = collections_viewer)
block.queue(concurrency_count=40)
block.launch(debug=True)
