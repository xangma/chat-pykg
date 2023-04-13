# chat-pykg/app.py
import datetime
import os
import gradio as gr
import chromadb
from chromadb.config import Settings
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
import shutil
import random, string
from chain import get_new_chain1
from ingest import ingest_docs

def randomword(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def change_tab():
    return gr.Tabs.update(selected=0)

def merge_collections(collection_load_names, vs_state): 
    merged_documents = [] 
    merged_embeddings = []
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=".persisted_data" # Optional, defaults to .chromadb/ in the current directory
    ))
    
    for collection_name in collection_load_names: 
        collection_name = collection_name
        if collection_name == '': 
            continue
        collection = client.get_collection(collection_name)
        collection = collection.get(include=["metadatas", "documents", "embeddings"])
        for i in range(len(collection['documents'])):
            merged_documents.append(Document(page_content=collection['documents'][i], metadata = collection['metadatas'][i]))
            merged_embeddings.append(collection['embeddings'][i])
    merged_collection_name = "merged_collection" 
    merged_vectorstore = Chroma.from_documents(documents=merged_documents, embeddings=merged_embeddings, collection_name=merged_collection_name) 
    return merged_vectorstore

def set_chain_up(openai_api_key, model_selector, k_textbox, max_tokens_textbox, vectorstore, agent):
    if agent == None or type(agent) == str: 
        if vectorstore != None:
            if model_selector in ["gpt-3.5-turbo", "gpt-4"]:
                if openai_api_key:
                    os.environ["OPENAI_API_KEY"] = openai_api_key
                    qa_chain = get_new_chain1(vectorstore, model_selector, k_textbox, max_tokens_textbox)
                    os.environ["OPENAI_API_KEY"] = ""
                    return qa_chain
                else:
                    return 'no_open_aikey'
            else:
                qa_chain = get_new_chain1(vectorstore, model_selector, k_textbox, max_tokens_textbox)
                return qa_chain
        else:
            return 'no_vectorstore'
    else:
        return agent

def delete_vs(all_collections_state, collections_viewer):
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=".persisted_data" # Optional, defaults to .chromadb/ in the current directory
    ))
    for collection in collections_viewer:
        client.delete_collection(collection)
        all_collections_state.remove(collection)
    return all_collections_state

def delete_all_vs(all_collections_state):
    shutil.rmtree(".persisted_data")
    return []

def list_collections(all_collections_state):
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=".persisted_data" # Optional, defaults to .chromadb/ in the current directory
    ))
    collection_names = [[c.name][0] for c in client.list_collections()]
    return collection_names

def update_checkboxgroup(all_collections_state):
    new_options = [i for i in all_collections_state]
    return gr.CheckboxGroup.update(choices=new_options)

def destroy_agent(agent):
    agent = None
    return agent

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
    gr.Markdown("<h3><center>chat-pykg</center></h3>")
    with gr.Tabs() as tabs:
        with gr.TabItem("Chat", id=0):
            with gr.Row():
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
                k_textbox.value = "20"
                max_tokens_textbox = gr.Textbox(
                    placeholder="max_tokens: Maximum number of tokens to generate",
                    label="max_tokens",
                    show_label=True,
                    lines=1,
                )
                max_tokens_textbox.value="2000"
            chatbot = gr.Chatbot()
            with gr.Row():
                message = gr.Textbox(
                    label="What's your question?",
                    placeholder="What is this code?",
                    lines=1,
                )
                submit = gr.Button(value="Send", variant="secondary").style(full_width=False)
            gr.Examples(
                examples=[
                    "What does this code do?",
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
        with gr.TabItem("Collections manager", id=1):
            with gr.Row():
                with gr.Column(scale=2):
                    all_collections_to_get = gr.List(headers=['New Collections to make'],row_count=3, label='Collections_to_get', show_label=True, interactive=True, max_cols=1, max_rows=3)
                    make_vs_button = gr.Button(value="Make new collection(s)", variant="secondary").style(full_width=False)
                with gr.Column(scale=2):
                    collections_viewer = gr.CheckboxGroup(choices=[], label='Collections_viewer', show_label=True)
                with gr.Column(scale=1):
                    get_vs_button = gr.Button(value="Load collection(s) to chat!", variant="secondary").style(full_width=False)
                    get_all_vs_names_button = gr.Button(value="List all saved collections", variant="secondary").style(full_width=False)
                    delete_vs_button = gr.Button(value="Delete selected saved collections", variant="secondary").style(full_width=False)
                    delete_all_vs_button = gr.Button(value="Delete all saved collections", variant="secondary").style(full_width=False)
        gr.HTML(
            "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
        )

        history_state = gr.State()
        agent_state = gr.State()
        vs_state = gr.State()
        all_collections_state = gr.State()
        chat_state = gr.State()

        submit.click(set_chain_up, inputs=[openai_api_key_textbox, model_selector, k_textbox, max_tokens_textbox, vs_state, agent_state], outputs=[agent_state])
        message.submit(chat, inputs=[message, history_state, agent_state], outputs=[chatbot, history_state])

        get_vs_button.click(merge_collections, inputs=[collections_viewer, vs_state], outputs=[vs_state])#.then(set_chain_up, inputs=[openai_api_key_textbox, model_selector, k_textbox, max_tokens_textbox, vs_state, agent_state], outputs=[agent_state])
        make_vs_button.click(ingest_docs, inputs=[all_collections_state, all_collections_to_get], outputs=[all_collections_state], show_progress=True).then(update_checkboxgroup, inputs = [all_collections_state], outputs = [collections_viewer])
        delete_vs_button.click(delete_vs, inputs=[all_collections_state, collections_viewer], outputs=[all_collections_state]).then(update_checkboxgroup, inputs = [all_collections_state], outputs = [collections_viewer])
        delete_all_vs_button.click(delete_all_vs, inputs=[all_collections_state], outputs=[all_collections_state]).then(update_checkboxgroup, inputs = [all_collections_state], outputs = [collections_viewer])
        get_all_vs_names_button.click(list_collections, inputs=[all_collections_state], outputs=[all_collections_state]).then(update_checkboxgroup, inputs = [all_collections_state], outputs = [collections_viewer])
        
        # Whenever chain parameters change, destroy the agent. 
        input_list = [openai_api_key_textbox, model_selector, k_textbox, max_tokens_textbox]
        output_list = [agent_state]
        for input_item in input_list:
            input_item.change(
                destroy_agent,
                inputs=output_list,
                outputs=output_list,
            )
        all_collections_state.value = list_collections(all_collections_state)
        block.load(update_checkboxgroup, inputs = all_collections_state, outputs = collections_viewer)
block.launch(debug=True)
