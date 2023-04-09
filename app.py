import datetime
import os
import gradio as gr
from abc import ABC
from typing import List, Optional, Any
import chromadb
import langchain
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, PythonCodeTextSplitter
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma

from chain import get_new_chain1
from ingest import ingest_docs

class CachedChroma(Chroma, ABC):
    """
    Wrapper around Chroma to make caching embeddings easier.
    
    It automatically uses a cached version of a specified collection, if available.
        Example:
            .. code-block:: python
                    from langchain.vectorstores import Chroma
                    from langchain.embeddings.openai import OpenAIEmbeddings
                    embeddings = OpenAIEmbeddings()
                    vectorstore = CachedChroma.from_documents_with_cache(
                        ".persisted_data", texts, embeddings, collection_name="fun_experiment"
                    )
        """
    
    @classmethod
    def from_documents_with_cache(
            cls,
            persist_directory: str,
            documents: List[Document],
            embedding: Optional[Embeddings] = None,
            ids: Optional[List[str]] = None,
            collection_name: str = Chroma._LANGCHAIN_DEFAULT_COLLECTION_NAME,
            client_settings: Optional[chromadb.config.Settings] = None,
            **kwargs: Any,
    ) -> Chroma:
        settings = chromadb.config.Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        )
        client = chromadb.Client(settings)
        collection_names = [c.name for c in client.list_collections()]

        if collection_name in collection_names:
            return Chroma(
                collection_name=collection_name,
                embedding_function=embedding,
                persist_directory=persist_directory,
                client_settings=client_settings,
            )

        return Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            ids=ids,
            collection_name=collection_name,
            persist_directory=persist_directory,
            client_settings=client_settings,
            **kwargs
        )

# def get_docs():
#     local_repo_path_1 = "pycbc/"
#     loaders = []
#     docs = []
#     for root, dirs, files in os.walk(local_repo_path_1):
#         for file in files:
#             file_path = os.path.join(root, file)
#             rel_file_path = os.path.relpath(file_path, local_repo_path_1)
#             # Filter by file extension
#             if any(rel_file_path.endswith(ext) for ext in [".py", ".sh"]):
#                 # Filter by directory
#                 if any(rel_file_path.startswith(d) for d in ["pycbc/", "examples/"]):
#                     docs.append(rel_file_path)
#             if any(rel_file_path.startswith(d) for d in ["bin/"]):
#                 docs.append(rel_file_path)
#     loaders.extend([TextLoader(os.path.join(local_repo_path_1, doc)).load() for doc in docs])
#     py_splitter = PythonCodeTextSplitter(chunk_size=1000, chunk_overlap=0)
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     documents = []
#     for load in loaders:
#         try:
#             if load[0].metadata['source'][-3:] == ".py" == "" or "pycbc/bin/" in load[0].metadata['source']:
#                 documents.extend(py_splitter.split_documents(load))
#         except Exception as e:
#             documents.extend(text_splitter.split_documents(load))
#     return documents

def set_chain_up(openai_api_key, model_selector, k_textbox, vectorstore, agent):

    # # set defaults
    # if not model_selector:
    #     model_selector = "gpt-3.5-turbo"
    # if not k_textbox:
    #     k_textbox = 10
    # else:
    #     k_textbox = int(k_textbox)
    if type(vectorstore) != list: 
        if model_selector in ["gpt-3.5-turbo", "gpt-4"]:
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
                qa_chain = get_new_chain1(vectorstore, model_selector, k_textbox)
                os.environ["OPENAI_API_KEY"] = ""
                return qa_chain
        else:
            qa_chain = get_new_chain1(vectorstore, model_selector, k_textbox)
            return qa_chain

def get_vectorstore(openai_api_key_textbox, model_selector, k_textbox, packagedocslist, vs_state, agent_state):
    vectorstore = ingest_docs(packagedocslist)
    agent_state = set_chain_up(openai_api_key_textbox, model_selector, k_textbox, vectorstore, agent_state)
    return vectorstore, agent_state

def chat(inp, history, agent):
    history = history or []
    if agent is None:
        history.append((inp, "Please paste your OpenAI key to use"))
        return history, history
    print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
    print("inp: " + inp)
    history = history or []
    output = agent({"question": inp, "chat_history": history})
    answer = output["answer"]
    history.append((inp, answer))
    print(history)
    return history, history

block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

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
        packagedocslist = gr.List(headers=['Package Docs URL'], label='Package docs URLs', show_label=True, interactive=True, max_cols=1, max_rows=5)
        submit_urls = gr.Button(value="Get docs", variant="secondary").style(full_width=False)

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

    state = gr.State()
    agent_state = gr.State()
    vs_state = gr.State()

    submit.click(chat, inputs=[message, state, agent_state], outputs=[chatbot, state])
    message.submit(chat, inputs=[message, state, agent_state], outputs=[chatbot, state])
    submit_urls.click(get_vectorstore, inputs=[openai_api_key_textbox, model_selector, k_textbox, packagedocslist, vs_state, agent_state], outputs=[vs_state, agent_state])

    # I need to also parse this code in the docstore so I can ask it to fix silly things like this below:
    openai_api_key_textbox.change(
        set_chain_up,
        inputs=[openai_api_key_textbox, model_selector, k_textbox, packagedocslist, agent_state],
        outputs=[agent_state],
    )
    model_selector.change(
        set_chain_up,
        inputs=[openai_api_key_textbox, model_selector, k_textbox, packagedocslist, agent_state],
        outputs=[agent_state],
    )
    k_textbox.change(
        set_chain_up,
        inputs=[openai_api_key_textbox, model_selector, k_textbox, packagedocslist, agent_state],
        outputs=[agent_state],
    )

block.launch(debug=True)
