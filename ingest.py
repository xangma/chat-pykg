import pickle
import tempfile
from langchain.document_loaders import SitemapLoader, ReadTheDocsLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, PythonCodeTextSplitter, MarkdownTextSplitter
from langchain.vectorstores.faiss import FAISS
import chromadb
import os
from langchain.vectorstores import Chroma
import shutil
from pathlib import Path
import subprocess
import tarfile
# import chromadb
from abc import ABC
from typing import List, Optional, Any
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from chromadb.config import Settings

# class CachedChroma(Chroma, ABC):
#     """
#     Wrapper around Chroma to make caching embeddings easier.
    
#     It automatically uses a cached version of a specified collection, if available.
#         Example:
#             .. code-block:: python
#                     from langchain.vectorstores import Chroma
#                     from langchain.embeddings.openai import OpenAIEmbeddings
#                     embeddings = OpenAIEmbeddings()
#                     vectorstore = CachedChroma.from_documents_with_cache(
#                         ".persisted_data", texts, embeddings, collection_name="fun_experiment"
#                     )
#         """
    
#     @classmethod
#     def from_documents_with_cache(
#             cls,
#             persist_directory: str,
#             documents: Optional[List[Document]] = None,
#             embedding: Optional[Embeddings] = None,
#             ids: Optional[List[str]] = None,
#             collection_name: str = Chroma._LANGCHAIN_DEFAULT_COLLECTION_NAME,
#             client_settings: Optional[chromadb.config.Settings] = None,
#             **kwargs: Any,
#     ) -> Chroma:
        # client_settings = Settings(
        #     chroma_db_impl="duckdb+parquet",
        #     persist_directory=persist_directory # Optional, defaults to .chromadb/ in the current directory
        # )
        # client = chromadb.Client(client_settings)
#         collection_names = [c.name for c in client.list_collections()]

#         if collection_name in collection_names:
#             return Chroma(
#                 collection_name=collection_name,
#                 embedding_function=embedding,
#                 persist_directory=persist_directory,
#                 client_settings=client_settings,
#             )
#         if documents:
#             return Chroma.from_documents(
#                 documents=documents,
#                 embedding=embedding,
#                 ids=ids,
#                 collection_name=collection_name,
#                 persist_directory=persist_directory,
#                 client_settings=client_settings,
#                 **kwargs
#             )
#         raise ValueError("Either documents or collection_name must be specified.")

def get_text(content):
    relevant_part = content.find("div", {"class": "markdown"})
    if relevant_part is not None:
        return relevant_part.get_text(separator=" ")
    else:
        return ""

def ingest_docs(all_collections_state, urls):
    """Get documents from web pages."""
    all_docs = []
    local = False
    folders=[]
    documents = []                    
    shutil.rmtree('downloaded/', ignore_errors=True)
    known_exts = ["py", "md"]
    py_splitter = PythonCodeTextSplitter(chunk_size=1000, chunk_overlap=0)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    md_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=0)
    for url in urls:
        paths_by_ext = {}
        docs_by_ext = {}
        for ext in known_exts + ["other"]:
            docs_by_ext[ext] = []
            paths_by_ext[ext] = []
        url = url[0]
        if url == '':
            continue
        if "." in url:
            local = True
            if len(url) > 1:
                folder = url.split('.')[1]
            else:
                folder = '.'
        else:
            destination = Path('downloaded/'+url)
            destination.mkdir(exist_ok=True, parents=True)
            destination = destination.as_posix()
            if url[0] == '/':
                url = url[1:]
            org = url.split('/')[0]
            repo = url.split('/')[1]
            repo_url = f"https://github.com/{org}/{repo}.git"
            # join all strings after 2nd slash
            folder = '/'.join(url.split('/')[2:])
            if folder[-1] == '/':
                folder = folder[:-1]
            if folder:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)

                    # Initialize the Git repository
                    subprocess.run(["git", "init"], cwd=temp_path)

                    # Add the remote repository
                    subprocess.run(["git", "remote", "add", "-f", "origin", repo_url], cwd=temp_path)

                    # Enable sparse-checkout
                    subprocess.run(["git", "config", "core.sparseCheckout", "true"], cwd=temp_path)

                    # Specify the folder to checkout
                    with open(temp_path / ".git" / "info" / "sparse-checkout", "w") as f:
                        f.write(f"{folder}/\n")

                    # Checkout the desired branch
                    res = subprocess.run(["git", "checkout", 'main'], cwd=temp_path)
                    if res.returncode == 1:
                        res = subprocess.run(["git", "checkout", "master"], cwd=temp_path)
                    res = subprocess.run(["cp", "-r", (temp_path / folder).as_posix(), '/'.join(destination.split('/')[:-1])])
                    folder = destination
        local_repo_path_1 = folder
        for root, dirs, files in os.walk(local_repo_path_1):
            for file in files:
                file_path = os.path.join(root, file)
                rel_file_path = os.path.relpath(file_path, local_repo_path_1)
                ext = rel_file_path.split('.')[-1]
                try:
                    if '.' not in [i[0] for i in rel_file_path.split('/')]:
                        if paths_by_ext.get(rel_file_path.split('.')[-1]) is None:
                            paths_by_ext["other"].append(rel_file_path)
                            docs_by_ext["other"].append(TextLoader(os.path.join(local_repo_path_1, rel_file_path)).load()[0])
                        else:
                            paths_by_ext[ext].append(rel_file_path)
                            docs_by_ext[ext].append(TextLoader(os.path.join(local_repo_path_1, rel_file_path)).load()[0])
                except Exception as e:
                    continue
        for ext in docs_by_ext.keys():
            if ext == "py":
                documents += py_splitter.split_documents(docs_by_ext[ext])
            if ext == "md":
                documents += md_splitter.split_documents(docs_by_ext[ext])
            # else:
            #     documents += text_splitter.split_documents(docs_by_ext[ext] 
        all_docs += documents
        embeddings = HuggingFaceEmbeddings()
        if 'downloaded/' in folder:
            folder = '-'.join(folder.split('/')[1:])
        if folder == '.':
            folder = 'chat-pykg'
        vectorstore = Chroma.from_documents(persist_directory=".persisted_data", documents=documents, embedding=embeddings, collection_name=folder)
        vectorstore.persist()
        all_collections_state.append(folder)
    return all_collections_state
    # embeddings = HuggingFaceEmbeddings()
    # merged_vectorstore = Chroma.from_documents(persist_directory=".persisted_data", documents=documents, embedding=embeddings, collection_name='merged_collections')
    # #vectorstore = FAISS.from_documents(documents, embeddings)
    # # # Save vectorstore
    # # with open("vectorstore.pkl", "wb") as f:
    # #     pickle.dump(vectorstore. , f)
    
    # return merged_vectorstore
    

if __name__ == "__main__":
    ingest_docs()
