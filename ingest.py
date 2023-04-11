import pickle

from langchain.document_loaders import SitemapLoader, ReadTheDocsLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, PythonCodeTextSplitter, MarkdownTextSplitter
from langchain.vectorstores.faiss import FAISS
import itertools
import os
import fsspec
from pathlib import Path

def get_text(content):
    relevant_part = content.find("div", {"class": "markdown"})
    if relevant_part is not None:
        return relevant_part.get_text(separator=" ")
    else:
        return ""

def ingest_docs(urls=[]):
    """Get documents from web pages."""
    folders=[]
    documents = []
    for url in urls:
        try:

            if "local:" in url:
                folders.append(url.split('local:')[1])
            else:
                url = url[0]
                if url[0] == '/':
                    url = url[1:]
                if url[-1] != '/':
                    url += '/'
                org = url.split('/')[0]
                repo = url.split('/')[1]
                # join all strings after 2nd slash
                folder = '/'.join(url.split('/')[2:])
                if folder[-1] != '/':
                    folder += '/'
                fs = fsspec.filesystem("github", org=org, repo=repo)
                # recursive copy
                destination = url
                destination.mkdir(exist_ok=True, parents=True)
                fs.get(fs.ls(folder), destination.as_posix(), recursive=True)
                folders.append(destination)
        except Exception as e:
            print(e)
    for folder in folders:
        try:
            py_splitter = PythonCodeTextSplitter(chunk_size=1000, chunk_overlap=0)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            md_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=0)
            local_repo_path_1 = folder
            known_exts = [".py", ".md", ".rst"]
            paths_by_ext = {}
            docs_by_ext = {}
            for ext in known_exts + ["other"]:
                docs_by_ext[ext] = []
                paths_by_ext[ext] = []
            for root, dirs, files in os.walk(local_repo_path_1):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_file_path = os.path.relpath(file_path, local_repo_path_1)
                    for ext in paths_by_ext.keys():
                        if '.' not in [i[0] for i in rel_file_path.split('/')]:
                            if rel_file_path.endswith(ext):
                                paths_by_ext[ext].append(rel_file_path)
                                docs_by_ext[ext].append(TextLoader(os.path.join(local_repo_path_1, rel_file_path)).load())
                            else:
                                paths_by_ext["other"].append(rel_file_path)
                                docs_by_ext["other"].append(TextLoader(os.path.join(local_repo_path_1, rel_file_path)).load())

            for ext in docs_by_ext.keys():
                if ext == ".py":
                    documents += py_splitter.split_documents(docs_by_ext[ext])
                elif ext == ".md" or ext == ".rst":
                    documents += md_splitter.split_documents(docs_by_ext[ext])
                else:
                    documents += text_splitter.split_documents(docs_by_ext[ext])
        except Exception as e:
            print(e)
            continue
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)
    return vectorstore

if __name__ == "__main__":
    ingest_docs()
