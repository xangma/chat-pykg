import pickle

from langchain.document_loaders import SitemapLoader, ReadTheDocsLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, PythonCodeTextSplitter, MarkdownTextSplitter
from langchain.vectorstores.faiss import FAISS
import itertools
import os

def get_text(content):
    relevant_part = content.find("div", {"class": "markdown"})
    if relevant_part is not None:
        return relevant_part.get_text(separator=" ")
    else:
        return ""

def ingest_docs(urls=[]):
    """Get documents from web pages."""
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    for url in urls:
        try:
            url = url[0]
            if len([i for i in map(''.join, itertools.product(*zip('sitemap'.upper(), 'sitemap'.lower()))) if i in url]) > 0:
                loader = SitemapLoader(
                    web_path=url, parsing_function=get_text
                )
            elif len([i for i in map(''.join, itertools.product(*zip('readthedocs'.upper(), 'readthedocs'.lower()))) if i in url]) > 0:
                loader = ReadTheDocsLoader(
                        path=url
                    )
            elif "local:" in url:
                local_repo_path_1 = url.split('local:')[1]
                loaders = []
                known_exts = [".py", ".md"]
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
                                else:
                                    paths_by_ext["other"].append(rel_file_path)

                # for each extension, load the files and split them
                for ext in paths_by_ext.keys():
                    for i in range(len(paths_by_ext[ext])):
                        try:
                            docs_by_ext[ext] += TextLoader(os.path.join(local_repo_path_1, paths_by_ext[ext][i])).load()
                        except Exception as e:
                            print(e)
                            continue
                py_splitter = PythonCodeTextSplitter(chunk_size=1000, chunk_overlap=0)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                md_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=0)
                for ext in docs_by_ext.keys():
                    if ext == ".py":
                        documents += py_splitter.split_documents(docs_by_ext[ext])
                    elif ext == ".md":
                        documents += md_splitter.split_documents(docs_by_ext[ext])
                    else:
                        documents += text_splitter.split_documents(docs_by_ext[ext])
            else:
                raise ValueError("No loader found for this url")
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
