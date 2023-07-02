# chat-pykg/collections.py
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar
import gradio as gr
import numpy as np
import os
import shutil
import logging
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
import chromadb
from chromadb.config import Settings
from ingest import embedding_chooser


def get_collections(collection_load_names, vs_state, agent_state, k_textbox, search_type_selector, vectorstore_radio, embedding_radio):
    agent_state = None
    if type(embedding_radio) == gr.Radio:
        embedding_radio = embedding_radio.value
    if type(vectorstore_radio) == gr.Radio:
        vectorstore_radio = vectorstore_radio.value
    persist_directory = os.path.join(".persisted_data", embedding_radio.replace(' ','_'))
    persist_directory_raw = Path('.persisted_data_raw')
    embedding_function = embedding_chooser(embedding_radio)
    documents = [] 
    embeddings = []
    vectorstores = []
    if vectorstore_radio == 'Chroma':
        for collection_name in collection_load_names: 
            collection_obj = Chroma(collection_name=collection_name, persist_directory=persist_directory, )
            collection = collection_obj._collection.get(include=["metadatas", "documents", "embeddings"])
            for i in range(len(collection['documents'])):
                documents.append(Document(page_content=collection['documents'][i], metadata = collection['metadatas'][i]))
                embeddings.append(collection['embeddings'][i])
            vectorstore = Chroma(collection_name="temp")
            vectorstore._collection.add(ids = collection['ids'], embeddings=collection['embeddings'], metadatas=collection['metadatas'], documents=collection['documents'])
            vectorstore._embedding_function = embedding_function
            vectorstore._collection.metadata = collection_name
            vectorstores.append(vectorstore)
    if vectorstore_radio == 'raw':
        for collection_name in collection_load_names: 
            if collection_name == '':
                continue
            collection_path = persist_directory_raw / collection_name
            docarr = np.load(collection_path.as_posix() +'.npy', allow_pickle=True)
            vectorstores.extend(docarr.tolist())
    return vectorstores, agent_state

def delete_collection(all_collections_state, collections_viewer, select_vectorstore_radio, embedding_radio):
    if type(embedding_radio) == gr.Radio:
        embedding_radio = embedding_radio.value
    if type(select_vectorstore_radio) == gr.Radio:
        select_vectorstore_radio = select_vectorstore_radio.value
    persist_directory = os.path.join(".persisted_data", embedding_radio.replace(' ','_'))
    persist_directory_raw = Path('.persisted_data_raw')
    if select_vectorstore_radio == 'Chroma':
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory, # Optional, defaults to .chromadb/ in the current directory
            anonymized_telemetry=False
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
    if type(select_vectorstore_radio) == gr.Radio:
        select_vectorstore_radio = select_vectorstore_radio.value
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
    if type(select_vectorstore_radio) == gr.Radio:
        select_vectorstore_radio = select_vectorstore_radio.value
    embedding_function = embedding_chooser(embedding_radio)
    persist_directory = os.path.join(".persisted_data", embedding_radio.replace(' ','_'))
    persist_directory_raw = Path('.persisted_data_raw')
    if select_vectorstore_radio == 'Chroma':
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory, # Optional, defaults to .chromadb/ in the current directory
            anonymized_telemetry=False
        ))
        collection_names = [i[1] for i in client._db.list_collections()]
        return collection_names
    if select_vectorstore_radio == 'raw':
        if os.path.exists(persist_directory_raw):
            return [f.name.split('.npy')[0] for f in os.scandir(persist_directory_raw)]
    return []