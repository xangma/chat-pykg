# chat-pykg/chain.py
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar
from pydantic import Extra, Field, root_validator
from langchain.chains.base import Chain
from langchain import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.llm import LLMChain
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.schema import BaseLanguageModel, BaseRetriever, Document
from langchain.prompts.prompt import PromptTemplate
from langchain.utilities.google_serper import GoogleSerperAPIWrapper
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.agents.self_ask_with_search.base import SelfAskWithSearchChain
from langchain.agents.self_ask_with_search.prompt import PROMPT

class ConversationalRetrievalChainWithGoogleSearch(ConversationalRetrievalChain):
    google_search_tool: GoogleSearchAPIWrapper

    def _get_docs(self, question: str, inputs: Dict[str, Any]) -> List[Document]:
        # Get documents from the retriever
        docs_from_retriever = self.retriever.get_relevant_documents(question)

        # Get search results from Google Search
        search_results = self.google_search_tool.results(question, num_results=self.google_search_tool.k)

        # Create documents from the search results
        docs_from_search = []
        for result in search_results:
            content = result.get("snippet", "")
            metadata = {"title": result["title"], "link": result["link"]}
            docs_from_search.append(Document(page_content=content, metadata=metadata))

        # Combine both lists of documents
        docs = docs_from_retriever + docs_from_search

        return self._reduce_tokens_below_limit(docs)

def get_new_chain1(vectorstore, vectorstore_radio, model_selector, k_textbox, search_type_selector, max_tokens_textbox) -> Chain:
    retriever = None
    if vectorstore_radio == 'Chroma':
        retriever = vectorstore.as_retriever(search_type=search_type_selector)
        retriever.search_kwargs = {"k":int(k_textbox)}
        if search_type_selector == 'mmr':
            retriever.search_kwargs = {"k":int(k_textbox), "fetch_k":4*int(k_textbox)}
    if vectorstore_radio == 'raw':
        if search_type_selector == 'svm':
            retriever = SVMRetriever.from_texts(merged_vectorstore, embedding_function)
            retriever.k = int(k_textbox)

    qa_template = """You are called chat-pykg and are an AI assistant coded in python using langchain and gradio. You are very helpful for answering questions about programming with various open source packages and libraries.
                You are given snippets of code and information in the Context below, as well as a Question to give a Helpful answer to. 
                Due to data size limitations, the snippets of code in the Context have been specifically filtered/selected for their relevance from a document store containing code from one or many packages and libraries.
                Each of the code snippets is marked with '# source: package/filename' so you can attempt to establish where they are located in their package structure and gain more understanding of the code.
                Please provide a helpful answer in markdown to the Question.
                Do not make up any hyperlinks that are not in the Context.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.

                =========
                Context:{context}
                =========
                Question: {question}
                Helpful answer:"""
    QA_PROMPT.template = qa_template

    condense_question_template = """Given the following conversation and a Follow Up Input, rephrase the Follow Up Input to be a Standalone question.
    The Standalone question will be used for retrieving relevant source code and information from a document store, where each document is marked with '# source: package/filename'.
    Therefore, in your Standalone question you must try to include references to related code or sources that have been mentioned in the Follow Up Input or Chat History.
    =========
    Chat History:
    {chat_history}
    =========
    Follow Up Input: {question}
    Standalone question in markdown:"""
    CONDENSE_QUESTION_PROMPT.template = condense_question_template
    if model_selector in ['gpt-4', 'gpt-3.5-turbo']:
        llm = ChatOpenAI(client = None, temperature=0.7, model_name=model_selector)
        doc_chain_llm = ChatOpenAI(client = None, streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0.7, model_name=model_selector, max_tokens=int(max_tokens_textbox))
    if model_selector == 'other':
        llm = HuggingFaceHub(repo_id="chavinlo/gpt4-x-alpaca")#, model_kwargs={"temperature":0, "max_length":64})
        doc_chain_llm = HuggingFaceHub(repo_id="chavinlo/gpt4-x-alpaca")
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(doc_chain_llm, chain_type="stuff", prompt=QA_PROMPT)#, document_prompt = PromptTemplate(input_variables=["source", "page_content"], template="{source}\n{page_content}"))
    
    # memory = ConversationKGMemory(llm=llm, input_key="question", output_key="answer")
    memory = ConversationBufferWindowMemory(input_key="question", output_key="answer", k=5)

    google_search_tool = GoogleSearchAPIWrapper(search_engine = "google", k = int(int(k_textbox)/2))

    qa_orig = ConversationalRetrievalChain(
        retriever=retriever, memory=memory, combine_docs_chain=doc_chain, question_generator=question_generator, verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    qa_with_google_search = ConversationalRetrievalChainWithGoogleSearch(
        retriever=retriever,
        memory=memory,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        google_search_tool=google_search_tool,
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    qa = qa_orig
    return qa