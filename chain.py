# chat-pykg/chain.py

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

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def get_new_chain1(vectorstore, model_selector, k_textbox, max_tokens_textbox) -> Chain:

    template = """You are called chat-pykg and are an AI assistant coded in python using langchain and gradio. You are very helpful for answering questions about various open source libraries.
                You are given the following extracted parts of code and a question. Provide a conversational answer to the question.
                Do NOT make up any hyperlinks that are not in the code.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                If the question is not about the package documentation, politely inform them that you are tuned to only answer questions about the package documentations.
                Question: {question}
                =========
                {context}
                =========
                Answer in Markdown:"""
    QA_PROMPT.template = template
    if model_selector in ['gpt-4', 'gpt-3.5-turbo']:
        llm = ChatOpenAI(client = None, temperature=0.7, model_name=model_selector)
        doc_chain_llm = ChatOpenAI(client = None, streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0.7, model_name=model_selector, max_tokens=int(max_tokens_textbox))
    if model_selector == 'other':
        llm = HuggingFaceHub(repo_id="chavinlo/gpt4-x-alpaca")#, model_kwargs={"temperature":0, "max_length":64})
        doc_chain_llm = HuggingFaceHub(repo_id="chavinlo/gpt4-x-alpaca")
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(doc_chain_llm, chain_type="stuff", prompt=QA_PROMPT)

    # memory = ConversationKGMemory(llm=llm, input_key="question", output_key="answer")
    memory = ConversationBufferWindowMemory(input_key="question", output_key="answer", k=5)
    retriever = vectorstore.as_retriever(search_type="similarity")
    if len(k_textbox) != 0:
        retriever.search_kwargs = {"k": int(k_textbox)}
    else:
        retriever.search_kwargs = {"k": 10}
    qa = ConversationalRetrievalChain(
        retriever=retriever, memory=memory, combine_docs_chain=doc_chain, question_generator=question_generator)

    return qa