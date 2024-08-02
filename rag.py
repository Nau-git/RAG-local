import dotenv
dotenv.load_dotenv()

import chainlit as cl
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import (
                                create_history_aware_retriever,
                                create_retrieval_chain,
                            )
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from prompts import contextualize_q_system_prompt, rag_system_prompt

contextualize = contextualize_q_system_prompt()
system_prompt = rag_system_prompt()

OPENING_MESSAGE = ("Welcome! How may I assist you today?")
LLM = 'gemma2:9b'
EMBEDDING = "all-MiniLM-L6-v2"
RETRIEVED_CONTEXT = 4

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING)
chat_model = ChatOllama(model=LLM)

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVED_CONTEXT})


@cl.on_chat_start
async def on_chat_start():

    # Contextualize question
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        chat_model, retriever, contextualize_q_prompt
    )

    # Answer question
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    document_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{page_content} and {source} and {file_name}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm=chat_model, prompt=qa_prompt, document_prompt=document_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    cl.user_session.set("runnable", rag_chain)
    cl.user_session.set("chat_history", [])

    await cl.Message(content=OPENING_MESSAGE).send()


@cl.on_message
async def on_message(message: cl.Message):
    
    runnable = cl.user_session.get("runnable") # type: Runnable
    chat_history = cl.user_session.get("chat_history") # type: List[cl.Message]

    chat_history.append({"role": "human", "content": message.content})
    msg = cl.Message(content="")
    
    answer = ''
    context = ''
    async for chunk in runnable.astream(
                        {'input': message.content, 'chat_history': chat_history},
                        config = RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])
                        ):

        if chunk.get('answer') is not None: 
            await msg.stream_token(chunk.get('answer'))
            answer += chunk.get('answer')

        elif chunk.get('chat_history') is not None:
            chat_history = chunk.get('chat_history')

        elif chunk.get('context') is not None:
            context = chunk.get('context')

    await msg.send()
    chat_history.append({"role": "assistant", "content": answer})