import sys
sys.path.insert(0, '/AcademiaGPT/ingest.py')
from ingest import runall
from constants import (ROOT_DIRECTORY)
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from tkinter import *
from tkinter import filedialog
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY, MODEL_PATH
import chainlit as cl

@cl.on_chat_start
async def init():
    # Instantiate the chain for that user session
    actions = [
        cl.Action(name="load_docs", value="example_value", description="Click me!")
    ]

    await cl.Message(content="Click here to select directory:", actions=actions).send()


    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-base", model_kwargs={"device": 'cpu'}
    )
    # load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()
    # Prepare the LLM
    # load the LLM for generating Natural Language responses.
    #llm = GPT4All(model=model_path, n_ctx=1000, backend='gptj', callbacks=callbacks, verbose=False)
    cl.user_session.set("memory", ConversationBufferMemory(memory_key="chat_history", input_key='question', output_key='answer', return_messages=True))

    callbacks = [StreamingStdOutCallbackHandler()]

    llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=4096, callbacks=callbacks, verbose=True, n_gpu_layers=100, n_batch=4096,  f16_kv=True)

    qa = ConversationalRetrievalChain.from_llm(llm=llm, memory = cl.user_session.get("memory"), retriever=retriever, return_source_documents=True, verbose=True)
    # Store the chain in the user session
    cl.user_session.set("chain", qa)


@cl.action_callback("load_docs")
async def on_action(action):
    file_path = filedialog.askdirectory()
    if file_path :
        runall('cpu', 'nuclear', file_path)
    else:
        file_path = "cancelled"
    await cl.Message(content=f"loaded all documents in {file_path}").send()

    

    # Optionally remove the action button from the chatbot user interface


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    qa = cl.make_async(chain)
    result = await qa({"question": message})
    answer = result["answer"]
    # Add the sources to the message
    finaldocs = []

    found_sources = []
    
    # Add the sources to the message
    x = 0
    for source in result["source_documents"]:
        x = x + 1
        source_name = source.metadata["source"]
        source_name = "\n" + str(x) + "." + source_name 
        # Get the index of the source
        found_sources.append(source_name)
        # Create the text element referenced in the message
        finaldocs.append(cl.Text(content=source.page_content, name = source_name))
        print(source.page_content)
    if found_sources:
        answer += f"\nSources: {','.join(found_sources)}"
    # Get the metadata and texts from the user session
    await cl.Message(content = answer, elements=finaldocs).send()