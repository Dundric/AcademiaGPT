import click
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY, MODEL_PATH


@click.command()
@click.option(
    "--device_type",
    default="cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ]
    ),
    help="Device to run on. (Default is cpu)",
)
def main(device_type):
    print(f"Running on: {device_type}")

    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": device_type}
    )
    # load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()
    # Prepare the LLM
    callbacks = [StreamingStdOutCallbackHandler()]
    # load the LLM for generating Natural Language responses.
    llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=2048, callbacks=callbacks, verbose=False)
    #llm = GPT4All(model=model_path, n_ctx=1000, backend='gptj', callbacks=callbacks, verbose=False)
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    # Interactive questions and answers
    while True:
        print("\nto cancel launch sequence type exit")
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain
        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        # # Print the relevant sources used for the answer
        print(
            "----------------------------------SOURCE DOCUMENTS---------------------------"
        )
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)
        print(
            "----------------------------------SOURCE DOCUMENTS---------------------------"
        )

if __name__ == "__main__":
    main()
