import os
from typing import List
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from multiprocessing import Pool
import glob
from constants import (CHROMA_SETTINGS, DOCUMENT_MAP, PERSIST_DIRECTORY, ROOT_DIRECTORY)


def load_single_document(file_path: str) -> List[Document]:
    # Loads a single document from a file path
    file_extension = os.path.splitext(file_path)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined")
    return loader.load()


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    # Loads all documents from the source documents directory
    all_files = []
    for ext in DOCUMENT_MAP:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
    p = Pool()
    docs = []
    for file in p.imap_unordered(load_single_document, filtered_files):
        docs.extend(file)
    p.close()
    return docs


def runall(device_type, topic, directory):
    #Set up embedding settings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-base",
        model_kwargs={"device": device_type},
        query_instruction = "Represent the " + topic + " document",
    )

    if(os.path.exists(f"{ROOT_DIRECTORY}/DB")) :
        # Load documents and split in chunks
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        print(f"Loading documents from {directory}")
        documents = load_documents(directory, [metadata['source'] for metadata in collection['metadatas']])
        if len(documents) != 0 :
            for doc in documents:
                templist = [doc]
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                texts = text_splitter.split_documents(templist)
                print(f"Loaded {len(templist)} documents from {directory}")
                print(f"Split into {len(texts)} chunks of text")
                print("\nThis may take a while you will see the model is ready when embedding is finished")
                # Create embeddings
                db.add_documents(texts)
        else :
            print("All Documents Already Loaded")
    else :
        # Load documents and split in chunks
        print(f"Loading documents from {directory}")
        documents = load_documents(directory)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        print(f"Loaded {len(documents)} documents from {directory}")
        print(f"Split into {len(texts)} chunks of text")
        print("\nThis may take a while you will see the model is ready when embedding is finished")
        # Create embeddings
        db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=PERSIST_DIRECTORY,
            client_settings=CHROMA_SETTINGS,
        )
    db.persist()
    db = None
    print("\nThe model is ready to be used run the command 'python runmodel.py'")
if __name__ == "__runall__":
    runall()
