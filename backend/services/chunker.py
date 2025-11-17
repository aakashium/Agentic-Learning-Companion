from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return splitter.split_documents(docs)
