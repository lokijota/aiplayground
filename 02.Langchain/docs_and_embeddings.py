import os
from langchain_openai import ChatOpenAI

# mostly code from https://python.langchain.com/docs/tutorials/retrievers/

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = "ollama"

model = ChatOpenAI(model="llama3.2", base_url = 'http://localhost:11434/v1')

# 01. Documents and Document Loaders

from langchain_core.documents import Document

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# 02. Load documents (eg a PDF)

from langchain_community.document_loaders import PyPDFLoader
# Other loaders here: https://python.langchain.com/docs/integrations/document_loaders/

# file_path = "./02.Langchain/nke-10k-2023.pdf"
file_path = "./02.Langchain/Maias_20001210.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load() 
print(f"Num pages: {len(docs)}")

# PyPDFLoader loads one Document object per PDF page. For each, we can easily access:
# - The string content of the page;
# - Metadata containing the file name and page number.

print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata) # source document, page, page label (python kvp)

# Text splitting https://python.langchain.com/docs/concepts/text_splitters/
# We will split our documents into chunks of 1000 characters with 200 characters of overlap between chunks. The overlap helps mitigate the
# possibility of separating a statement from important context related to it. We use the RecursiveCharacterTextSplitter, which will recursively
# split the document using common separators like new lines until each chunk is the appropriate size. This is the recommended text splitter for
# generic text use cases.

from langchain_text_splitters import RecursiveCharacterTextSplitter

# We set add_start_index=True so that the character index where each split Document starts within the initial Document is preserved as metadata attribute “start_index”.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(f"#Splits: {len(all_splits)}")
print("\nSplit 0:", all_splits[0])
print("\nSplit 1:", all_splits[1])
print("\nSplit 2:", all_splits[2])
# note the metadata on this, inc 

# 03. Embeddings

from langchain_ollama import OllamaEmbeddings

embedding_model1 = OllamaEmbeddings(model="llama3.2")
embedding_model2 = OllamaEmbeddings(model="nomic-embed-text")

emb1_vector_1 = embedding_model1.embed_query(all_splits[0].page_content) #len = 3072
emb1_vector_2 = embedding_model1.embed_query(all_splits[1].page_content) #len = 3072

emb2_vector_1 = embedding_model2.embed_query(all_splits[0].page_content) #len = 768
emb2_vector_2 = embedding_model2.embed_query(all_splits[1].page_content) #len = 768

assert len(emb1_vector_1) == len(emb1_vector_2)
print(f"Generated vectors of length {len(emb1_vector_1)} for llama3.2")
print(f"Generated vectors of length {len(emb2_vector_1)} for nomic-embed-text")

print(emb1_vector_1[:10])
print(emb1_vector_2[:10])
print(emb2_vector_1[:10])
print(emb2_vector_2[:10])

# 04. Vector store

from langchain_chroma import Chroma

vector_store = Chroma(embedding_function=embedding_model2)
# vector_store.reset_collection() # always restarts at zero, so this is not needed
print(f"Chroma persist folder: {vector_store._persist_directory}")

if vector_store._chroma_collection.count() == 0: # does nothing / not persistent
    ids = vector_store.add_documents(documents=all_splits) # also support update / requires ids

print(ids[:5])
print("Elements in chroma collection:", vector_store._chroma_collection.count())

results = vector_store.similarity_search("Quem é João da Ega?") 
print("# results: ", len(results), "\nFetched chunk: ", results[0])

results = vector_store.similarity_search_with_score("Quem é João da Ega?")
for result, score in results:
   print(f"Score: {score}\n\nText: {result}\n")
   # lower = more similar, higher (1.0) less similar

maria_embedding = embedding_model2.embed_query("Quem é Maria Eduarda?")
results = vector_store.similarity_search_by_vector(maria_embedding)
print("Maria Eduarda?", results[0])

# only works in async function, methods started with .axxxx are async
# results = await vector_store.asimilarity_search("Quem é João da Ega?"")
# print(results[0])


## 05. Retrievers

# TBD https://python.langchain.com/docs/tutorials/retrievers/