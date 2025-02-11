import faiss
from langchain.chat_models import init_chat_model
from langchain_community.docstore import InMemoryDocstore
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
index = faiss.IndexFlatL2(len(embeddings.embed_query("should be 3072 but let's see")))
vector_store = FAISS(embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

loader = PyPDFLoader(
    file_path="pdfs/brochure.pdf",
    mode = "single",
    pages_delimiter = "\n\f",
)

docs = []
docs_lazy = loader.lazy_load()

for doc in docs_lazy:
    docs.append(doc)
print(docs[0].page_content[:500])
print(docs[0].metadata)
print(f"Total characters: {len(docs[0].page_content)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
)
all_splits = text_splitter.split_documents(docs)

print(f"Split docs into {len(all_splits)} sub-documents.")

document_ids = vector_store.add_documents(all_splits)
print(document_ids[:3])
vector_store.save_local("faiss_store")

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

vector_store = FAISS.load_local("faiss_store",
                                embeddings,
                                allow_dangerous_deserialization=True)

results = vector_store.similarity_search_with_relevance_scores("Does TRD Pro have moonroof?", k=3)

context = ""

for document in results:
    print(document)
    context += context + f" CONTENT: {document[0].page_content} SOURCE: {document[0].metadata}"
    print("\n")

print(context)
rag_prompt = '''You are an assistant for question-answering tasks. Use the following pieces of retrieved context to 
answer the question. If you don't know the answer, just say that you don't know. Always mention the 'source' and 
'title' of the document you use context from. Question: {question} Context: {context} Answer:'''
llm_response = llm.invoke(rag_prompt.format(question="Does TRD Pro have moonroof?", context=context))
print(llm_response)
print('FINAL ANSWER: ' + llm_response.content)