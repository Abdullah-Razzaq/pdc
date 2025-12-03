from src.data_loader import load_docs
from src.vectorstore import CHROMAVectorStore
from src.search import RAGRetriever
from src.embedding import EmbeddingPipeline

if __name__ == "__main__":
    
    #docs = load_docs(data_directory='./data')
    store = CHROMAVectorStore()
    #store.load_and_add_docs(documents=docs)
    store.load()
    r = RAGRetriever(vector_store=store, llm_repo='openai/gpt-oss-120b')
    response = r.search(query='what is the Section 121 of the Insurance Ordinance, 2000')
    print(response)