import os
import sys
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DB_PATH=r"D:\python-project\news-nexus\data"

def retrieve_documents(query,k=4,keyword_filter=True):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
    results = vector_store.similarity_search(query, k=k+2)
    final_results=[]

    if keywords:
        query_terms = set(query.lower().split())

        for doc,score in results:
            content = doc.page_content.lower()
            term_machines = sum(1 for term in query_terms if term in content)
            boosted_score = score - (term_machines * 0.5)
            final_results.append((doc, boosted_score))
            final_results.sort(key=lambda x: x[1], reverse=True)
            final_results = final_results[:k]
    else:
        final_results = results[:k]
        return final_results
    
    return results

if __name__=="__main__":
    test_query = "What is the impact of GenAI on productivity?"
    retrieved_docs=retrieve_documents(test_query)

    print(f"\n--Top {len(retrieved_docs)} Results--")
    for i,(doc,score) in enumerate(retrieved_docs):
        print(f"\n [Result {i+1}] (Score:{score:.4f})")
        print(f"Source: {doc.metadata.get('source','unknown')}")
        print(f"Content Snippet: {doc.page_content[:200]}...")