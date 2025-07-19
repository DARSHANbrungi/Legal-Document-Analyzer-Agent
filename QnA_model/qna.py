import torch
import chromadb
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai
import json
import numpy as np

# 🔹 Model and Tokenizer Setup
MODEL_NAME = "law-ai/InLegalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# 🔹 Load ChromaDB
chroma_client = chromadb.PersistentClient(path="../chroma_db")
collection = chroma_client.get_or_create_collection(name="legal_qna")

def get_embedding(text):
    """Generates embeddings using InLegalBERT."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def retrieve_context(query, top_k=3, fallback_threshold=5):
    """Retrieves relevant legal contexts from ChromaDB using hybrid RAG."""
    query_embedding = get_embedding(query)
    
    # Step 1: Search using Answer Embeddings
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where={"type": "answer"}
    )
    retrieved_contexts = [res["answer"] for res in results["metadatas"][0]]
    
    # Step 2: Fallback Search if necessary
    if len(retrieved_contexts) < fallback_threshold:
        fallback_results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where={"type": "question"}
        )
        for res in fallback_results["metadatas"][0]:
            if res["answer"] not in retrieved_contexts:
                retrieved_contexts.append(res["answer"])
    
    return " ".join(retrieved_contexts)

# 🔹 Configure Gemini API Key
genai.configure(api_key="AIzaSyBxAdkbQrOysrWgwb-T53xzGseZl9r4TEM")

def answer_question(query):
    """Answers legal questions using Gemini 2.5 with retrieved context."""
    context = retrieve_context(query)
    #if not context:
        #return "No relevant legal information found."
    
    prompt = (
        f"Using the following legal text, answer the question. If the answer is not found, generate it based on Indian law. "
        f"Give the answer without explanation about the context:\n\n"
        f"Legal Text: {context}\n\nQuestion: {query}\n\nAnswer:"
    )
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

if __name__ == "__main__":
    query = input("Enter your legal question: ")
    response = answer_question(query)
    print(f"🔹 Answer: {response}")
