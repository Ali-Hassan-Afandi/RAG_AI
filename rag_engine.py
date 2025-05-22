from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai

model = SentenceTransformer('all-MiniLM-L6-v2')

genai.configure(api_key="YOUR_API_KEY")
llm = genai.GenerativeModel(model_name="gemini-1.5-flash")

class RAG:
    def __init__(self):
        self.index = None
        self.text_chunks = []

    def chunk_text(self, text, size=300):
        return [text[i:i+size] for i in range(0, len(text), size)]

    def create_embeddings(self, texts):
        return model.encode(texts)

    def build_index(self, chunks):
        self.text_chunks = chunks
        embeddings = self.create_embeddings(chunks)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))

    def retrieve(self, query, k=3):
        q_emb = model.encode([query])
        distances, indices = self.index.search(np.array(q_emb), k)
        return [self.text_chunks[i] for i in indices[0]]

    def answer_question(self, query):
        retrieved = self.retrieve(query)
        context = "\n".join(retrieved)
        prompt = f"Context: {context}\n\nQuestion: {query}"
        response = llm.generate_content(prompt)
        return response.text
