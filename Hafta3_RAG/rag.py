"""Amac:
1- FAISS Vector DB'i guncelle
2- Qwen3-Embedding modelini yukleme
3- Ollama uzerinden gemma3:4B modelini kullan
4- Kullanici sorusunu embedding'e cevirmek
5- VectorDB den en yakin 3 chunk'i getirmek
6- Qwen3-Reranker-0.6B ile chunk'lari yeniden siralamak
7- Chunk'lari LLM'e gonder ve cevap uretmek
"""

import json
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder
from ollama import chat

# Modelleri ve VectorDB'i yukle

embedding_model=SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
reranker = CrossEncoder('Qwen/Qwen3-Reranker-0.6B')

index = faiss.read_index('vector_db/character/index.faiss')

with open('vector_db/character/chunks.json', 'r',encoding='utf-8') as f:
    chunks = json.load(f)

print('VectorDB yuklendi')
print('Embedding modeli yuklendi')
print('Reranker modeli yuklendi')
print('LLM: gemma3:4B')

# Kullanicidan soru al

question = input('\nSorunuz:')

print("\nKullanici Sorusu:")
print(question)

# Soruyu Embedding'e Cevir

question_embedding = embedding_model.encode([question], normalize_embeddings=True).astype('float32')


print("\nSoru embedding'e cevrildi")
print('Embedding boyutu:', question_embedding.shape)

# VectorDB'den en yakin 3 chunk'i getirme

scores, indices = index.search(question_embedding, 3)

retrieved_chunks = [chunks[i] for i in indices[0]]
print("\nGetirilen Chunks")
for i , chunk in enumerate(retrieved_chunks):
    print(f"\nChunk {i + 1}")
    print(chunk)
    print("Similarity:", scores[0][i])


# Reranker

pairs = [(question, chunk) for chunk in retrieved_chunks]

rerank_score = reranker.predict(pairs)

reranked = sorted(zip(retrieved_chunks, rerank_score), key=lambda x: x[1], reverse=True)
print("\nReranker sonrasi:")

for i, chunk in enumerate(reranked):
    print(f"\nChunk {i + 1}")
    print(chunk)
    print('Rerank Score:',rerank_score)


# Chunk'lari LLM'e gonder

context = '\n\n'.join([chunk for chunk, rerank_score in reranked])

prompt = f"""
Asagidaki bilgileri kullanarak kullanicinin sorusunu cevapla.
Cevap dokumanda yoksa bilmedigini soyle ama selamlama yapan olursa ona da cevap ver

Bilgiler:
{context}

Soru:
{question}
"""
print("\nLLM'e gonderilen context:")
print(context)

response = chat(
    model = 'gemma3:4b',
    messages = [{
        'role':'user',
        'content':prompt
    }]
)
print("\nLLM cevabi:")
print(response.message.content)


