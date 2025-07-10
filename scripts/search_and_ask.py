from sentence_transformers import SentenceTransformer
import faiss, requests

query = input("質問は？： ")
model = SentenceTransformer("sentence-transformers/LaBSE")
q_vec = model.encode([query])

index = faiss.read_index("index/index.faiss")
with open("index/paths.txt") as f:
    paths = f.read().splitlines()
texts = [open(p).read() for p in paths]

D, I = index.search(q_vec, k=1)
context = texts[I[0][0]]

res = requests.post("http://localhost:11434/api/generate", json={
    "model": "mistral",
    "prompt": f"以下を参考にして質問に答えて：\n{context}\n\n質問：{query}"
})

print("AIの答え：", res.json()["response"])
