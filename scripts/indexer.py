from sentence_transformers import SentenceTransformer
import faiss, os

model = SentenceTransformer("sentence-transformers/LaBSE")
texts, paths = [], []

for root, _, files in os.walk("logs"):
    for file in files:
        if file.endswith(".md"):
            path = os.path.join(root, file)
            texts.append(open(path).read())
            paths.append(path)

vecs = model.encode(texts)
index = faiss.IndexFlatL2(vecs.shape[1])
index.add(vecs)
faiss.write_index(index, "index/index.faiss")
with open("index/paths.txt", "w") as f:
    f.write("\n".join(paths))
