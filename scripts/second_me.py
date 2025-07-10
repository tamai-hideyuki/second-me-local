#!/usr/bin/env python3
import os
import sys
import argparse
from datetime import date
import json
import requests
from sentence_transformers import SentenceTransformer
import faiss

def init_logs():
    d = date.today().strftime("%Y-%m-%d")
    base = os.path.join("logs", d)
    os.makedirs(base, exist_ok=True)
    templates = [("01", "chat"), ("02", "thoughts"), ("03", "tasks")]
    for n, name in templates:
        path = os.path.join(base, f"{n}_{name}.md")
        if not os.path.exists(path):
            with open(path, "w", encoding='utf-8') as f:
                f.write(f"# {name.title()}（{d}）\n\n")
    print(f"Initialized log templates under {base}/")


def index_embeddings(model_name="sentence-transformers/LaBSE"):
    model = SentenceTransformer(model_name)
    texts, paths = [], []
    for root, _, files in os.walk("logs"):
        for file in files:
            if file.endswith(".md"):
                path = os.path.join(root, file)
                with open(path, encoding='utf-8') as f:
                    texts.append(f.read())
                paths.append(path)
    if not texts:
        print("No logs found. Run init or add logs first.")
        return
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    os.makedirs("index", exist_ok=True)
    faiss.write_index(index, "index/index.faiss")
    with open("index/paths.txt", "w", encoding='utf-8') as f:
        f.write("\n".join(paths))
    print(f"Indexed {len(texts)} log files into index/index.faiss")


def call_llm(payload, model_api_url="http://localhost:11434/api/generate"):
    # Handle streaming JSON lines or full JSON
    res = requests.post(model_api_url, json=payload, stream=True)
    res.raise_for_status()
    text = ''
    for line in res.iter_lines():
        # Each line is a JSON object representing a chunk
        try:
            data = json.loads(line.decode('utf-8'))
        except json.JSONDecodeError:
            continue  # skip non-JSON
        # Append any text in 'response' or 'output_text'
        chunk = data.get('response') or data.get('output_text')
        if chunk:
            text += chunk
        # If the chunk indicates completion, stop reading
        if data.get('done') is True:
            break
        if not line:
            continue
        try:
            data = json.loads(line.decode('utf-8'))
            chunk = data.get('response') or data.get('output_text')
            if chunk:
                text += chunk
        except json.JSONDecodeError:
            # ignore non-JSON lines
            continue
    return text.strip()


def ask_question(prompt, model_api_url="http://localhost:11434/api/generate", llm_model="mistral"):
    try:
        index = faiss.read_index("index/index.faiss")
    except Exception:
        print("Index not found. Run 'second_me.py index' first.")
        return
    with open("index/paths.txt", encoding='utf-8') as f:
        paths = f.read().splitlines()
    embed_model = SentenceTransformer("sentence-transformers/LaBSE")
    q_vec = embed_model.encode([prompt])
    D, I = index.search(q_vec, k=1)
    context = ''
    if I and I[0].size > 0:
        context = open(paths[I[0][0]], encoding='utf-8').read()
    payload = {"model": llm_model, "prompt": f"以下の情報を参考にして答えてください：\n{context}\n質問：{prompt}", "stream": True}
    answer = call_llm(payload, model_api_url)
    # save to chat log
    d = date.today().strftime("%Y-%m-%d")
    log_dir = os.path.join("logs", d)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "01_chat.md")
    with open(log_path, "a", encoding='utf-8') as f:
        f.write(f"Q: {prompt}\nA: {answer}\n\n")
    print("AIの答え：", answer)


def summarize_day(model_api_url="http://localhost:11434/api/generate", llm_model="mistral"):
    d = date.today().strftime("%Y-%m-%d")
    log_path = os.path.join("logs", d, "01_chat.md")
    if not os.path.exists(log_path):
        print("No chat log for today to summarize.")
        return
    text = open(log_path, encoding='utf-8').read()
    prompt = f"以下のログを要約し、重要なポイント・ToDo・気づきを3つずつ出力してください。\n{text}"
    payload = {"model": llm_model, "prompt": prompt, "stream": True}
    summary = call_llm(payload, model_api_url)
    summary_path = os.path.join("logs", d, "00_summary.md")
    with open(summary_path, "w", encoding='utf-8') as f:
        f.write(summary)
    print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Second Me CLI")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("init", help="Initialize log templates")
    subparsers.add_parser("index", help="Index logs for semantic search")
    ask_p = subparsers.add_parser("ask", help="Ask a question with semantic context")
    ask_p.add_argument("prompt", nargs='+', help="Question prompt")
    subparsers.add_parser("summarize", help="Summarize today's chat log")
    args = parser.parse_args()

    if args.command == "init":
        init_logs()
    elif args.command == "index":
        index_embeddings()
    elif args.command == "ask":
        prompt = " ".join(args.prompt)
        ask_question(prompt)
    elif args.command == "summarize":
        summarize_day()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
