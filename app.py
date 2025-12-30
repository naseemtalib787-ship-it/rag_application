import os
import pandas as pd
import numpy as np
import faiss
import gradio as gr

from sentence_transformers import SentenceTransformer
from groq import Groq

# =========================
# 🔑 Groq API Key (HF Secrets)
# =========================
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

# =========================
# 🧠 Embedding Model
# =========================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# 📚 Globals
# =========================
faiss_index = None
documents = []

# =========================
# 🔗 Google Drive Knowledge Base (public link)
# =========================
CSV_URL = "https://drive.google.com/uc?id=1Zw_om0Qtdxle8CqSu-lE6P5WlnnQDrxF"

# =========================
# 📥 Load Knowledge Base
# =========================
def load_knowledge_base():
    global faiss_index, documents

    try:
        df = pd.read_csv(CSV_URL)
    except Exception as e:
        return f"❌ Failed to load knowledge base: {str(e)}"

    # Convert each row into a document
    documents = [
        " | ".join([f"{col}: {row[col]}" for col in df.columns])
        for _, row in df.iterrows()
    ]

    embeddings = embedding_model.encode(documents)
    embeddings = np.array(embeddings).astype("float32")

    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

    return f"✅ Knowledge Base Loaded\n📄 Total Records: {len(documents)}"

# =========================
# 🔍 RAG Question Answering
# =========================
def ask_question(question):
    if faiss_index is None:
        return "❌ Knowledge base not loaded."

    query_embedding = embedding_model.encode([question]).astype("float32")
    distances, indices = faiss_index.search(query_embedding, k=3)

    # Relevance check
    if distances[0][0] > 2.5:
        return "### ❌ Answer\nI don’t know."

    context = "\n".join([documents[i] for i in indices[0]])

    prompt = f"""
Answer the question using ONLY the information below.
If the answer is not present, say exactly: "I don’t know."

Information:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content

    return f"### ✅ Answer\n{answer}"

# =========================
# 🎨 HF-Friendly UI
# =========================
css = """
body {
    background: linear-gradient(135deg, #1d2671, #c33764);
}

.gradio-container {
    font-family: 'Inter', sans-serif;
}

h1 {
    text-align: center;
    font-size: 42px;
    font-weight: 900;
    color: white;
}

.panel {
    background: white;
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.25);
}

.answer-box {
    background: #f8fafc;
    border-left: 8px solid #6366f1;
    padding: 20px;
    border-radius: 14px;
    font-size: 16px;
    line-height: 1.7;
}

button {
    background: linear-gradient(90deg, #6366f1, #22d3ee);
    color: white;
    font-weight: 800;
    border-radius: 14px;
    padding: 12px;
}
"""

# =========================
# 🖥️ Gradio App
# =========================
with gr.Blocks(css=css) as demo:
    gr.Markdown("<h1>📚 Private Knowledge RAG Assistant</h1>")

    with gr.Column(elem_classes="panel"):
        load_btn = gr.Button("📥 Load Knowledge Base")
        status = gr.Textbox(label="System Status", interactive=False)

    load_btn.click(load_knowledge_base, outputs=status)

    with gr.Column(elem_classes="panel"):
        gr.Markdown("### ❓ Ask a Question")
        question = gr.Textbox(
            placeholder="Ask anything from the knowledge base...",
            lines=2
        )
        ask_btn = gr.Button("🚀 Get Answer")

        answer = gr.Markdown(
            value="### ✅ Answer\nYour answer will appear here.",
            elem_classes="answer-box"
        )

    ask_btn.click(ask_question, inputs=question, outputs=answer)

demo.launch()
