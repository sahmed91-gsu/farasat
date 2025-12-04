import os
import json
import logging
import pandas as pd
import chromadb
import shutil
import tempfile
from tqdm import tqdm
from flask import Flask, render_template, request, Response, stream_with_context
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = "BAAI/bge-large-en-v1.5"
# THE FIX: Use /tmp for everything to avoid permission errors
DB_PATH = os.path.join(tempfile.gettempdir(), "chroma_db_runtime")

# Paths for artifacts
PARQUET_FILE = "data/knowledge_base.parquet"
VECTOR_FILE = "data/vectors.npy"
X_PATH = "data/X_train.csv"
Y_PATH = "data/y_train.csv"

from logic import StreamingSpecialistAgent

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# GLOBAL ASSETS
embed_model = None
chroma_collection = None
X_train_df = None
y_train_df = None

def build_database_internal():
    """
    Forces a database build in the writable /tmp directory.
    """
    import numpy as np
    print("⚙️  BUILDING DATABASE FROM ARTIFACTS (SELF-HEALING)...")
    
    if not os.path.exists(PARQUET_FILE) or not os.path.exists(VECTOR_FILE):
        print(f"❌ CRITICAL ERROR: Artifacts missing.")
        return False

    try:
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
            
        # Build in /tmp
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.create_collection(name="production_db", metadata={"hnsw:space": "cosine"})
        
        df = pd.read_parquet(PARQUET_FILE)
        vectors = np.load(VECTOR_FILE)
        
        # Batch Insert
        batch_size = 5000
        total = len(df)
        print(f"   Indexing {total} records...")
        
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            collection.add(
                ids=[str(k) for k in range(i, end)],
                documents=df["document"].iloc[i:end].tolist(),
                embeddings=vectors[i:end].tolist(),
                metadatas=df["metadata"].iloc[i:end].tolist()
            )
        print("✅ Database Built Successfully in /tmp.")
        return True
    except Exception as e:
        print(f"❌ BUILD FAILED: {e}")
        return False

def boot_system():
    global embed_model, chroma_collection, X_train_df, y_train_df
    print("\n" + "="*50 + "\n🚀 SYSTEM BOOT SEQUENCE\n" + "="*50)

    # 1. LOAD MODEL
    print("1. Loading Embedding Model...")
    embed_model = SentenceTransformer(MODEL_NAME)

    # 2. LOAD STATS
    print("2. Loading Statistics...")
    if os.path.exists(X_PATH) and os.path.exists(Y_PATH):
        X_train_df = pd.read_csv(X_PATH)
        y_train_df = pd.read_csv(Y_PATH)["TARGET"]
        print(f"   [OK] Loaded {len(X_train_df)} records.")
    else:
        print("   [WARNING] CSVs not found.")

    # 3. CHECK/BUILD DATABASE IN /tmp
    print(f"3. Checking Database at '{DB_PATH}'...")
    
    # IF DB DOES NOT EXIST, BUILD IT NOW
    if not os.path.exists(DB_PATH):
        print("   ⚠️ Database not found on disk. Initiating Auto-Build...")
        success = build_database_internal()
        if not success:
            print("   ❌ CRITICAL: Could not build DB. App will fail.")
            return

    # 4. CONNECT
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        chroma_collection = client.get_collection("production_db")
        count = chroma_collection.count()
        print(f"   ✅ [OK] Connected to Database ({count} records).")
    except Exception as e:
        print(f"   ❌ CONNECTION ERROR: {e}")

# Run Boot
boot_system()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_stream', methods=['POST'])
def predict_stream():
    if not chroma_collection:
        print("⚠️ Chroma collection is None. Retrying connection...")
        try:
            client = chromadb.PersistentClient(path=DB_PATH)
            global chroma_collection
            chroma_collection = client.get_collection("production_db")
        except:
            return {"error": "Database initialization failed. Check server logs."}, 503
    
    req_data = request.json
    raw_input = req_data.get("note", "")
    
    def generate():
        agent = StreamingSpecialistAgent(
            raw_input, 
            embed_model, 
            chroma_collection,
            X_train=X_train_df,
            y_train=y_train_df
        )
        for step in agent.run_stream():
            yield json.dumps(step) + "\n"

    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    req_id = data.get("id")
    actual_outcome = data.get("actual_outcome")
    if not req_id or not actual_outcome: return {"status": "error"}, 400

    updated_lines = []
    found = False
    try:
        if os.path.exists("audit_history.jsonl"):
            with open("audit_history.jsonl", "r") as f:
                lines = f.readlines()
            for line in lines:
                record = json.loads(line)
                if record.get("id") == req_id:
                    record["ground_truth"] = actual_outcome
                    record["feedback_score"] = 1 if record["prediction"] == actual_outcome else -1
                    found = True
                updated_lines.append(json.dumps(record) + "\n")
            if found:
                with open("audit_history.jsonl", "w") as f:
                    f.writelines(updated_lines)
                return {"status": "success"}
    except Exception as e: return {"status": "error", "message": str(e)}
    return {"status": "not_found"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)