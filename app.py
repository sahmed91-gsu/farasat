import os
import json
import logging
import pandas as pd
import chromadb
from flask import Flask, render_template, request, Response, stream_with_context
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = "BAAI/bge-large-en-v1.5"
DB_PATH = "./local_chroma_db"

from logic import StreamingSpecialistAgent

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# GLOBAL ASSETS
embed_model = None
chroma_collection = None
X_train_df = None
y_train_df = None

DATA_DIR = "data"
X_PATH = os.path.join(DATA_DIR, "X_train.csv")
Y_PATH = os.path.join(DATA_DIR, "y_train.csv")

def boot_system():
    global embed_model, chroma_collection, X_train_df, y_train_df
    print("\n" + "="*50 + "\nSYSTEM BOOT\n" + "="*50)

    print("1. Loading Embedding Model...")
    embed_model = SentenceTransformer(MODEL_NAME)

    print("2. Loading Statistical Data (CSVs)...")
    try:
        if os.path.exists(X_PATH) and os.path.exists(Y_PATH):
            X_train_df = pd.read_csv(X_PATH)
            y_train_df = pd.read_csv(Y_PATH)["TARGET"]
            print(f"   [OK] Loaded {len(X_train_df)} training records.")
        else:
            print("   [WARNING] CSV files not found. Statistical context will be disabled.")
    except Exception as e:
        print(f"   [ERROR] CSV Load Failed: {e}")

    print(f"3. Connecting to Database at '{DB_PATH}'...")
    if not os.path.exists(DB_PATH):
        print("CRITICAL: Database folder not found. Run 'python setup_local_db.py'.")
        return

    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        chroma_collection = client.get_collection("production_db")
        print("   [OK] Database Connected.")
    except Exception as e:
        print(f"DB CONNECTION ERROR: {e}")

boot_system()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_stream', methods=['POST'])
def predict_stream():
    if not chroma_collection:
        return {"error": "Database not connected."}, 503
    
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
    
    if not req_id or not actual_outcome:
        return {"status": "error"}, 400

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
    except Exception as e:
        return {"status": "error", "message": str(e)}

    return {"status": "not_found"}

if __name__ == '__main__':
    # CHANGED FOR DEPLOYMENT:
    # host='0.0.0.0' makes it accessible externally
    # port=7860 is required by Hugging Face Spaces
    app.run(host='0.0.0.0', port=7860, debug=False)