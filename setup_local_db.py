import pandas as pd
import numpy as np
import chromadb
import os
from tqdm import tqdm

# --- CONFIG ---
PARQUET_FILE = os.path.join("data", "knowledge_base.parquet")
VECTOR_FILE = os.path.join("data", "vectors.npy")
DB_PATH = "./local_chroma_db" # We build this folder locally

def build_local_db():
    print(f"--- BUILDING LOCAL DATABASE FROM ARTIFACTS ---")
    
    # 1. Check Files
    if not os.path.exists(PARQUET_FILE) or not os.path.exists(VECTOR_FILE):
        print("Error: Artifacts missing.")
        return

    # 2. Initialize Persistent DB (Saves to disk, doesn't eat RAM)
    print(f"1. Creating database at '{DB_PATH}'...")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Reset if exists to ensure clean build
    try: client.delete_collection("production_db")
    except: pass
    
    collection = client.create_collection(
        name="production_db", 
        metadata={"hnsw:space": "cosine"}
    )

    # 3. Load Data in Stream (To save RAM)
    print("2. Reading Artifacts...")
    df = pd.read_parquet(PARQUET_FILE)
    vectors = np.load(VECTOR_FILE)
    total_records = len(df)
    print(f"   Loaded {total_records} records.")

    # 4. Batch Insert (Crucial for preventing RAM freeze)
    BATCH_SIZE = 2000 
    print("3. Indexing Data (Batched)...")
    
    for i in tqdm(range(0, total_records, BATCH_SIZE)):
        end = min(i + BATCH_SIZE, total_records)
        
        collection.add(
            ids=[str(k) for k in range(i, end)],
            documents=df["document"].iloc[i:end].tolist(),
            embeddings=vectors[i:end].tolist(),
            metadatas=df["metadata"].iloc[i:end].tolist()
        )

    print("\nSUCCESS: Database built at './local_chroma_db'")
    print("   You can now run 'python app.py' without RAM issues.")

if __name__ == "__main__":
    build_local_db()