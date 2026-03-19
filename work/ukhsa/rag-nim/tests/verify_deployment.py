import requests
import json
import time
import os
import argparse
import sys

# Default URLs
RAG_URL = "http://localhost:8081"
INGEST_URL = "http://localhost:8082"
COLLECTION_NAME = "verification_test"
SAMPLE_FILE = "data/multimodal/woods_frost.pdf"

def main():
    parser = argparse.ArgumentParser(description="Verify RAG Deployment")
    parser.add_argument("--rag-url", default=RAG_URL, help=f"RAG server URL (default: {RAG_URL})")
    parser.add_argument("--ingest-url", default=INGEST_URL, help=f"Ingestor server URL (default: {INGEST_URL})")
    parser.add_argument("--file", default=SAMPLE_FILE, help=f"Sample file to upload (default: {SAMPLE_FILE})")
    args = parser.parse_args()

    # Normalize URLs
    rag_url = args.rag_url.rstrip("/")
    ingest_url = args.ingest_url.rstrip("/")

    print("\n🚀 Starting Deployment Verification...")

    # 1. Health Checks
    print("\nStep 1: Checking Health...")
    try:
        rag_health = requests.get(f"{rag_url}/health").json()
        print(f"✅ RAG Server Health: OK")
    except Exception as e:
        print(f"❌ RAG Server Health Check failed: {e}")
        sys.exit(1)

    try:
        ingest_health = requests.get(f"{ingest_url}/health").json()
        print(f"✅ Ingestor Server Health: OK")
    except Exception as e:
        print(f"❌ Ingestor Server Health Check failed: {e}")
        sys.exit(1)

    # 2. Upload Document
    print(f"\nStep 2: Uploading {args.file}...")
    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        sys.exit(1)

    with open(args.file, "rb") as f:
        files = {"documents": (os.path.basename(args.file), f, "application/pdf")}
        data = {
            "data": json.dumps({
                "collection_name": COLLECTION_NAME,
                "blocking": False,
                "split_options": {"chunk_size": 512, "chunk_overlap": 150}
            })
        }
        resp = requests.post(f"{ingest_url}/v1/documents", files=files, data=data)
        if resp.status_code != 200:
            print(f"❌ Upload failed: {resp.text}")
            sys.exit(1)
        
        task_id = resp.json().get("task_id")
        print(f"✅ Upload successful. Task ID: {task_id}")

    # 3. Wait for Ingestion
    print("\nStep 3: Waiting for ingestion to complete...")
    timeout = 120
    start = time.time()
    completed = False
    while time.time() - start < timeout:
        status_resp = requests.get(f"{ingest_url}/v1/status", params={"task_id": task_id}).json()
        state = status_resp.get("state")
        if state == "FINISHED":
            print("✅ Ingestion completed!")
            completed = True
            break
        elif state == "FAILED":
            print(f"❌ Ingestion failed: {status_resp}")
            sys.exit(1)
        print(f"⏳ Status: {state}...")
        time.sleep(5)
    
    if not completed:
        print("❌ Ingestion timed out.")
        sys.exit(1)

    # 4. Perform Query
    print("\nStep 4: Performing Query...")
    query = "What is the poem 'Woods in Frost' about?"
    payload = {
        "messages": [{"role": "user", "content": query}],
        "model": "meta/llama-3.1-70b-instruct", # Typical model name, will be routed to NIM
        "collection_names": [COLLECTION_NAME]
    }
    
    try:
        start_query = time.time()
        resp = requests.post(f"{rag_url}/generate", json=payload)
        resp.raise_for_status()
        result = resp.json()
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        print(f"\nQuery: {query}")
        print(f"Answer: {answer[:300]}...")
        print(f"Time Taken: {time.time() - start_query:.2f}s")
        
        if "frost" in answer.lower() or "woods" in answer.lower():
            print("\n✅ Verification SUCCESS: Response is grounded in retrieved context.")
        else:
            print("\n⚠️ Verification WARNING: Response might not be correctly grounded.")
    except Exception as e:
        print(f"❌ Query failed: {e}")

    # 5. Cleanup (Optional)
    print("\nStep 5: Cleaning up...")
    try:
        requests.delete(f"{ingest_url}/v1/collections", params={"collection_name": COLLECTION_NAME})
        print("✅ Cleanup complete.")
    except:
        pass

    print("\n✨ Deployment verification finished.")

if __name__ == "__main__":
    main()
