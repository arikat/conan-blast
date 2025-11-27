import torch
import numpy as np
import sys
import os
import argparse
from tqdm import tqdm
import torch.nn.functional as F

try:
    from training import (
        ProteinConvNetEncoder, 
        parse_uniref50_fasta, 
        integer_encode, 
        PAD_TOKEN_ID, 
        AA, 
        get_embedding
    )
except ImportError:
    print("Error: Could not import from simple_NN_for_protein_sequence.py")
    print("Make sure you are running this script from the same directory as simple_NN_for_protein_sequence.py")
    sys.exit(1)

def batch_encode_sequences(sequences, model, device, max_len=512, batch_size=128):
    """
    Generates embeddings for a list of sequences in batches.
    """
    embeddings = []
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Generating Embeddings"):
            batch_seqs = sequences[i:i + batch_size]
            tensor_batch = []
            for seq in batch_seqs:
                encoded = integer_encode(seq)
                if len(encoded) > max_len:
                    encoded = encoded[:max_len]
                padded = np.full(max_len, PAD_TOKEN_ID, dtype=np.int64)
                padded[:len(encoded)] = encoded
                tensor_batch.append(padded)
            
            tensor_batch = torch.tensor(np.array(tensor_batch)).to(device)
            
            batch_embeddings = model(tensor_batch)
            embeddings.append(batch_embeddings.cpu())
            
    return torch.cat(embeddings, dim=0)

def main():
    parser = argparse.ArgumentParser(description="Search for similar protein sequences.")
    parser.add_argument("sequence", nargs="?", help="Input protein sequence string")
    args = parser.parse_args()

    MODEL_PATH = 'model.pth'
    UNIREF_PATH = './uniref50/uniref50.fasta.gz'
    DB_LIMIT = None  # set to None, otherwise integer <200000 for testing
    
    EMBEDDING_DIM = 128
    OUTPUT_DIM = 256
    NUM_FILTERS = 256
    FILTER_SIZES = [3, 5, 9, 15]
    MAX_SEQ_LEN = 512
    
    # Device - default cpu
    device = torch.device("cpu")
    if torch.cuda.is_available():
         device = torch.device("cuda")

    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    model = ProteinConvNetEncoder(
        vocab_size=len(AA) + 1,
        embedding_dim=EMBEDDING_DIM,
        num_filters=NUM_FILTERS,
        filter_sizes=FILTER_SIZES,
        output_dim=OUTPUT_DIM
    ).to(device)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        print("Please ensure the model architecture parameters match the saved model.")
        return

    model.eval()

    # --- 2. Prepare the Database ---
    CACHE_FILE = 'database.pt'
    
    if os.path.exists(CACHE_FILE):
        print(f"Database Cache: {CACHE_FILE}...")
        cache_data = torch.load(CACHE_FILE)
        db_embeddings = cache_data['embeddings']
        db_headers = cache_data['headers']
        db_sequences = cache_data['sequences']
        print(f"Database Embeddings Shape: {db_embeddings.shape}")
    else:
        print(f"Cache not found. Loading {DB_LIMIT} sequences from database...")
        if not os.path.exists(UNIREF_PATH):
             print(f"Error: Database file '{UNIREF_PATH}' not found.")
             return

        db_data = parse_uniref50_fasta(UNIREF_PATH, limit=DB_LIMIT)
        print(f"{len(db_data)} sequences.")
        
        db_headers = [item[0] for item in db_data]
        db_sequences = [item[1] for item in db_data]

        print("Generating embeddings for database...")
        db_embeddings = batch_encode_sequences(db_sequences, model, device, max_len=MAX_SEQ_LEN)
        print(f"Database Embeddings Shape: {db_embeddings.shape}")
        
        print(f"Saving embeddings to cache: {CACHE_FILE}...")
        torch.save({
            'embeddings': db_embeddings,
            'headers': db_headers,
            'sequences': db_sequences
        }, CACHE_FILE)


    if args.sequence:
        query_sequence = args.sequence
    else:
        try:
            print("Enter protein sequence:")
            query_sequence = input().strip()
        except EOFError:
            query_sequence = ""

    if not query_sequence:
        print("Error: No sequence provided.")
        return
    
    print(f"\nQuery Sequence (Length: {len(query_sequence)})")

    query_embedding = get_embedding(model, query_sequence, MAX_SEQ_LEN, device)
    query_tensor = torch.tensor(query_embedding).to("cpu") 

    # Calculate Cosine Similarity (dot product)
    similarities = torch.mm(query_tensor, db_embeddings.T).squeeze(0)

    TOP_K = 10
    top_k_scores, top_k_indices = torch.topk(similarities, k=TOP_K)

    print(f"\n--- Top {TOP_K} Matches ---")
    for i in range(TOP_K):
        idx = top_k_indices[i].item()
        score = top_k_scores[i].item()
        
        header = db_headers[idx]
        seq = db_sequences[idx]
        
        print(f"Match #{i+1}")
        print(f"Similarity Score: {score:.4f}")
        print(f"{header}")
        print(f"{seq}")
        print("-" * 50)

if __name__ == '__main__':
    main()
