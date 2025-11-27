# CoNaN-blast (v0.0.1)

**CoNaN-blast** (**Co**nvolutional **N**eur**a**l **N**etwork blast) is a simple neural network Proof-Of-Concept for searching protein databases to find sequences similar to a query protein. Unlike traditional alignment-based methods (like BLAST), CoNaN-blast uses a Convolutional Neural Network to encode protein sequences into dense vector embeddings. The search is then performed by finding the nearest neighbors in this vector space using cosine similarity, allowing for the detection of structural or functional homologies that might be missed by sequence identity alone.

## Architecture

The core of CoNaN-blast is a **ProteinConvNetEncoder**. It is designed to capture local motifs and patterns in protein sequences using parallel convolutional filters of varying sizes.

**Schematic:**

```
    Input[Protein Sequence] --> Emb[Embedding Layer]
    Emb --> Conv1[Conv1D (k=3)]
    Emb --> Conv2[Conv1D (k=5)]
    Emb --> Conv3[Conv1D (k=9)]
    Emb --> Conv4[Conv1D (k=15)]
    
    Conv1 --> ReLU1[ReLU]
    Conv2 --> ReLU2[ReLU]
    Conv3 --> ReLU3[ReLU]
    Conv4 --> ReLU4[ReLU]
    
    ReLU1 --> Pool1[Global Max Pool]
    ReLU2 --> Pool2[Global Max Pool]
    ReLU3 --> Pool3[Global Max Pool]
    ReLU4 --> Pool4[Global Max Pool]
    
    Pool1 --> Concat[Concatenate]
    Pool2 --> Concat
    Pool3 --> Concat
    Pool4 --> Concat
    
    Concat --> FC[Fully Connected Layer]
    FC --> Norm[L2 Normalization]
    Norm --> Output[256-dim Embedding Vector]
```

1.  **Embedding**: Amino acids are embedded into 128-dimensional vectors.
2.  **Convolution**: 4 parallel convolutional layers with kernel sizes [3, 5, 9, 15] scan the sequence to capture motifs of different lengths (256 filters each).
3.  **Pooling**: Global Max Pooling extracts the most prominent features detected by each filter across the entire sequence.
4.  **Projection**: Features are concatenated and projected to a 256-dimensional output space.

## Search

The `search.py` script allows you to query the database. It converts your query sequence into an embedding using the trained model and finds the top 10 most similar sequences from the pre-computed database.

### Usage

**Command Line Mode:**
Just pass one-line sequences directly as an argument. EG B3GNT2 from `https://www.uniprot.org/uniprotkb/Q9NY97/entry`
```bash
python search.py "MSVGRRRIKLLGILMMANVFIYFIMEVSKSSSQEKNGKGEVIIPKEKFWKISTPPEAYWNREQEKLNRQYNPILSMLTNQTGEAGRLSNISHLNYCEPDLRVTSVVTGFNNLPDRFKDFLLYLRCRNYSLLIDQPDKCAKKPFLLLAIKSLTPHFARRQAIRESWGQESNAGNQTVVRVFLLGQTPPEDNHPDLSDMLKFESEKHQDILMWNYRDTFFNLSLKEVLFLRWVSTSCPDTEFVFKGDDDVFVNTHHILNYLNSLSKTKAKDLFIGDVIHNAGPHRDKKLKYYIPEVVYSGLYPPYAGGGGFLYSGHLALRLYHITDQVHLYPIDDVYTGMCLQKLGLVPEKHKGFRTFDIEEKNKNNICSYVDLMLVHSRKPQEMIDIWSQLQSAHLKC"
```

*Note: The first run will process the database and cache embeddings to `database.pt`, which may take some time. Subsequent runs will be instant.*

## Training

The model was trained using a self-supervised contrastive learning approach (following SimCLR (Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf)).

*   **Goal**: To learn a representation where augmented views (randomly masked versions) of the same protein sequence are close in vector space, while distinct sequences are far apart.
*   **Objective Function**: **Triplet Margin Loss** / **Supervised Contrastive Loss** (Self-Supervised mode). The model minimizes the distance between "positive" pairs (augmentations of the same sequence) and maximizes the distance between "negative" pairs (other sequences in the batch).
*   **Dataset**: Reviewed Uniprot enzymes with an associated BRENDA EC number, **filtered for length (<= 550 AA)** and quality (removed fragmentary sequences).
*   **Training Time**: Approximately **9 hours on CPU**.

### Files

*   `training.py`: The script used to train the model
*   `search.py`: The inference tool for searching the database
*   `model.pth`: model weights (for reviewed enzymes only)
*   `database.pt`: A cached pytorch file containing the precomputed embeddings and headers for the database sequences.

### Requirements

*   python 3.11
*   pytorch
*   numpy
*   tqdm
*   scikit-learn

## Future Directions

*   **Hugging Face Module**: Create a huggingface module to allow users to easily input sequences and visualize results without using the command line.
*   **Expand Training Data**: Train on UniRef50 (currently compute-limited).
*   **Optimization**: Implement FAISS or other vector search libraries to speed up the search process for very large databases.
