import torch

def compute_steering_vector(embeddings_1, embeddings_2, save_path=None):
    V1 = embeddings_1.mean(dim=0)
    V2 = embeddings_2.mean(dim=0)
    V = V1 - V2
    if save_path:
        torch.save(V, save_path)
    return V
