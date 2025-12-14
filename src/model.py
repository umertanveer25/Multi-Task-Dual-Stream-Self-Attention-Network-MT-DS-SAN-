import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        # We treat "features" as a sequence or project them to a sequence
        attn_output, _ = self.multihead_attn(x, x, x)
        return self.layer_norm(x + attn_output)

class MultiTaskModel(nn.Module):
    def __init__(self, input_dim_cont, input_dim_cat, embed_dim=64, num_heads=4):
        super(MultiTaskModel, self).__init__()
        
        # --- Continuous Stream ---
        self.cont_proj = nn.Linear(input_dim_cont, embed_dim)
        self.cont_attn = SelfAttention(embed_dim, num_heads)
        
        # --- Categorical Stream ---
        # We assume input_dim_cat is the number of one-hot features
        self.cat_proj = nn.Linear(input_dim_cat, embed_dim)
        self.cat_attn = SelfAttention(embed_dim, num_heads)
        
        # --- Fusion ---
        self.fusion_dim = embed_dim * 2
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # --- Task 1: Classification Head ---
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1), # Binary classification
            nn.Sigmoid()
        )
        
        # --- Task 2: Reconstruction Head (Auxiliary) ---
        # Decodes back to original dimensions
        self.decoder_cont = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim_cont),
            nn.Sigmoid() # Cont features are MinMax scaled [0,1]
        )
        
        self.decoder_cat = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim_cat),
            nn.Sigmoid() # Cat features are OneHot (0 or 1), can use Softmax if we split by field, but Sigmoid is simple approx for reconstruction
        )

    def forward(self, x_cont, x_cat):
        # 1. Projection
        # Unsqueeze to make it a sequence of length 1 for Attention if we want global attention
        # OR better: Project to higher dim and treat as 1 token, or break down features?
        # For simplicity in this novel architecture:
        # We project the entire feature vector to an embedding, then apply self-attention 
        # heavily on that embedding to refine internal correlations.
        
        # Cont Stream
        h_cont = self.cont_proj(x_cont) # (B, embed_dim)
        h_cont = h_cont.unsqueeze(1)    # (B, 1, embed_dim)
        h_cont = self.cont_attn(h_cont) # Self-Attention
        h_cont = h_cont.squeeze(1)      # (B, embed_dim)

        # Cat Stream
        h_cat = self.cat_proj(x_cat)    # (B, embed_dim)
        h_cat = h_cat.unsqueeze(1)
        h_cat = self.cat_attn(h_cat)
        h_cat = h_cat.squeeze(1)
        
        # Fusion
        fused = torch.cat([h_cont, h_cat], dim=1) # (B, 2*embed_dim)
        latent = self.fusion_layer(fused)         # (B, 128)
        
        # Outputs
        pred_label = self.classifier(latent)
        
        recon_cont = self.decoder_cont(latent)
        recon_cat = self.decoder_cat(latent)
        
        return pred_label, recon_cont, recon_cat

if __name__ == "__main__":
    # Test Instantiation
    model = MultiTaskModel(30, 10)
    print(model)
    print("Model architecture ready.")
