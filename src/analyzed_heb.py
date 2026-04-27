import torch
import data
import lm
import math
from transformer import TransformerLM

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def analyze_hebrew():
    print(f"Evaluating on device: {device}")
    
    # 1. Parameters used in main.py for the Hebrew run
    seq_len = 128
    batch_size = 64
    n_layers = 6
    n_heads = 6
    embed_size = 192
    mlp_hidden_size = embed_size * 4
    
    # Paths (adjust data_path if your Hebrew text is stored elsewhere)
    data_path = "../data/he/" 
    model_path = "transformer_hebrew_final.pth"
    
    print("Loading Hebrew data and tokenizer...")
    tokenizer, tokenized_data = data.load_data(data_path)
    
    print("Initializing model architecture...")
    model = TransformerLM(
        n_layers, n_heads, embed_size, seq_len, 
        tokenizer.vocab_size(), mlp_hidden_size, 
        with_residuals=True
    ).to(device)
    
    print(f"Loading trained weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Recorded Final Training Loss: {checkpoint.get('loss', 'Unknown'):.4f}")
    
    # 2. Evaluate Perplexity
    print("\nCalculating Validation Perplexity (100 batches)...")
    model.eval()
    total_loss = 0.0
    batch_count = 0
    
    torch.manual_seed(42)
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))
    
    with torch.no_grad():
        for batch in data.batch_items(data_iter, batch_size):
            if batch_count >= 100:
                break
            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = model(batch_x)
            loss = lm.compute_loss(logits, batch_y)
            
            total_loss += loss.item()
            batch_count += 1

    avg_loss = total_loss / max(1, batch_count)
    perplexity = math.exp(avg_loss)
    
    print(f"--> Hebrew Avg Validation Loss: {avg_loss:.4f}")
    print(f"--> Hebrew Perplexity: {perplexity:.2f}")

    # 3. Generate a sample of Hebrew text
    print("\nGenerating Hebrew text sample...")
    prefix_tokens = tokenizer.tokenize(" ") # Start with a space
    
    sampled_tokens = model.better_sample_continuation(
        prefix=prefix_tokens,
        max_tokens_to_generate=200,
        temperature=0.8,
        topK=5
    )
    
    sampled_text = tokenizer.detokenize(sampled_tokens)
    print("\n" + "="*40 + " HEBREW SAMPLE " + "="*40)
    print(sampled_text)
    print("="*95 + "\n")

if __name__ == "__main__":
    analyze_hebrew()