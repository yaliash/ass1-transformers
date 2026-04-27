import torch
import os
import math
import shutil
import data
import lm
from transformer import TransformerLM

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Evaluating on device: {device}")

configs = [
    {"run": 1, "n_layers": 6, "n_heads": 6, "embed_size": 192, "desc": "Baseline"},
    {"run": 2, "n_layers": 4, "n_heads": 6, "embed_size": 384, "desc": "Wide & Shallow"},
    {"run": 3, "n_layers": 8, "n_heads": 4, "embed_size": 128, "desc": "Deep & Narrow"},
    {"run": 4, "n_layers": 6, "n_heads": 8, "embed_size": 256, "desc": "High Cap + Strong Reg"},
    {"run": 5, "n_layers": 3, "n_heads": 4, "embed_size": 128, "desc": "Micro Test"},
]

def evaluate_model(model, data_iter, num_eval_batches=100, batch_size=64):
    """Runs the model over N batches and returns the average loss."""
    model.eval() # Turn off dropout, etc.
    total_loss = 0.0
    
    with torch.no_grad(): # Don't compute gradients to save memory and time
        batch_count = 0
        for batch in data.batch_items(data_iter, batch_size):
            if batch_count >= num_eval_batches:
                break
            
            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = model(batch_x)
            loss = lm.compute_loss(logits, batch_y)
            
            total_loss += loss.item()
            batch_count += 1
            
    return total_loss / max(1, batch_count)

def main():
    seq_len = 128
    
    # IMPORTANT: For real evaluation, point this to text files the model HAS NOT seen.
    # If you only have the training data, this will still give a stable average.
    eval_data_path = "../data/en/" 
    
    print("Loading evaluation data and tokenizer...")
    tokenizer, tokenized_data = data.load_data(eval_data_path)
    
    best_ppl = float('inf')
    best_model_path = None

    print("\nStarting rigorous evaluation...")
    print(f"{'Run':<5} | {'Description':<25} | {'Avg Loss':<10} | {'Perplexity':<10}")
    print("-" * 60)

    for cfg in configs:
        checkpoint_path = f"transformer_run_{cfg['run']}_L{cfg['n_layers']}_H{cfg['n_heads']}_E{cfg['embed_size']}.pth"
        
        if not os.path.exists(checkpoint_path):
            continue

        try:
            # 1. Initialize the exact model architecture
            mlp_hidden_size = cfg['embed_size'] * 4
            model = TransformerLM(
                cfg['n_layers'],
                cfg['n_heads'],
                cfg['embed_size'],
                seq_len,
                tokenizer.vocab_size(),
                mlp_hidden_size,
                with_residuals=True,
            ).to(device)

            # 2. Load the trained weights
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])

            # 3. Create a fresh, deterministic data iterator so every model sees the exact same batches
            # Using a fixed random seed ensures a fair apples-to-apples comparison
            torch.manual_seed(42) 
            data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

            # 4. Evaluate over 100 batches to get a stable average
            avg_loss = evaluate_model(model, data_iter, num_eval_batches=100)
            
            # 5. Calculate Perplexity
            perplexity = math.exp(avg_loss)

            print(f"{cfg['run']:<5} | {cfg['desc']:<25} | {avg_loss:<10.4f} | {perplexity:<10.2f}")

            if perplexity < best_ppl:
                best_ppl = perplexity
                best_model_path = checkpoint_path

        except Exception as e:
            print(f"Error evaluating Run {cfg['run']}: {e}")

    if best_model_path:
        print("\n" + "="*60)
        print(f"🏆 SMARTEST PICK: {best_model_path}")
        print(f"With a winning perplexity of: {best_ppl:.2f}")
        
        output_name = "smarter_best_model.pth"
        shutil.copyfile(best_model_path, output_name)
        print(f"Copied the best model to -> '{output_name}'")

if __name__ == "__main__":
    main()