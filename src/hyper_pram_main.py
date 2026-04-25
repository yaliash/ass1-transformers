from __future__ import annotations
import torch
import os
import lm
from torch import nn, optim
from transformer import TransformerLM
import data

# Device selection for Mac (MPS/CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Training on device: {device}")

if __name__ == "__main__":
    seq_len = 128
    batch_size = 64
    data_path = "../data/en/"
    gradient_clipping = 1.0
    num_batches_to_train = 50000

    tokenizer, tokenized_data = data.load_data(data_path)

    # Define our hyperparameter combinations based on the grid search table
    configs = [
        {"run": 1, "n_layers": 6, "n_heads": 6, "embed_size": 192, "lr": 5e-4, "wd": 0.01, "desc": "Baseline"},
        {"run": 2, "n_layers": 4, "n_heads": 6, "embed_size": 384, "lr": 5e-4, "wd": 0.01, "desc": "Wide & Shallow"},
        {"run": 3, "n_layers": 8, "n_heads": 4, "embed_size": 128, "lr": 5e-4, "wd": 0.01, "desc": "Deep & Narrow"},
        {"run": 4, "n_layers": 6, "n_heads": 8, "embed_size": 256, "lr": 1e-3, "wd": 0.1,  "desc": "High Cap + Strong Reg"},
        {"run": 5, "n_layers": 3, "n_heads": 4, "embed_size": 128, "lr": 1e-3, "wd": 0.01, "desc": "Micro Test"},
    ]

    for cfg in configs:
        print(f"\n{'='*50}")
        print(f"Starting Run {cfg['run']}: {cfg['desc']}")
        print(f"Params: Layers={cfg['n_layers']}, Heads={cfg['n_heads']}, Embed={cfg['embed_size']}, LR={cfg['lr']}, WD={cfg['wd']}")
        print(f"{'='*50}")

        mlp_hidden_size = cfg['embed_size'] * 4
        checkpoint_path = f"transformer_run_{cfg['run']}_L{cfg['n_layers']}_H{cfg['n_heads']}_E{cfg['embed_size']}.pth"

        # Re-initialize data iterator for each run
        data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

        # Initialize Model
        model: torch.nn.Module = TransformerLM(
            cfg['n_layers'],
            cfg['n_heads'],
            cfg['embed_size'],
            seq_len,
            tokenizer.vocab_size(),
            mlp_hidden_size,
            with_residuals=True,
        ).to(device)

        # Initialize Optimizer with specific Weight Decay
        optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], betas=[0.9, 0.95], weight_decay=cfg['wd'])
        
        # Initialize OneCycleLR Scheduler (Handles Warmup + Cosine Decay automatically)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=cfg['lr'], 
            total_steps=num_batches_to_train,
            pct_start=0.1 # Warmup for the first 10% of batches
        )

        start_batch = 0
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_batch = checkpoint['num_batches']
            print(f"Resuming from batch {start_batch}")
        else:
            print("No checkpoint found. Starting from scratch.")

        model.train()
        num_batches = start_batch
        last_loss = None
        
        try:
            for batch in data.batch_items(data_iter, batch_size):
                if num_batches >= num_batches_to_train:
                    print(f"Reached target batch count for Run {cfg['run']}. Training complete.")
                    break

                batch_x, batch_y = lm.batch_to_labeled_samples(batch)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                logits = model(batch_x)
                loss = lm.compute_loss(logits, batch_y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                
                optimizer.step()
                scheduler.step() # Update learning rate per batch

                num_batches += 1
                last_loss = loss.item()

                if num_batches % 10 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"[Run {cfg['run']}] Batch {num_batches} | Loss: {last_loss:.4f} | LR: {current_lr:.6f}")

                if num_batches % 100 == 0:
                    for _ in range(1):
                        model.eval()
                        sampled = tokenizer.detokenize(
                            model.better_sample_continuation(
                                prefix=tokenizer.tokenize("Hello"),
                                max_tokens_to_generate=50, # Shortened for faster grid searching
                                temperature=0.8, 
                                topK=5
                            )
                        )
                        model.train()
                        print(f"Model sample: '''{sampled}'''\n")

                if num_batches % 1000 == 0:
                    torch.save(
                        {
                            "num_batches": num_batches,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "loss": loss.item(),
                        },
                        checkpoint_path,
                    )

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            user_input = input("Press 'c' to continue to the next config, or any other key to abort all: ")
            if user_input.lower() == 'c':
                pass # Will save final and move to the next iteration of the outer loop
            else:
                break # Breaks out of the outer config loop entirely

        # Save final state for this run
        final_loss = last_loss if last_loss is not None else float("nan")
        torch.save(
            {
                "num_batches": num_batches,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": final_loss,
            },
            checkpoint_path,
        )
        print(f"Final model for Run {cfg['run']} saved to {checkpoint_path}. Total batches: {num_batches}")

    print("All configured runs have finished executing.")
