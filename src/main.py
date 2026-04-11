from __future__ import annotations
import torch
import os

# Device selection for Mac (MPS/CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Training on device: {device}")

if __name__ == "__main__":
    import lm
    from torch import nn, optim
    from transformer import TransformerLM

    import data

    seq_len = 128
    batch_size = 64
    data_path = "../data/en/"
    n_layers = 6
    n_heads = 6
    embed_size = 192
    mlp_hidden_size = embed_size * 4

    learning_rate = 5e-4
    gradient_clipping = 1.0

    num_batches_to_train = 50000

    checkpoint_path = "transformer_checkpoint.pth"

    tokenizer, tokenized_data = data.load_data(data_path)
    # NOTE: are data items are longer by one than the sequence length, They will be shortened by 1 when converted to training examples.
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

    model: torch.nn.Module = TransformerLM(
        n_layers,
        n_heads,
        embed_size,
        seq_len,
        tokenizer.vocab_size(),
        mlp_hidden_size,
        with_residuals=True,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.9, 0.95])

    # --- Checkpoint Loading ---
    start_batch = 0
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device) # map_location ensures we can load a GPU checkpoint onto a Mac or CPU
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
                print("Reached target batch count. Training complete.")
                break

            # Data prep
            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            logits = model(batch_x)
            loss = lm.compute_loss(logits, batch_y)

            # Optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            num_batches += 1
            last_loss = loss.item()

            # Logging (Every 10)
            if num_batches % 10 == 0:
                print(f"Seen {num_batches} batches. last loss is: {last_loss}")

            # Sampling (Every 100)
                if num_batches % 100 == 0:
                    for _ in range(1):
                        model.eval()
                        sampled = tokenizer.detokenize(
                            model.better_sample_continuation(
                                prefix=tokenizer.tokenize("Hello"),
                                max_tokens_to_generate=100, 
                                temperature=0.8, 
                                topK=5
                            )
                        )
                        model.train()
                        print(f"Model sample: '''{sampled}'''")
                    print("")

                    # Checkpointing (Every 1000)
                    if num_batches % 1000 == 0:
                        print(f"Saving checkpoint to {checkpoint_path}...")
                        torch.save(
                            {
                                "num_batches": num_batches,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": loss.item(),
                            },
                            checkpoint_path,
                        )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")

    # Final Save
    final_loss = last_loss if last_loss is not None else float("nan")
    torch.save(
        {
            "num_batches": num_batches,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": final_loss,
        },
        "transformer_final.pth",
    )
    print(f"Final model saved. Total batches: {num_batches}")
