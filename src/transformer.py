from torch import nn
import torch
import torch.nn.functional as F
import attention
import mlp

class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, mlp_hidden_size: int, max_context_len, with_residuals: bool = False, is_prenorm: bool = True):
        super().__init__()
        self.causal_attention = attention.CausalSelfAttention(embed_size, n_heads, max_context_len)
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.with_residuals = with_residuals
        self.is_prenorm = is_prenorm

    def forward(self, inputs, return_attention=False):
        x = inputs
        if self.with_residuals:
            if self.is_prenorm:
                if return_attention:
                    attn_out, head_attentions = self.causal_attention(self.layer_norm_1(x), return_attention=True)
                    x = x + attn_out
                else:
                    x = x + self.causal_attention(self.layer_norm_1(x))
                x = x + self.mlp(self.layer_norm_2(x))
            else:
                if return_attention:
                    attn_out, head_attentions = self.causal_attention(x, return_attention=True)
                    x = self.layer_norm_1(x + attn_out)
                else:
                    x = self.layer_norm_1(x + self.causal_attention(x))
                x = self.layer_norm_2(x + self.mlp(x))
            return (x, head_attentions) if return_attention else x
        else:
            if self.is_prenorm:
                if return_attention:
                    attn_out, head_attentions = self.causal_attention(self.layer_norm_1(x), return_attention=True)
                    x = attn_out
                else:
                    x = self.causal_attention(self.layer_norm_1(x))
                x = self.mlp(self.layer_norm_2(x))
            else:
                if return_attention:
                    attn_out, head_attentions = self.causal_attention(x, return_attention=True)
                    x = self.layer_norm_1(attn_out)
                else:
                    x = self.layer_norm_1(self.causal_attention(x))
                x = self.layer_norm_2(self.mlp(x))
            return (x, head_attentions) if return_attention else x

class Embed(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_context_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size) # (b, n) -> (b, n, d)
        self.position_embeddings = nn.Embedding(max_context_len, embed_size)
        self.max_context_len = max_context_len

    def forward(self, x):
        # x has the shape (b, n) where b is batch dimension and n is sequence length.
        # each item is an int, indicating a vocabulary item.
        # The output should be of shape (b, n, d), where d is the embedding dimension.
        tok_embeddings = self.token_embeddings(x)
        positions = torch.arange(x.size(-1), device=x.device)
        pos_embeddings = self.position_embeddings(positions)
        return tok_embeddings + pos_embeddings # (b, n, d) + (n, d) -> (b, n, d)


class TransformerLM(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            embed_size: int,
            max_context_len: int,
            vocab_size: int,
            mlp_hidden_size: int,
            with_residuals: bool,
            ):
        super().__init__()
        self.embed = Embed(vocab_size, embed_size, max_context_len)
        self.layers = nn.ModuleList([TransformerDecoderBlock(n_heads, embed_size, mlp_hidden_size, max_context_len, with_residuals) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embed_size)
        self.word_prediction = nn.Linear(embed_size, vocab_size)
        self.max_context_len = max_context_len

        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print("Parameter count: %.2fM" % (n_params/1e6,))

    def forward(self, inputs, return_attention=False, return_hidden_states=False):
        x = self.embed(inputs)
        all_attentions = {}   # {layer_idx: [head_0_weights, ..., head_k_weights]}
        hidden_states  = []   # list of (B, N, D) tensors, one per layer (post-block, pre-LN)

        for layer_idx, layer in enumerate(self.layers):
            if return_attention:
                x, head_attentions = layer(x, return_attention=True)
                all_attentions[layer_idx] = head_attentions
            else:
                x = layer(x)
            if return_hidden_states:
                hidden_states.append(x)

        x = self.layer_norm(x)
        logits = self.word_prediction(x)

        extras = []
        if return_attention:
            extras.append(all_attentions)
        if return_hidden_states:
            extras.append(hidden_states)
        return (logits, *extras) if extras else logits

    def init_weights(self):
        # initialize weights
        for pn, p in self.named_parameters():
            if isinstance(p, nn.LayerNorm):
                torch.nn.init.zeros_(p.bias)
                torch.nn.init.ones_(p.weight)
            elif isinstance(p, nn.Linear):
                torch.nn.init.normal_(p.weight, mean=0.0, std=0.02)
                if p.bias is not None:
                    torch.nn.init.zeros_(p.bias)
            elif isinstance(p, nn.Embedding):
                torch.nn.init.normal_(p.weight, mean=0.0, std=0.02)


    def sample_continuation(self, prefix: list[int], max_tokens_to_generate: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        device = next(self.parameters()).device

        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.long, device=device))
                #logits = self(torch.tensor([feed_to_lm], dtype=torch.int32))
                logits_for_last_token = logits[0][-1]
                distribution_for_last_token = F.softmax(logits_for_last_token, dim=-1)
                sampled_token = int(torch.multinomial(distribution_for_last_token, num_samples=1).item())
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated

    def better_sample_continuation(self, prefix: list[int], max_tokens_to_generate: int, temperature: float, topK: int) -> list[int]:
        device = next(self.parameters()).device 
        feed_to_lm = prefix[:]
        generated = []
        
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
    
                logits = self(torch.tensor([feed_to_lm], dtype=torch.long, device=device))
                logits_for_last_token = logits[0, -1, :]
                
                if temperature > 0.0:
                    logits_for_last_token = logits_for_last_token / temperature
                
                if topK > 0:
                    top_values, _ = torch.topk(logits_for_last_token, topK)
                    min_accepted_value = top_values[-1] 
                    logits_for_last_token[logits_for_last_token < min_accepted_value] = float('-inf')
                
                distribution_for_last_token = F.softmax(logits_for_last_token, dim=-1)
                sampled_token = int(torch.multinomial(distribution_for_last_token, num_samples=1).item())
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated

