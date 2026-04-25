"""
Part 5: Analysis / Interpretability
Extracts and analyzes attention patterns from the trained transformer LM.

Patterns analysed
-----------------
1.  Previous-token head        : attends to i-1
2.  First-token head           : attends to position 0
3.  Space-Seeking (Delimiter)  : attends to most-recent space / comma / newline
4.  Capitalization Tracker     : attends to most-recent sentence-end (. or \n)
5.  Vowel/Consonant Tracker    : consonant queries → vowel keys (or vice-versa)
6.  Induction Head             : attends to j+1 when chars[j]==chars[i-1]
7.  Hebrew Prefix/Suffix Head  : attends to morphological prefix/suffix chars (Hebrew data)
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────────────
CHECKPOINT     = "transformer_final.pth"
if not os.path.exists(CHECKPOINT):
    CHECKPOINT = "transformer_checkpoint.pth"

# Separate checkpoint for the Hebrew-trained model (must be trained on Hebrew data).
# Set to None to skip Hebrew analysis automatically.
HEB_CHECKPOINT = "transformer_hebrew_final.pth"
if not os.path.exists(HEB_CHECKPOINT):
    HEB_CHECKPOINT = "transformer_hebrew_checkpoint.pth"
if not os.path.exists(HEB_CHECKPOINT):
    HEB_CHECKPOINT = None   # no Hebrew checkpoint → section will be skipped

DATA_PATH    = "../data/en/"
HEB_PATH     = "../data/he/"
PLOTS_DIR    = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── model config (must match training) ────────────────────────────────────────
SEQ_LEN    = 128
N_LAYERS   = 6
N_HEADS    = 6
EMBED_SIZE = 192
MLP_HIDDEN = EMBED_SIZE * 4

# ── character sets ─────────────────────────────────────────────────────────────
VOWELS         = set("aeiouAEIOU")
CONSONANTS     = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")
DELIMITERS     = set(" ,\n")          # word-boundary delimiters
SENTENCE_ENDS  = set(".\n")           # capitalization triggers

# Hebrew morphological characters
HEB_PREFIXES   = set("בלמשהכו")       # ב ל מ ש ה כ ו  – prepositional/conjunctive prefixes
HEB_SUFFIX_1   = ("י", "ם")           # ים  plural suffix (two consecutive chars)
HEB_SUFFIX_2   = ("ו", "ת")           # ות  plural suffix

# ── sample texts ───────────────────────────────────────────────────────────────
ENGLISH_SAMPLES = [
    "To be, or not to be, that is the question:",
    "All the world's a stage, and all the men and women merely players.",
    "Friends, Romans, countrymen, lend me your ears;",
    "What a piece of work is a man! How noble in reason!",
    "Now is the winter of our discontent made glorious summer.",
    "The lady doth protest too much, methinks.",
    "Good night, good night! Parting is such sweet sorrow,",
    "We know what we are, but know not what we may be.",
    "Love looks not with the eyes, but with the mind,",
    "The quality of mercy is not strained.",
    "All that glitters is not gold; Often have you heard that told.",
    "Come what come may, time and the hour runs through the roughest day.",
    "Shakespeare wrote many sonnets. Some scholars believe Shakespeare also wrote plays.",
    "The letter e appears everywhere. Even sentences need the letter e repeatedly.",
]

HEBREW_SAMPLES = [
    "בשדות ובכרמים",
    "השמש זורחת בבוקר",
    "מה יפו לילות בכנען",
    "ושוב ושוב האביב הגיע",
    "בין הנהרות והיערות",
    "לבבות שלמים ועיניים פקוחות",
    "מתחת לכוכבים השמיים",
    "ועוד ילדים משחקים בשדה",
]


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(data_path, checkpoint_path=CHECKPOINT):
    import data as datamod
    from transformer import TransformerLM

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    tokenizer, _ = datamod.load_data(data_path)

    # Read vocab_size from the checkpoint itself so the model shape always matches
    # the saved weights, regardless of which tokenizer is active.
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_vocab_size = ckpt["model_state_dict"]["embed.token_embeddings.weight"].shape[0]

    if ckpt_vocab_size != tokenizer.vocab_size():
        print(f"  WARNING: checkpoint vocab_size={ckpt_vocab_size} "
              f"!= tokenizer vocab_size={tokenizer.vocab_size()}. "
              f"Using checkpoint size – tokenizer may mis-decode rare tokens.")

    model = TransformerLM(
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        embed_size=EMBED_SIZE,
        max_context_len=SEQ_LEN,
        vocab_size=ckpt_vocab_size,
        mlp_hidden_size=MLP_HIDDEN,
        with_residuals=True,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Checkpoint batch={ckpt.get('num_batches','?')}, loss={ckpt.get('loss',float('nan')):.4f}")
    return model, tokenizer, device


# ══════════════════════════════════════════════════════════════════════════════
# Attention extraction
# ══════════════════════════════════════════════════════════════════════════════

def get_attention_for_text(model, tokenizer, text, device):
    """
    Returns:
        chars           : list of characters for the input tokens
        all_attentions  : dict {layer_idx: [head_attn(N,N), ...]}  (numpy, batch dim removed)
    """
    token_ids = tokenizer.tokenize(text)[:SEQ_LEN]
    chars = list(tokenizer.detokenize(token_ids))
    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        _, all_attentions = model(x, return_attention=True)
    result = {li: [h[0].cpu().numpy() for h in heads]
              for li, heads in all_attentions.items()}
    return chars, result


# ══════════════════════════════════════════════════════════════════════════════
# Aggregate statistics
# ══════════════════════════════════════════════════════════════════════════════

def _most_recent_before(chars, i, charset):
    """Return the index of the most recent character in charset at position <= i, or -1."""
    for j in range(i, -1, -1):
        if chars[j] in charset:
            return j
    return -1


def build_aggregate_stats(model, tokenizer, device, sample_texts):
    """
    Computes per-(layer, head) scalar scores across all sample_texts.
    Returns a dict of stat_name -> ndarray of shape (N_LAYERS, N_HEADS).
    """
    prev_token       = np.zeros((N_LAYERS, N_HEADS))
    first_token      = np.zeros((N_LAYERS, N_HEADS))
    delimiter_recency= np.zeros((N_LAYERS, N_HEADS))  # Space-Seeking head
    cap_tracker      = np.zeros((N_LAYERS, N_HEADS))  # Capitalization Tracker
    cons_to_vowel    = np.zeros((N_LAYERS, N_HEADS))  # consonant query → vowel key
    vowel_to_cons    = np.zeros((N_LAYERS, N_HEADS))  # vowel query → consonant key
    induction_match  = np.zeros((N_LAYERS, N_HEADS))  # Induction Head – matching-prefix positions
    induction_base   = np.zeros((N_LAYERS, N_HEADS))  # Induction Head – non-matching positions
    rel_pos_mass     = defaultdict(lambda: np.zeros((N_LAYERS, N_HEADS)))

    n_samples = 0

    for text in sample_texts:
        chars, all_attentions = get_attention_for_text(model, tokenizer, text, device)
        N = len(chars)
        n_samples += 1

        for li in range(N_LAYERS):
            for hi in range(N_HEADS):
                attn = all_attentions[li][hi]  # (N, N)  row=query, col=key

                # ── 1. Previous-token attention ─────────────────────────────
                if N > 1:
                    prev_token[li, hi] += np.mean([attn[i, i-1] for i in range(1, N)])

                # ── 2. First-token attention ─────────────────────────────────
                first_token[li, hi] += attn[:, 0].mean()

                # ── 3. Space-Seeking / Delimiter Head ───────────────────────
                # For each query i, how much attention goes to the most-recent delimiter?
                delim_weights = []
                for i in range(N):
                    j = _most_recent_before(chars, i, DELIMITERS)
                    if j >= 0:
                        delim_weights.append(attn[i, j])
                if delim_weights:
                    delimiter_recency[li, hi] += np.mean(delim_weights)

                # ── 4. Capitalization Tracker ────────────────────────────────
                # For each query i, how much attention goes to the most-recent sentence-end?
                cap_weights = []
                for i in range(N):
                    j = _most_recent_before(chars, i, SENTENCE_ENDS)
                    if j >= 0:
                        cap_weights.append(attn[i, j])
                if cap_weights:
                    cap_tracker[li, hi] += np.mean(cap_weights)

                # ── 5. Vowel / Consonant Tracker ─────────────────────────────
                cons_rows  = [i for i, c in enumerate(chars) if c in CONSONANTS]
                vowel_cols = [j for j, c in enumerate(chars) if c in VOWELS]
                vowel_rows = [i for i, c in enumerate(chars) if c in VOWELS]
                cons_cols  = [j for j, c in enumerate(chars) if c in CONSONANTS]
                if cons_rows and vowel_cols:
                    cons_to_vowel[li, hi] += attn[np.ix_(cons_rows, vowel_cols)].mean()
                if vowel_rows and cons_cols:
                    vowel_to_cons[li, hi] += attn[np.ix_(vowel_rows, cons_cols)].mean()

                # ── 6. Induction Head ─────────────────────────────────────────
                # For query i and key k: "matching prefix" means chars[k-1] == chars[i-1].
                # An induction head gives high attention exactly to those positions.
                match_vals, non_match_vals = [], []
                for i in range(1, N):
                    for k in range(1, i):          # causal: k < i only
                        if chars[k - 1] == chars[i - 1]:
                            match_vals.append(attn[i, k])
                        else:
                            non_match_vals.append(attn[i, k])
                if match_vals:
                    induction_match[li, hi] += np.mean(match_vals)
                if non_match_vals:
                    induction_base[li, hi] += np.mean(non_match_vals)

                # ── Relative position profile ─────────────────────────────────
                for offset in range(min(N, 20)):
                    vals = [attn[i, i - offset] for i in range(offset, N)]
                    rel_pos_mass[offset][li, hi] += np.mean(vals)

    # Normalise
    for arr in [prev_token, first_token, delimiter_recency, cap_tracker,
                cons_to_vowel, vowel_to_cons, induction_match, induction_base]:
        arr /= n_samples
    for offset in rel_pos_mass:
        rel_pos_mass[offset] /= n_samples

    # Induction score = ratio of match attention vs. non-match attention
    induction_score = np.divide(
        induction_match, induction_base,
        out=np.zeros_like(induction_match),
        where=induction_base > 1e-9,
    )

    return {
        "prev_token":        prev_token,
        "first_token":       first_token,
        "delimiter_recency": delimiter_recency,
        "cap_tracker":       cap_tracker,
        "cons_to_vowel":     cons_to_vowel,
        "vowel_to_cons":     vowel_to_cons,
        "induction_score":   induction_score,
        "rel_pos":           rel_pos_mass,
    }


def build_hebrew_stats(model, tokenizer, device, sample_texts):
    """
    Hebrew-specific: prefix-attention and suffix-attention scores.
    """
    prefix_score = np.zeros((N_LAYERS, N_HEADS))
    suffix_score = np.zeros((N_LAYERS, N_HEADS))
    n_samples = 0

    for text in sample_texts:
        chars, all_attentions = get_attention_for_text(model, tokenizer, text, device)
        N = len(chars)
        n_samples += 1

        # prefix positions: single-char prefixes (followed by a non-space)
        prefix_cols = [j for j in range(N - 1)
                       if chars[j] in HEB_PREFIXES and (j == 0 or chars[j-1] == " ")]
        # suffix positions: two-char ים or ות
        suffix_cols = [j for j in range(N - 1)
                       if (chars[j], chars[j+1]) in (HEB_SUFFIX_1, HEB_SUFFIX_2)]

        for li in range(N_LAYERS):
            for hi in range(N_HEADS):
                attn = all_attentions[li][hi]
                if prefix_cols:
                    prefix_score[li, hi] += attn[:, prefix_cols].sum(axis=1).mean()
                if suffix_cols:
                    suffix_score[li, hi] += attn[:, suffix_cols].sum(axis=1).mean()

    if n_samples:
        prefix_score /= n_samples
        suffix_score /= n_samples

    return {"heb_prefix": prefix_score, "heb_suffix": suffix_score}


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _char_ticks(ax, chars, max_ticks=24):
    N = len(chars)
    step = max(1, N // max_ticks)
    ticks = list(range(0, N, step))
    labels = [chars[i] for i in ticks]
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=5, rotation=90)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=5)


def plot_attention_heatmaps(chars, all_attentions, title_prefix="", save_prefix="attn"):
    """All (layer, head) heatmaps in one figure."""
    n_layers = len(all_attentions)
    n_heads  = len(all_attentions[0])
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(3 * n_heads, 3 * n_layers))
    snippet = "".join(chars[:40]) + ("…" if len(chars) > 40 else "")
    fig.suptitle(f"{title_prefix}  \"{snippet}\"", fontsize=8, y=1.01)

    for li in range(n_layers):
        for hi in range(n_heads):
            ax = axes[li][hi]
            ax.imshow(all_attentions[li][hi], aspect="auto", cmap="Blues", vmin=0, vmax=1)
            ax.set_title(f"L{li}H{hi}", fontsize=7)
            _char_ticks(ax, chars)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{save_prefix}_heatmap.png")
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {path}")


def plot_single_head(chars, attn, layer, head, title, filename):
    """Detailed heatmap for one (layer, head)."""
    N = len(chars)
    size = max(6, N * 0.22)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(attn, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"{title}  (L{layer} H{head})", fontsize=10)
    ax.set_xlabel("Key (attended to)"); ax.set_ylabel("Query (attending from)")
    _char_ticks(ax, chars)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {path}")


def plot_induction_overlay(chars, attn, layer, head, filename):
    """
    Heatmap for an induction head with the 'matching-prefix' positions highlighted.
    Green dots mark (query i, key k) where chars[k-1] == chars[i-1].
    """
    N = len(chars)
    size = max(6, N * 0.22)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    ax.imshow(attn, aspect="auto", cmap="Blues", vmin=0, vmax=1)

    # Overlay matching-prefix positions
    match_i, match_k = [], []
    for i in range(1, N):
        for k in range(1, i):
            if chars[k - 1] == chars[i - 1]:
                match_i.append(i); match_k.append(k)
    if match_i:
        ax.scatter(match_k, match_i, s=8, c="lime", alpha=0.5, linewidths=0, label="matching prefix")
        ax.legend(fontsize=7, loc="upper left")

    ax.set_title(f"Induction Head  (L{layer} H{head})\n"
                 f"Green = positions where chars[k-1] == chars[i-1]", fontsize=9)
    ax.set_xlabel("Key"); ax.set_ylabel("Query")
    _char_ticks(ax, chars)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {path}")


def plot_mean_attention_heatmaps(model, tokenizer, device, sample_texts, display_len=30):
    """
    Mean attention heatmap: run all sample texts through the model, crop each
    attention matrix to display_len×display_len, and average (counting only
    positions that actually exist in each sample). This preserves absolute-
    position patterns (e.g. first-token anchor) that the toeplitz approach loses.
    """
    acc   = np.zeros((N_LAYERS, N_HEADS, display_len, display_len))
    count = np.zeros((display_len, display_len))

    for text in sample_texts:
        chars, all_attn = get_attention_for_text(model, tokenizer, text, device)
        N = min(len(chars), display_len)
        count[:N, :N] += 1
        for li in range(N_LAYERS):
            for hi in range(N_HEADS):
                acc[li, hi, :N, :N] += all_attn[li][hi][:N, :N]

    # Normalise by number of samples that contributed to each position
    denom = np.where(count > 0, count, 1)
    mean_attn = acc / denom[np.newaxis, np.newaxis, :, :]

    fig, axes = plt.subplots(N_LAYERS, N_HEADS,
                             figsize=(3 * N_HEADS, 3 * N_LAYERS))
    fig.suptitle("Mean attention heatmap (averaged over all samples)\n"
                 "rows=query, cols=key", fontsize=10)

    for li in range(N_LAYERS):
        for hi in range(N_HEADS):
            M  = mean_attn[li, hi]
            ax = axes[li][hi]
            ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=M.max() or 1)
            ax.set_title(f"L{li}H{hi}", fontsize=7)
            ax.tick_params(labelbottom=False, labelleft=False,
                           bottom=False, left=False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "mean_attention_heatmap.png")
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {path}")


def plot_rel_pos_profiles(rel_pos_mass):
    """Bar-chart per (layer, head): avg attention vs. relative offset."""
    offsets = sorted(rel_pos_mass.keys())
    fig, axes = plt.subplots(N_LAYERS, N_HEADS,
                             figsize=(3 * N_HEADS, 2.5 * N_LAYERS), sharex=True)
    fig.suptitle("Avg attention weight by relative position offset\n"
                 "(0 = self, 1 = previous token, …)", fontsize=10)
    for li in range(N_LAYERS):
        for hi in range(N_HEADS):
            ax = axes[li][hi]
            vals = [rel_pos_mass[k][li, hi] for k in offsets]
            ax.bar(offsets, vals, color="steelblue", width=0.8)
            ax.set_title(f"L{li}H{hi}", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "rel_pos_profiles.png")
    plt.savefig(path, dpi=100, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {path}")


def plot_stat_summary(stats, stat_names, filename="stat_summary.png"):
    """One heatmap panel per stat showing (layer × head) grid."""
    n = len(stat_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, stat_names):
        grid = stats[name]
        im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", vmin=0)
        ax.set_title(name.replace("_", " "), fontsize=9)
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.set_xticks(range(grid.shape[1])); ax.set_yticks(range(grid.shape[0]))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for li in range(grid.shape[0]):
            for hi in range(grid.shape[1]):
                ax.text(hi, li, f"{grid[li,hi]:.2f}", ha="center", va="center",
                        fontsize=6, color="black")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Console summary
# ══════════════════════════════════════════════════════════════════════════════

SCALAR_STATS = [
    ("prev_token",        "prev"),
    ("first_token",       "first"),
    ("delimiter_recency", "delim"),
    ("cap_tracker",       "cap"),
    ("cons_to_vowel",     "C→V"),
    ("vowel_to_cons",     "V→C"),
    ("induction_score",   "induc"),
]


def print_pattern_table(stats):
    print("\n" + "=" * 80)
    print("DOMINANT ATTENTION PATTERN PER (LAYER, HEAD)")
    print("=" * 80)
    col_w = 8

    # Header row
    print(f"{'':8}", end="")
    for _, short in SCALAR_STATS:
        print(f" {short:>{col_w}}", end="")
    print()
    print("-" * (8 + (col_w + 1) * len(SCALAR_STATS)))

    for li in range(N_LAYERS):
        for hi in range(N_HEADS):
            scores = {name: stats[name][li, hi] for name, _ in SCALAR_STATS}
            best   = max(scores, key=scores.get)
            tag    = next(s for n, s in SCALAR_STATS if n == best)
            label  = f"L{li}H{hi}({tag})"
            print(f"{label:<8}", end="")
            for name, _ in SCALAR_STATS:
                v = scores[name]
                marker = " *" if name == best else "  "
                print(f"{marker}{v:>{col_w - 2}.3f}", end="")
            print()

    print("=" * 80)


def find_best_head(stats, stat_name):
    grid = stats[stat_name]
    idx  = np.unravel_index(np.argmax(grid), grid.shape)
    return int(idx[0]), int(idx[1])

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load English model ────────────────────────────────────────────────────
    model, tokenizer, device = load_model_and_tokenizer(DATA_PATH)

    # ── 1. Aggregate stats ────────────────────────────────────────────────────
    print("\n[1] Computing aggregate statistics (English)...")
    stats = build_aggregate_stats(model, tokenizer, device, ENGLISH_SAMPLES)

    # ── 2. Mean attention heatmap (averaged over all samples) ────────────────
    print("\n[2] Mean attention heatmap...")
    plot_mean_attention_heatmaps(model, tokenizer, device, ENGLISH_SAMPLES)

    # ── 3. Relative-position profiles ────────────────────────────────────────
    print("\n[3] Relative-position profiles...")
    plot_rel_pos_profiles(stats["rel_pos"])

    # ── 4. Scalar stat summary heatmap ───────────────────────────────────────
    print("\n[4] Scalar stat summary...")
    scalar_names = [n for n, _ in SCALAR_STATS]
    plot_stat_summary(stats, scalar_names)

    # ── 5. Console pattern table ──────────────────────────────────────────────
    print("\n[5] Pattern table:")
    print_pattern_table(stats)

    # ── 6. Spotlight: best head per pattern ───────────────────────────────────
    print("\n[6] Spotlight heatmaps for best head per pattern...")
    spotlight_text = ENGLISH_SAMPLES[0]
    chars, all_attn = get_attention_for_text(model, tokenizer, spotlight_text, device)

    spotlight_configs = [
        ("prev_token",        "Best previous-token head",      "spot_prev_token.png"),
        ("first_token",       "Best first-token head",         "spot_first_token.png"),
        ("delimiter_recency", "Best Space-Seeking head",       "spot_delimiter.png"),
        ("cap_tracker",       "Best Capitalization Tracker",   "spot_cap_tracker.png"),
        ("cons_to_vowel",     "Best Consonant→Vowel head",     "spot_cons_to_vowel.png"),
    ]
    for stat_name, title, filename in spotlight_configs:
        li, hi = find_best_head(stats, stat_name)
        score  = stats[stat_name][li, hi]
        print(f"  {title}: L{li}H{hi}  (score={score:.3f})")
        plot_single_head(chars, all_attn[li][hi], li, hi, title, filename)

    # ── 7. Induction Head ────────────────────────────────────────────────────
    print("\n[7] Induction Head analysis...")
    # Use a text with long repeated words for a clear induction signal
    induction_text = ENGLISH_SAMPLES[12]   # "Shakespeare wrote … Shakespeare …"
    print(f"  Text: \"{induction_text}\"")
    chars_ind, all_attn_ind = get_attention_for_text(model, tokenizer, induction_text, device)
    li_ind, hi_ind = find_best_head(stats, "induction_score")
    score_ind = stats["induction_score"][li_ind, hi_ind]
    print(f"  Best Induction Head: L{li_ind}H{hi_ind}  (match/base ratio={score_ind:.3f})")
    plot_induction_overlay(chars_ind, all_attn_ind[li_ind][hi_ind],
                           li_ind, hi_ind, "spot_induction.png")


    # ── 8. Hebrew analysis (requires a Hebrew-trained checkpoint) ────────────
    heb_data_exists = (
        os.path.isdir(HEB_PATH)
        and any(f.endswith(".txt") for f in os.listdir(HEB_PATH))
    ) if os.path.isdir(HEB_PATH) else False

    # ── Hebrew analysis (uncomment once a Hebrew checkpoint is trained) ───────
    # if heb_data_exists and HEB_CHECKPOINT is not None:
    #     print("\n[8] Hebrew Prefix/Suffix Head analysis...")
    #     heb_model, heb_tokenizer, _ = load_model_and_tokenizer(HEB_PATH, HEB_CHECKPOINT)
    #     heb_stats = build_hebrew_stats(heb_model, heb_tokenizer, device, HEBREW_SAMPLES)
    #     plot_stat_summary(heb_stats, ["heb_prefix", "heb_suffix"],
    #                       filename="heb_stat_summary.png")
    #     li_p, hi_p = find_best_head(heb_stats, "heb_prefix")
    #     li_s, hi_s = find_best_head(heb_stats, "heb_suffix")
    #     print(f"  Best Hebrew-prefix head:  L{li_p}H{hi_p}  ({heb_stats['heb_prefix'][li_p,hi_p]:.3f})")
    #     print(f"  Best Hebrew-suffix head:  L{li_s}H{hi_s}  ({heb_stats['heb_suffix'][li_s,hi_s]:.3f})")
    #     chars_heb, attn_heb = get_attention_for_text(
    #         heb_model, heb_tokenizer, HEBREW_SAMPLES[0], device)
    #     plot_single_head(chars_heb, attn_heb[li_p][hi_p], li_p, hi_p,
    #                      "Hebrew Prefix Head", "spot_heb_prefix.png")
    #     plot_single_head(chars_heb, attn_heb[li_s][hi_s], li_s, hi_s,
    #                      "Hebrew Suffix Head", "spot_heb_suffix.png")

    print(f"\nAll plots saved to: {os.path.abspath(PLOTS_DIR)}/")
    print("Done.")


if __name__ == "__main__":
    main()
