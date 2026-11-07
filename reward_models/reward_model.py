import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from pathlib import Path

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# BASE_MODEL = "distilbert/distilbert-base-cased"

EMBED_DIM = None  # auto-detected from config if None


def mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def meanmax_pool(hidden, attention_mask):
    mask_f = attention_mask.unsqueeze(-1).float()
    mean = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
    mx = hidden.masked_fill(mask_f == 0, float("-inf")).max(1).values
    return torch.cat([mean, mx], dim=-1)


def load_reward_model(path):
    """
    Load a trained MiniLM reward model.

    Usage:
        model, tokenizer = load_reward_model("paperbd/neuraltxt-reward-22M")
        score = model.score("reference text", "candidate text")
    """
    raw_source = str(path)
    local_path = Path(raw_source).expanduser()
    is_local = local_path.exists()
    source = str(local_path) if is_local else raw_source

    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(source)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Encoder: load the local checkpoint / HF repo, falling back to the base
    # encoder only when the saved encoder weights are absent.
    try:
        if is_local and not (local_path / "model.safetensors").exists():
            encoder = AutoModel.from_pretrained(BASE_MODEL)
        else:
            encoder = AutoModel.from_pretrained(source)
    except Exception:
        encoder = AutoModel.from_pretrained(BASE_MODEL)

    # Head (~3KB)
    # Auto-detect embedding dim and pooling from the saved head's input width:
    # heads trained with mean+max concat pooling are 2x hidden_size wide.
    dim = EMBED_DIM or encoder.config.hidden_size
    head_state = None
    candidates = ["head_weights.pt", "head_weights.bin"]
    if is_local:
        for fname in candidates:
            hp = local_path / fname
            if hp.exists():
                head_state = torch.load(str(hp), weights_only=True, map_location="cpu")
                break
    else:
        for fname in candidates:
            try:
                hp = hf_hub_download(repo_id=source, filename=fname)
            except Exception:
                continue
            head_state = torch.load(hp, weights_only=True, map_location="cpu")
            break
    if head_state is None:
        raise FileNotFoundError(
            f"No head_weights.pt/.bin found for reward model {source!r}. "
            "This loader expects a trained reward-model directory or HF repo."
        )
    pool_fn = mean_pool
    if head_state is not None:
        in_dim = head_state["1.weight"].shape[1]
        if in_dim == 2 * dim:
            pool_fn = meanmax_pool
        dim = in_dim
    head = nn.Sequential(nn.Dropout(0.1), nn.Linear(dim, 1))
    if head_state is not None:
        head.load_state_dict(head_state)
    head.eval()

    class RewardScorer:
        def __init__(self):
            self.encoder = encoder
            self.head = head

        def score(self, reference, response):
            text = f"{reference} [SEP] {response}"
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )
                pooled = pool_fn(outputs.last_hidden_state, enc["attention_mask"])
                return self.head(pooled).item()

        def score_batch(self, references, responses, batch_size=128):
            scores = []
            for start in range(0, len(references), batch_size):
                batch_refs = references[start:start + batch_size]
                batch_resps = responses[start:start + batch_size]
                texts = [f"{r} [SEP] {c}" for r, c in zip(batch_refs, batch_resps)]
                enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.encoder(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                    )
                    pooled = pool_fn(outputs.last_hidden_state, enc["attention_mask"])
                    scores.extend(self.head(pooled).squeeze(-1).tolist())
            return scores

    return RewardScorer(), tokenizer
