"""Set Transformer implementation shared by Notebook 07 and its inference package.

The notebook imports this file after verifying the prepared shared dependencies.
Keep the exported folder together when calling load_inference_model independently.
"""

import functools
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_layer
from torch.utils.data import DataLoader, Dataset

MODULE_ROOT = Path(__file__).resolve().parent
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
if (MODULE_ROOT / "otc_shared.py").is_file():
    sys.path[:0] = [str(MODULE_ROOT / "frozen"), str(MODULE_ROOT)]

import otc_shared as shared_module
from otc_shared import STATUS, SUPPORTED, check_payload, sha_file
from hrp.data_cleaning import normalize_name

if (MODULE_ROOT / "otc_shared.py").is_file():
    if sha_file(shared_module.__file__) != sha_file(MODULE_ROOT / "otc_shared.py"):
        raise RuntimeError("A different shared module is already loaded")

QUERY_NAMES = ["name", "dosage_form", "route", "native_strengths", "product_ndc"]
candidate_cache = {}
DEVICE = torch.device("cpu")
USE_AMP = False
SEED = 42
CHUNK_SIZE = 160
SET_SLOT_BUDGET = 512
TRAIN_CONFIG = {"batch_size": 32}
model = None


def configure_runtime(encoding, device, use_amp=False, seed=42, set_slot_budget=512, batch_size=32):
    """Apply notebook settings without changing model parameters or random seeds."""
    global DEVICE, USE_AMP, SEED, CHUNK_SIZE, SET_SLOT_BUDGET, TRAIN_CONFIG
    DEVICE = torch.device(device)
    USE_AMP = DEVICE.type == "cuda" and use_amp
    SEED = seed
    CHUNK_SIZE = int(encoding["text_chunk_bytes"])
    SET_SLOT_BUDGET = set_slot_budget
    TRAIN_CONFIG = {"batch_size": batch_size}
    text_chunks.cache_clear()
    candidate_cache.clear()

def query_texts(query):
    return [
        query.get("name", ""), query.get("dosage_form", ""),
        " ".join(query.get("route", [])), " ".join(query.get("native_strengths", [])),
        query.get("product_ndc", "")
    ]

def candidate_texts(candidate):
    key = candidate["product_ndc"]
    if key not in candidate_cache:
        native, spl, rx = candidate["fda_active_ingredients"], candidate["spl"], candidate["rxnorm"]
        pairs = rx["ingredient_pairs"]
        texts = [
            " | ".join(candidate[k] for k in ("display_name", "brand_name", "generic_name")),
            " | ".join([i["name"] for i in native] + [i["name"] for i in pairs]),
            " | ".join(i.get("strength", "") for i in native),
            candidate["dosage_form"] + " | " + " ".join(candidate["route"]),
            " | ".join([candidate["product_ndc"], spl["label_id"], spl["set_id"],
                         str(spl["version"]), str(spl["effective_time"]), rx["product_rxcui"]]
                        + [" ".join(candidate["provenance"].get("product_uniis", []))]
                        + [p["rxcui"] + " " + p["term_type"] + " " + " ".join(p["uniis"]) for p in pairs]),
            " | ".join(spl["active_ingredient_text"] + spl["structured_substance_names"]
                        + [rx["product_name"]])
        ]
        source = candidate["provenance"]
        flags = [
            not native, not spl["active_ingredient_text"], not candidate["dosage_form"],
            not candidate["route"], not rx["product_rxcui"],
            source["original_versions_confirmed"], source["raw_ndc_name_matches_catalog"],
            source.get("raw_form_route_confirmed", False), rx["cached_pairs_confirmed"]
        ]
        candidate_cache[key] = (texts, [float(v) for v in flags])
    return candidate_cache[key]

@functools.lru_cache(maxsize=32768)
def text_chunks(text):
    values = np.frombuffer((normalize_name(text) or " ").encode("utf-8"), dtype=np.uint8).astype(np.uint16) + 1
    step = CHUNK_SIZE - 6
    starts = range(0, max(1, len(values) - 6), step)
    out = np.zeros((len(starts), CHUNK_SIZE), dtype=np.uint16)
    for row, start in enumerate(starts):
        block = values[start:start + CHUNK_SIZE]
        out[row, :len(block)] = block
    return out

def encode_fields(texts):
    chunks, owners = [], []
    for index, text in enumerate(texts):
        block = text_chunks(text)
        chunks.append(block)
        owners.extend([index] * len(block))
    return {
        "tokens": torch.from_numpy(np.concatenate(chunks).astype(np.int64)),
        "owner": torch.tensor(owners, dtype=torch.long),
        "field_count": len(texts)
    }

class MedicineSets(Dataset):
    def __init__(self, rows, labelled=True):
        self.items = []
        self.labels = []
        for row in rows:
            check_payload(row["payload"])
            for index, item in enumerate(row["payload"]["products"]):
                self.items.append(item)
                if labelled:
                    target = row["target"]["products"][index]
                    self.labels.append({
                        "status": STATUS.index(target["status"]),
                        "sufficiency": float(target["status"] in SUPPORTED),
                        "form": float(target["formulation"] is not None),
                        "candidate": row["candidate_labels"][index]
                    })
                else:
                    self.labels.append(None)
    def __len__(self):
        return len(self.items)
    def __getitem__(self, index):
        # The model receives only the source dictionary
        return self.items[index], self.labels[index], index

def collate(batch):
    batch_size = len(batch)
    width = max(max(1, len(item["candidates"])) for item, _, _ in batch)
    qtexts, ctexts, flags, positions, qmissing = [], [], [], [], []
    valid = torch.zeros(batch_size, width, dtype=torch.bool)
    real = torch.zeros_like(valid)
    y_candidate = torch.zeros(batch_size, width)
    y_status, y_sufficiency, y_form, indices = [], [], [], []
    counts = []
    labelled = batch[0][1] is not None
    for row, (item, label, index) in enumerate(batch):
        fields = query_texts(item["request"])
        qtexts.extend(fields)
        qmissing.append([float(not item["request"].get(k)) for k in QUERY_NAMES])
        n = len(item["candidates"])
        counts.append(n)
        indices.append(index)
        valid[row, :max(1, n)] = True
        real[row, :n] = True
        for col, candidate in enumerate(item["candidates"]):
            fields, missing = candidate_texts(candidate)
            ctexts.extend(fields)
            flags.append(missing)
            positions.append(row * width + col)
        if n == 0:
            # One learned sentinel prevents an entirely masked attention row
            ctexts.extend([""] * 6)
            flags.append([1.0] * 5 + [0.0] * 4)
            positions.append(row * width)
        if labelled:
            if label is None:
                raise ValueError("Mixed labelled and inference batch")
            y_candidate[row, :n] = torch.tensor(label["candidate"], dtype=torch.float32)
            y_status.append(label["status"])
            y_sufficiency.append(label["sufficiency"])
            y_form.append(label["form"])
    features = {
        "query": encode_fields(qtexts), "candidate": encode_fields(ctexts),
        "candidate_flags": torch.tensor(flags, dtype=torch.float32),
        "query_missing": torch.tensor(qmissing, dtype=torch.float32),
        "positions": torch.tensor(positions, dtype=torch.long),
        "valid": valid, "real": real
    }
    target = {
        "status": torch.tensor(y_status, dtype=torch.long),
        "sufficiency": torch.tensor(y_sufficiency, dtype=torch.float32),
        "form": torch.tensor(y_form, dtype=torch.float32), "candidate": y_candidate
    } if labelled else None
    return features, target, indices, counts

def move(value):
    if isinstance(value, torch.Tensor):
        return value.to(DEVICE, non_blocking=DEVICE.type == "cuda")
    if isinstance(value, dict):
        return {k: move(v) for k, v in value.items()}
    return value

def pack_batches(indices, dataset, batch_size):
    batches, current, width = [], [], 0
    for index in indices:
        next_width = max(width, max(1, len(dataset.items[index]["candidates"])))
        if current and (len(current) == batch_size or next_width * (len(current) + 1) > SET_SLOT_BUDGET):
            batches.append(current)
            current, width = [], 0
        current.append(index)
        width = max(width, max(1, len(dataset.items[index]["candidates"])))
    if current:
        batches.append(current)
    return batches

def train_batches(dataset, batch_size, epoch):
    rng = np.random.default_rng(SEED + epoch)
    indices = rng.permutation(len(dataset)).tolist()
    batches = []
    for start in range(0, len(indices), batch_size * 20):
        window = sorted(indices[start:start + batch_size * 20],
                        key=lambda index: len(dataset.items[index]["candidates"]))
        batches.extend(pack_batches(window, dataset, batch_size))
    rng.shuffle(batches)
    return batches

def loader(dataset, batch_size=32, epoch=None):
    if epoch is None:
        indices = sorted(range(len(dataset)), key=lambda index: len(dataset.items[index]["candidates"]))
        batches = pack_batches(indices, dataset, batch_size)
    else:
        batches = train_batches(dataset, batch_size, epoch)
    return DataLoader(dataset, batch_sampler=batches, collate_fn=collate,
                      num_workers=0, pin_memory=DEVICE.type == "cuda")

class CharacterCNN(nn.Module):
    def __init__(self, hidden=128, embedding=32):
        super().__init__()
        self.embedding = nn.Embedding(257, embedding, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embedding, 48, k, padding=k // 2) for k in (3, 5, 7)])
        self.project = nn.Sequential(nn.Linear(144, hidden), nn.GELU(), nn.LayerNorm(hidden))


    def encode_chunk(self, tokens):
        valid = tokens.ne(0).unsqueeze(1)
        values = self.embedding(tokens).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            hidden = F.gelu(conv(values))
            pooled.append(hidden.masked_fill(~valid, torch.finfo(hidden.dtype).min).amax(dim=2))
        return self.project(torch.cat(pooled, dim=1))

    def forward(self, fields):
        tokens, owner = fields["tokens"], fields["owner"]
        encoded = []
        for start in range(0, tokens.shape[0], 512):
            chunk = tokens[start:start + 512]
            # Recompute large convolution batches during backward to save activation memory
            if self.training and torch.is_grad_enabled() and tokens.shape[0] > 512:
                encoded.append(checkpoint_layer(self.encode_chunk, chunk, use_reentrant=False))
            else:
                encoded.append(self.encode_chunk(chunk))
        chunks = torch.cat(encoded, dim=0)
        fields_out = chunks.new_zeros(fields["field_count"], chunks.shape[1])
        fields_out.scatter_reduce_(
            0, owner[:, None].expand(-1, chunks.shape[1]), chunks,
            reduce="amax", include_self=False
        )
        return fields_out

class AttentionBlock(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, key_mask):
        value, _ = self.attention(queries, keys, keys, key_padding_mask=~key_mask, need_weights=False)
        x = self.norm1(queries + self.dropout(value))
        return self.norm2(x + self.dropout(self.ff(x)))

class HybridSetTransformer(nn.Module):
    def __init__(self, hidden=128, heads=4, layers=2, dropout=0.1, active_status=None):
        super().__init__()
        self.query_cnn, self.candidate_cnn = CharacterCNN(hidden), CharacterCNN(hidden)
        self.query_field = nn.Parameter(torch.randn(5, hidden) * 0.02)
        self.query_project = nn.Sequential(nn.Linear(hidden + 5, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.candidate_project = nn.Sequential(
            nn.Linear(6 * hidden + 9, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout)
        )
        self.cross = AttentionBlock(hidden, heads, dropout)
        self.set_layers = nn.ModuleList([AttentionBlock(hidden, heads, dropout) for _ in range(layers)])
        self.pool_seed = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.pool = AttentionBlock(hidden, heads, dropout)
        self.candidate_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        def head(size):
            return nn.Sequential(nn.Linear(hidden * 2 + 1, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, size))
        self.status_head, self.sufficiency_head, self.form_head = head(6), head(1), head(1)
        active = active_status if active_status is not None else [True] * 6
        self.register_buffer("active_status", torch.tensor(active, dtype=torch.bool))

    def forward(self, features):
        valid = features["valid"]
        batch_size, width = valid.shape
        query_tokens = self.query_cnn(features["query"]).reshape(batch_size, 5, -1) + self.query_field
        query = self.query_project(torch.cat((query_tokens.mean(dim=1), features["query_missing"]), dim=1))
        source_tokens = self.candidate_cnn(features["candidate"]).reshape(-1, 6 * query.shape[1])
        source = self.candidate_project(torch.cat((source_tokens, features["candidate_flags"]), dim=1))
        flat = source.new_zeros(batch_size * width, source.shape[1])
        candidates = flat.index_copy(0, features["positions"], source).reshape(batch_size, width, -1)
        query_mask = torch.ones(batch_size, 5, dtype=torch.bool, device=valid.device)
        candidates = self.cross(candidates, query_tokens, query_mask)
        for block in self.set_layers:
            candidates = block(candidates, candidates, valid)
            candidates = candidates.masked_fill(~valid.unsqueeze(-1), 0)
        seed = self.pool_seed.expand(batch_size, -1, -1) + query.unsqueeze(1)
        pooled = self.pool(seed, candidates, valid).squeeze(1)
        # Set size is source information and is unchanged by candidate order
        count = torch.log1p(features["real"].sum(dim=1, keepdim=True).to(pooled.dtype))
        fused = torch.cat((pooled, query, count), dim=1)
        candidate_input = torch.cat((candidates, query[:, None].expand(-1, width, -1)), dim=-1)
        return {
            "candidate": self.candidate_head(candidate_input).squeeze(-1),
            "status": self.status_head(fused).masked_fill(~self.active_status, -1e4),
            "sufficiency": self.sufficiency_head(fused).squeeze(-1),
            "form": self.form_head(fused).squeeze(-1)
        }

@torch.inference_mode()
def predict_dataset(dataset):
    if model is None:
        raise RuntimeError("Load an inference model before requesting predictions")
    model.eval()
    decisions = [None] * len(dataset)
    for features, _, indices, counts in loader(dataset, TRAIN_CONFIG["batch_size"]):
        features = move(features)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            output = model(features)
        candidate = output["candidate"].float().sigmoid().cpu().numpy()
        status = output["status"].float().softmax(-1).cpu().numpy()
        sufficiency = output["sufficiency"].float().sigmoid().cpu().numpy()
        form = output["form"].float().sigmoid().cpu().numpy()
        for row, (index, n) in enumerate(zip(indices, counts)):
            decisions[index] = {
                "candidate": candidate[row, :n].tolist(), "status": status[row].tolist(),
                "sufficiency": float(sufficiency[row]), "form": float(form[row])
            }
    if any(d is None for d in decisions):
        raise RuntimeError("Missing model decisions")
    return decisions

def load_inference_model(root, device=None):
    global model, DEVICE, USE_AMP, CHUNK_SIZE, SET_SLOT_BUDGET, TRAIN_CONFIG, SEED
    root = Path(root)
    model_config = json.loads((root / "model_config.json").read_text(encoding="utf-8"))
    encoding_config = json.loads((root / "encoding.json").read_text(encoding="utf-8"))
    runtime = json.loads((root / "inference_config.json").read_text(encoding="utf-8"))
    DEVICE = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    USE_AMP = DEVICE.type == "cuda" and runtime.get("precision", "fp16") == "fp16"
    CHUNK_SIZE = int(encoding_config["text_chunk_bytes"])
    SET_SLOT_BUDGET, SEED = runtime["set_slot_budget"], runtime["seed"]
    TRAIN_CONFIG = {"batch_size": runtime["batch_size"]}
    text_chunks.cache_clear()
    candidate_cache.clear()
    model = HybridSetTransformer(**model_config).to(DEVICE)
    model.load_state_dict(torch.load(root / "model.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    return model
