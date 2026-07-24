# Embedding Surgery 2: Domain-Specialized Nomic Embed Model

This repository contains the code and artifacts for **embedding surgery** – extending the vocabulary and embedding matrix of the Nomic Embed text model (`nomic-ai/nomic-embed-text-v1-unsupervised`) with 15 new domain-specific tokens. The original model is a state‑of‑the‑art BERT‑like encoder trained with Masked Language Modeling (MLM) and optimized for long‑sequence retrieval. By surgically adding tokens like `kubernetes`, `postgres`, and `encapsulation` as single units, we preserve the pretrained weights while enabling efficient, lossless representation of technical jargon.

---

## 📖 Overview

The goal of this project is to **add 15 new tokens** to the vocabulary of a 768‑dimension BERT‑style model without retraining from scratch. Instead of relying on subword splits (e.g., `kubernetes` → `ku`, `##ber`, `##net`, `##es`), we inject the complete words as indivisible tokens and initialize their embeddings as the **average of their original subword embeddings**. This technique – often called *embedding surgery* – retains the model’s semantic knowledge while making domain terms directly accessible.

The final model can be loaded with a standard `AutoModel` interface and is ready for fine‑tuning or immediate inference on technical text.

---

## 🧠 Background & Paper Insights

The base model is a variant of BERT described in the paper [**Nomic Embed: Training a Reproducible Long Context Text Embedder**](https://arxiv.org/pdf/2402.01613). Key architectural features include:

- **Rotary Positional Embeddings** (RoPE) instead of absolute positions.
- **SwiGLU activation** (replacing GeLU).
- **Flash Attention** for efficient long‑context processing.
- **Zero dropout** during pretraining.
- **Vocabulary size padded to a multiple of 64** (exactly 30,528 rows, while the tokenizer has 30,522 tokens).

This padding creates a gap of 6 unused rows, but the model was actually pretrained with a logical vocabulary of 30,522. When we add 15 tokens, we must increase the embedding matrix to 30,592 rows (again a multiple of 64) and place the new tokens starting at index 30,522, leaving the original 30,522 rows untouched.

---

## 🔧 Methodology

The surgery is performed in several careful steps:

1. **Inspect architecture** – confirm tokenizer vocabulary size (30,522) and embedding matrix shape (30,528 × 768).
2. **Calculate initialisation vectors** – for each new token, tokenize it with the original tokenizer to get its subword IDs, then average their embeddings.
3. **Modify tokenizer** – rewrite `vocab.txt` to append the 15 new tokens, delete stale `tokenizer.json`, and reload with `never_split` to prevent punctuation splitting (e.g., `c++` was replaced by `encapsulation` to avoid issues).
4. **Expand embedding matrix** – instantiate a new `nn.Embedding` of shape (30,592, 768), copy the first 30,528 rows from the original, and inject the averaged vectors at indices 30,522..30,536.
5. **Update model config** – set `config.vocab_size` to the logical size (30,537) and fix the missing `pad_token_id` (set to 0).
6. **Verification** – assert that the original 30,522 embeddings are unchanged, each new token is a single token, and its embedding equals the calculated average.
7. **Save** – persist the model and tokenizer locally.

---

## 📦 Custom Tokens Added

| Token            | Type        |
|------------------|-------------|
| `cybersecurity`  | Domain term  |
| `kubernetes`     | Technology   |
| `microservices`  | Architecture  |
| `hashmap`        | Data structure|
| `backpropagation`| ML term      |
| `asynchronous`   | Programming  |
| `postgres`       | Database     |
| `minio`          | Storage      |
| `nextjs`         | Framework    |
| `javascript`     | Language     |
| `docker`         | Container    |
| `nginx`          | Web server   |
| `ssh`            | Protocol     |
| `ddr4`           | Hardware     |
| `encapsulation`  | OOP concept  |

*(`c++` was replaced by `encapsulation` to avoid tokenizer splitting on `+`.)*

---

## 🔍 Verification

The notebook runs a comprehensive test suite:

- **Old embeddings unchanged** – first 30,522 rows are bit‑for‑bit identical.
- **New tokens single‑piece** – each appears as a single token after tokenization.
- **Initialisation correct** – the embedding of each new token equals the average of its original subword embeddings.
- **Forward pass stable** – no NaNs or infinities; output shape is `(batch, seq_len, 768)`.

Example output from the notebook:

```
Tokens mapping check:
[CLS]           -> 101
routing         -> 16972
microservices   -> 30524
...
[SEP]           -> 102

Success: Forward pass produced stable vectors. Output shape: torch.Size([1, 17, 768])
```

---

## 🚀 Usage

### Loading the custom model

```python
from transformers import AutoModel, BertTokenizer

model = AutoModel.from_pretrained("abood1521/cs-nomic-embed-text-v1-unsupervised", trust_remote_code=True)
tokenizer = BertTokenizer.from_pretrained("abood1521/cs-nomic-embed-text-v1-unsupervised")

text = "Routing microservices with kubernetes and postgres on docker"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)  # torch.Size([1, 13, 768])
```

**Important:** Always set `trust_remote_code=True` because the model uses a custom architecture (`NomicBertModel`) defined in the original repository.

### Fine‑tuning

The model is fully compatible with the `transformers` `Trainer` API. You can fine‑tune it on your own MLM or contrastive learning objectives – the new tokens will be trained just like any other.

---

## 📁 Repository Structure

```
.
├── main.ipynb               # Complete surgery notebook (Phase 1)
├── README.md                # This file
├── ROADMAP.md               # Planning and paper analysis notes
└── nomic-custom-stack/      # Saved model & tokenizer (not included in repo)
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    ├── vocab.txt
    └── ...
```

---

## 🧪 Reproducibility

To reproduce the surgery from scratch:

1. Clone this repository.
2. Install dependencies: `pip install torch transformers huggingface_hub`.
3. Open `main.ipynb` and run all cells.
4. Set `PUSH = True` in the final cell and provide a valid Hugging Face token to upload your custom model.

All steps are deterministic and the notebook includes thorough assertions.

---

## 📚 References

- Nomic AI. *Nomic Embed: Training a Reproducible Long Context Text Embedder*. [arXiv:2402.01613](https://arxiv.org/pdf/2402.01613).
- Devlin et al. *BERT: Pre‑training of Deep Bidirectional Transformers for Language Understanding*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805).
- Hugging Face Transformers documentation.

---

## 📄 License

The base model is released under the Apache‑2.0 license. This repository contains code and documentation only; the modified model weights are not distributed here. If you upload your own version, please respect the original model's license.

---

*Happy embedding surgery!* 🏥