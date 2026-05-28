# Setup & research notes — Turkish Gemma 4 E4B

## 1. Frisson Labs blog summary

**Article:** [Grafting a Speech Head onto Gemma 4 E4B](https://www.frisson-labs.com/gemma4-e4b-architecture)

### Architecture

| Component | Role |
|-----------|------|
| **Gemma 4 E4B** (~4.5B effective) | Frozen multimodal decoder: text, image, audio **in** → text logits **out** |
| **Audio tower (input)** | Mono 16 kHz → embeddings in shared 2560-d hidden space (ASR / audio understanding, not speech generation) |
| **Tap point** | Learned mix of **last 6 decoder layers**, **before** tied text vocabulary head |
| **Trainable head (~152M params)** | Maps Gemma hidden states → **Mimi codec tokens** (8 codebooks @ 12.5 Hz) |
| **Frozen Mimi decoder** | Codec tokens → 24 kHz waveform (no text passed to Mimi at inference) |

### Training / inference setup (their smoke test)

- **Text path:** Prompt template `Say this naturally as speech:\n{text}`; Gemma processes text; audio head predicts Mimi tokens; Mimi decodes audio.
- **Audio-in path:** WAV contains instruction + sentence; **no text** to Gemma; same head on audio-conditioned hidden states.
- **Loss:** Cross-entropy on Mimi-encoded teacher speech (parallel prediction over codebooks).
- **Data scale:** 128 train / 36 valid / 36 test — tiny, phrase-level.
- **Checkpoints:** ~500 steps (text smoke), ~850 steps (audio continuation).
- **Code/data:** [gemma4-audio](https://github.com/frisson-labs/gemma4-audio) (minimal reproduction set + samples).

### Why quality was poor (by their own account)

1. **Severe overfitting** — memorized short English phrases, not open-ended generation.
2. **No temporal / duration modeling** — fixed parallel codec head; weak prosody and timing.
3. **Tiny dataset** — architecture proof only (128 samples).
4. **English-only smoke prompts** — no Turkish (or broad language) coverage.
5. **Head reads states before text is “spoken” as an answer** — trained to render **given** text, not **model-generated** replies.
6. **Codec-token objective ≠ intelligibility** — ASR word-match on 5/7 clips; phonetically garbled output still possible.
7. **Frozen Gemma** — no adaptation of decoder representations for speech or Turkish.

### Concrete improvements for a **Turkish-focused** better version

| Area | Recommendation |
|------|----------------|
| **Language base** | Continual pretrain or large SFT on Turkish (**LTC-100B SFT slice**, **InstrucTurca**, **tascib/turkish-instruction**) before or jointly with speech-head training so hidden states encode Turkish morphology and discourse. |
| **Speech data** | 100+ hours Turkish studio TTS + spontaneous speech; phrase diversity; speaker and domain balance; avoid English-only teacher WAVs. |
| **Training objective** | Train head on **generated answer** hidden states (autoregressive decode first), not only fixed templates; add duration/streaming losses. |
| **Gemma adaptation** | LoRA on late decoder layers (Turkish SFT) while training head, or stage-2 after Turkish text SFT. |
| **Evaluation** | Turkish **WER/CER** (Whisper-large-v3-tr or native ASR), **UTMOS/similarity**, MOS listening tests, semantic match (embedding) vs reference. |
| **Production fallback** | Gemma Turkish text + low-latency **Turkish neural TTS** (Piper/Coqui/Azure) until codec head meets bar. |
| **Multimodal product** | Keep Gemma’s audio **input** for understanding Turkish voice chat; separate quality bar for **output** speech. |

---

## 2. Hugging Face datasets (Turkish)

Prioritized for Gemma-class **SFT**, **continual pretrain**, **translation**, **QA/summarization eval**, and optional **speech** pairing (external audio).

| Dataset | HF path | Size / type | License (check card) | Suggested use |
|---------|---------|-------------|----------------------|---------------|
| **InstrucTurca** | `turkish-nlp-suite/InstrucTurca` | ~2.58M instruction pairs; diverse (code, math, medical, etc.) | Commercial-friendly (Arctic translation pipeline; Apache-2.0 cited) | **SFT** primary mix; filter for quality |
| **Turkish instruction (merged)** | `tascib/turkish-instruction` | 324k Alpaca-style rows; deduplicated | **CC BY-SA 4.0** | **SFT**; good single-loader baseline |
| **Lumees Turkish Corpus 100B** | `lumees/turkish-corpus-100b` | ~103B tok pretrain + ~2.2B tok SFT (ChatML); pilot 10B subset | See dataset card / citation | **Continual pretrain** + large **SFT**; use pilot for dev |
| **Turkish-LLM-v10-Training** | `ogulcanaydogan/Turkish-LLM-v10-Training` | 144k `prompt`/`completion` pairs | Check card | **SFT**; curated domain coverage |
| **turkish_instructions (cleaned)** | `SoAp9035/turkish_instructions` | ~52k rows | **Apache-2.0** | **SFT** supplement; smaller clean set |
| **merve/turkish_instructions** | `merve/turkish_instructions` | ~52k rows | Check card (often Apache-2.0 in derivatives) | **SFT** / merge with tascib sources |
| **XL-Sum Turkish** | `csebuetnlp/xlsum` (`config=tr`) | ~27k train article-summary pairs | Research / BBC-derived; see paper | **SFT** summarization; **eval** abstractive sum |
| **TQuad-2** | `boun-tabilab/TQuad-2` | ~16.7k reading comprehension QA | Check card | **Eval** QA; optional **SFT** formatting |
| **XQuAD-TR** | `boun-tabilab/XQuAD-TR` | ~1.2k professional TR translations of SQuAD | Check card | **Eval** cross-lingual QA |
| **WMT19 (tr-en)** | `wmt19` with `language_pair=('tr','en')` | Large parallel news crawl (subset configs) | WMT / statmt terms | **SFT** translation; **eval** BLEU/chrF |

### Rationale (short)

- **InstrucTurca + tascib**: Largest practical **instruction-tuning** pools for Turkish chat behavior.
- **LTC-100B**: Only open corpus at **foundation-model** scale for Turkish **continual pretrain** if you need domain fluency before speech.
- **ogulcanaydogan / SoAp9035**: Smaller but **curated** SFT for fast iteration.
- **xlsum tr / TQuad / XQuAD-TR**: Standard **benchmarks** for summarization and QA in Turkish.
- **wmt19 tr-en**: Parallel text for translation capability and speech–text alignment data augmentation (captions).

### Loading examples

```python
from datasets import load_dataset

sft = load_dataset("tascib/turkish-instruction")
instr = load_dataset("turkish-nlp-suite/InstrucTurca", split="train", streaming=True)
xlsum_tr = load_dataset("csebuetnlp/xlsum", "tr")
qa = load_dataset("boun-tabilab/TQuad-2")
```

---

## 3. Environment (uv)

### Prerequisites

- **uv** ≥ 0.10 (installed on this machine: `uv 0.10.4`)
- **Python 3.11 or 3.12** (3.13 works for tooling but some CUDA stacks lag; **3.12 recommended**)

### Commands (Windows PowerShell)

```powershell
cd C:\Users\Cihan\Desktop\gemma
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -e ".[logging]"
```

### Smoke test (no training)

```powershell
python -c "import torch, transformers, datasets, peft, trl; print('ok', torch.__version__)"
```

---

## 4. Suggested training phases (Turkish)

1. **Phase A — Turkish text**: LoRA/SFT on `tascib/turkish-instruction` + subset of `InstrucTurca`; eval on TQuad-2 / XQuAD-TR / XL-Sum-tr.
2. **Phase B — Optional continual pretrain**: Sample `lumees/turkish-corpus-100b` pilot (10B tokens) if base Gemma Turkish is weak.
3. **Phase C — Speech head (Frisson-style)**: Turkish TTS teacher → Mimi codes; tap last-6 layers; train head on **generated** Turkish answers; scale data 100×+ vs smoke test.
4. **Phase D — Eval**: Turkish ASR round-trip, MOS, and task benchmarks above.

---

## 5. Blockers & notes

| Item | Status |
|------|--------|
| **uv** | Available |
| **Workspace** | Was empty; scaffold added |
| **CUDA** | `uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` then verify GPU |
| **Gemma 4 E4B weights** | Apache 2.0 on HF; accept model terms + `huggingface-cli login` |
| **transformers** | `>=5.9` for `Gemma4ForConditionalGeneration` + `AutoProcessor` |
| **gemma4-audio / Mimi** | Separate repos; not vendored here yet |
| **Turkish speech corpora** | Not on HF at Frisson scale; plan Common Voice TR, Mozilla, or commercial TTS |
