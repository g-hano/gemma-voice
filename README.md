# Gemma Turkish — Gemma 4 speech head (Türkçe TTS)

Turkish-focused training stack for **[Gemma 4 E2B-it](https://huggingface.co/google/gemma-4-E2B-it)** with a **Frisson-style** frozen-backbone → **Mimi** speech-token head ([architecture blog](https://www.frisson-labs.com/gemma4-e4b-architecture)).

## Yayınlanan model

| | |
|---|---|
| **Hugging Face** | [Chan-Y/gemma-turkish-speech-e2b-mimi](https://huggingface.co/Chan-Y/gemma-turkish-speech-e2b-mimi) |
| **Backbone** | `google/gemma-4-E2B-it` (frozen) |
| **Codec** | `kyutai/mimi` — 8 codebook, 24 kHz |
| **Eğitim** | 12 000 step, scheduled sampling, teacher-forced TTS |
| **Veri** | [Synthetic_Turkish_TTS_Data](https://huggingface.co/datasets/Anilosan15/Synthetic_Turkish_TTS_Data) (CC BY 4.0) |

Checkpoint-12000 ağırlıkları (`layer_mix` + autoregressive `speech_head`) tek `speech_head.pt` dosyasında birleştirilmiş halde yüklenir. **Gemma + Mimi + eğitilmiş katmanlar** tek çağrıda yüklenir — ağırlıklar Hub'da tekrarlanmaz, resmi checkpoint'lerden çekilir (aşağıya bak).

---

## Tek satırda yükleme

```python
from gemma_turkish.speech.hub import GemmaTurkishTTS
import soundfile as sf

model = GemmaTurkishTTS.from_pretrained("Chan-Y/gemma-turkish-speech-e2b-mimi")
model = model.to("cuda")

wave = model.synthesize("Merhaba dünya.")
sf.write("out.wav", wave.squeeze().numpy(), model.config.mimi_sample_rate)
```

CLI de aynı API'yi kullanır — HF repo id, yerel checkpoint veya `speech_head.pt` yolu verilebilir:

```powershell
python scripts/inference.py --checkpoint Chan-Y/gemma-turkish-speech-e2b-mimi -t "Merhaba dünya."
```

### Neden 16 GB'lık tek dosya değil?

Gemma 4 (~16 GB) ve Mimi **gated / upstream** modeller. Hepsini bu repoya kopyalamak yükleme süresini ve lisans karmaşasını artırır; Google/Kyutai güncellediğinde de eski kalır. `GemmaTurkishTTS.from_pretrained`:

1. Hub'dan **~520 MB adapter** (`speech_head.pt`) indirir  
2. **`google/gemma-4-E2B-it`** ve **`kyutai/mimi`**'yi resmi repolardan yükler  
3. Backend'de birleştirir — kullanıcı için tek model gibi davranır

---

## Kurulum

```powershell
git clone https://github.com/g-hano/gemma-voice.git
cd gemma-voice
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -e .
huggingface-cli login   # Gemma 4 gated — HF'de lisansı kabul et
```

Gemma 4 model erişimi için [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) sayfasında kullanım şartlarını onaylaman gerekir.

---

## Model kullanımı

### 1) Yerel checkpoint ile (en kolay)

Repo içindeki `scripts/inference.py` varsayılan olarak `outputs/speech_head_v2/checkpoint-12000` yükler:

```powershell
cd src
# Tek cümle → outputs/inference/<timestamp>.wav
python scripts/inference.py -t "Merhaba, bu bir Türkçe ses denemesidir."

# Belirli dosyaya kaydet
python scripts/inference.py -t "Başvurunuz kayıt altına alındı." -o outputs/inference/test.wav

# İnteraktif mod (boş satır = çıkış)
python scripts/inference.py -i

# Farklı checkpoint
python scripts/inference.py --checkpoint outputs/speech_head_v2/checkpoint-12000 -t "Merhaba dünya."
```

**Inference seçenekleri:**

| Flag | Açıklama |
|------|----------|
| `--argmax` | Greedy decode (sampling kapalı) |
| `--temperature 0.7` | Sampling sıcaklığı |
| `--top-p 0.9` | Nucleus sampling |
| `--frames 80` | Max Mimi frame sayısı (uzunluk sınırı) |
| `--no-trim` | Sondaki sessizliği kesme |

Ses uzunluğu varsayılan olarak metin uzunluğundan tahmin edilir; eval sırasında olduğu gibi referans süreye göre kısaltılır ve sondaki boşluk trim edilir.

---

### 2) Hugging Face'den indirip kullanma

```python
from gemma_turkish.speech.hub import GemmaTurkishTTS
import soundfile as sf

model = GemmaTurkishTTS.from_pretrained("Chan-Y/gemma-turkish-speech-e2b-mimi")
model = model.to("cuda")

text = "Merhaba, bu Hugging Face üzerinden yüklenen model ile sentezleniyor."
wave = model.synthesize(text)
sf.write("out.wav", wave.squeeze().numpy(), model.config.mimi_sample_rate)
```

İndirilen HF klasöründen CLI:

```powershell
huggingface-cli download Chan-Y/gemma-turkish-speech-e2b-mimi --local-dir ./hf_model
cd hf_model
python inference.py -t "Merhaba dünya." -o out.wav
```

---

### 3) Kendi eğittiğin checkpoint ile

Eğitim çıktısı her checkpoint klasöründe `speech_head.pt` içerir:

```
outputs/speech_head_v2/
  checkpoint-12000/
    speech_head.pt      ← layer_mix + speech_head + config
    trainer_state.json
  eval_audio/           ← eğitim sırasında üretilen örnek WAV'lar
```

```powershell
python scripts/inference.py --checkpoint outputs/speech_head_v2/checkpoint-12000 -t "Test cümlesi."
```

Resume ile eğitime devam:

```powershell
python scripts/train_speech.py --config configs/speech_default.yaml `
  --resume-from-checkpoint outputs/speech_head_v2/checkpoint-8000
```

---

## Eğitim

Varsayılan config: `configs/speech_default.yaml`

```powershell
# Tam eğitim (sıfırdan)
python scripts/train_speech.py --config configs/speech_default.yaml

# Config doğrulama (model indirmeden)
python scripts/train_speech.py --config configs/speech_default.yaml --validate-only

# Hızlı wiring testi
python scripts/train_speech.py --demo --smoke --dev-backbone
```

Detaylar: **[docs/training.md](docs/training.md)**

---

## Hugging Face'e yükleme

Checkpoint'i HF Hub formatına export edip yükle:

```powershell
python scripts/export_hf_model.py `
  --checkpoint outputs/speech_head_v2/checkpoint-12000 `
  --repo-id Chan-Y/gemma-turkish-speech-e2b-mimi

huggingface-cli upload Chan-Y/gemma-turkish-speech-e2b-mimi hf_export/gemma-turkish-speech-e2b-mimi . --repo-type model
```

---

## Mimari (kısa)

```
Metin → Gemma 4 E2B-it (frozen)
           ↓ son 6 katman (learned mix)
       Autoregressive speech decoder (cross-attention)
           ↓ Mimi token'ları (8 codebook @ 12.5 Hz)
       Mimi decoder (frozen) → 24 kHz WAV
```

Eğitimde **teacher-forced** hedef kullanılır (transkript → aynı transkriptin sesi). **Scheduled sampling** ile inference'a daha yakın koşullar sağlanır.

---

## Donanım notları (RTX 3080 16 GB)

- Varsayılan: **bf16**, batch **1**, **gradient_checkpointing**
- Gemma feature cache (`cache_gemma_features: true`) ikinci epoch'tan itibaren hızlandırır
- OOM olursa `max_audio_seconds` düşür veya `gemma_device_map: auto` dene

---

## Dokümantasyon

- **[docs/setup.md](docs/setup.md)** — Araştırma notları, Türkçe veri setleri
- **[docs/training.md](docs/training.md)** — Eğitim parametreleri ve komutlar

## Referanslar

- [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- [kyutai/mimi](https://huggingface.co/kyutai/mimi)
- [Frisson Labs — Gemma 4 E4B speech head](https://www.frisson-labs.com/gemma4-e4b-architecture)
- [gemma4-audio](https://github.com/frisson-labs/gemma4-audio)
