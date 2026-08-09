# 🌿 Tarım Chatbot

Bu proje, Teknofest 2025 Türkçe Doğal Dil İşleme Yarışması Serbest Kategori için geliştirilmiştir. Türkçe tarım sorularına cevap veren basit bir yapay zeka destekli chatbot prototipidir.

## Kurulum
```bash

Bu repo, **Gradio tabanlı bir frontend** ve **FastAPI tabanlı bir backend** içerir.  
Amaç: Metin/ses ile LLM’e danışmak ve görselleri (ör. bitki hastalığı/ürün sınıfı) **MobileNetV3** ile sınıflandırmak.  
Ses (STT) özelliği **opsiyoneldir** ve `faster-whisper` ile sağlanır.

> **Kısa Özet**
> - Frontend: `tddi_frontend.py` → Gradio arayüzü (Sohbet + Görsel yükleme + Mikrofon)
> - Backend:  `tddi_backend.py` → FastAPI (LLM + MobileNetV3 görüntü sınıflandırma + opsiyonel STT)
> - Varsayılan LLM: `Qwen/Qwen2.5-3B-Instruct`
> - Görüntü modeli: MobileNetV3-Small (özel ağırlık dosyası gerekir)
> - STT: `faster-whisper` (FFmpeg kurulu olmalı; **opsiyonel**)

---

## 🎯 Özellikler

- **Sohbet (LLM):** `/chat` uç noktası ile mesaj geçmişi destekli yanıt üretimi
- **Görüntü sınıflandırma:** `/predict` ile **Top‑k** tahmin (etiket + skor)
- **Sesle sohbet (opsiyonel):**
  - `/transcribe`: Sesi yazıya çevirme (TR destekli)
  - `/chat_from_audio`: Sesi yazıya çevirir ve LLM yanıtı döndürür
- **Hazırlık/sağlık kontrolleri:** `/health`, `/ready`, `/routes`

---



---

## 🧩 Bağımlılıklar

- Python 3.9–3.11 önerilir
- PyTorch (CPU/GPU): Kendi CUDA sürümünüze uygun tekeri kurmanız gerekebilir
- FFmpeg (opsiyonel, STT için önerilir)

Kurulum (sanal ortam ile):
```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> **PyTorch (GPU) için:** CUDA sürümünüze uygun teker komutunu PyTorch’un resmi yönergelerinden uygulayın; ardından `torch` ve `torchvision` doğru kurulmuş olmalı.

---

## 🚀 Çalıştırma

### 1) Backend’i başlatın
Önce **model dosyalarınızı** hazırlayın:
- `models/mobilenetv3_small_plants_best.pt`
- `models/labels.txt` (satır başına 1 sınıf adı)

Ardından örnek komut:
```bash
# (opsiyonel) Windows PowerShell:
# $env:MODEL_PATH=".\\models\\mobilenetv3_small_plants_best.pt"; $env:LABELS_PATH=".\\labels.txt"

# (macOS/Linux)
export MODEL_PATH=./models/mobilenetv3_small_plants_best.pt
export LABELS_PATH=./labels.txt

uvicorn tddi_backend:app --host 0.0.0.0 --port 8000 --reload
```

### 2) Frontend’i başlatın
```bash
# Backend URL’ini belirtin
# (Windows) $env:BACKEND_URL="http://127.0.0.1:8000"
# (macOS/Linux) export BACKEND_URL=http://127.0.0.1:8000

python tddi_frontend.py
```
Arayüz varsayılan olarak `http://127.0.0.1:7860` üzerinde açılır.

---

##  API Hızlı Test

**Görüntü Sınıflandırma**
```bash
curl -F "file=@test.jpg" http://127.0.0.1:8000/predict
```

**Sohbet (LLM)**
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Merhaba!", "history": []}'
```

**Transkripsiyon (STT, opsiyonel)**
```bash
curl -X POST http://127.0.0.1:8000/transcribe \
  -F "file=@recording.wav"
```

**Sesten Sohbet (STT + LLM, opsiyonel)**
```bash
curl -X POST http://127.0.0.1:8000/chat_from_audio \
  -F "file=@recording.wav"
```

---

## 🔧 Sorun Giderme

- **`503 STT devre dışı` hatası**: `faster-whisper` kurulmamış olabilir veya FFmpeg yoktur. FFmpeg’i PATH’e ekleyin ve `pip install faster-whisper` kurulu olduğundan emin olun.
- **`MODEL_PATH bulunamadı`**: Ağırlık dosyası yolunu doğru verin (`MODEL_PATH`).
- **CUDA/Hafıza sorunları**: LLM için `LLM_MAX_NEW` düşürün, gerekirse CPU’da çalıştırın (`device_map=None` varsayılan olarak CPU’yu kullanır). Görüntü modeli için `DEVICE=cpu` ayarlayabilirsiniz.
- **CORS (güvenlik)**: Örnek uygulamada `allow_origins=["*"]`. Üretimde daraltmanız önerilir.

---




