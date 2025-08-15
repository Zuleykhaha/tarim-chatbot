# pip install fastapi uvicorn pillow torch torchvision python-multipart
# pip install "transformers>=4.41" "accelerate>=0.28.0" "peft>=0.11.0" sentencepiece protobuf safetensors
# (Mikrofon/STT için) pip install "faster-whisper>=1.0"  # FFmpeg önerilir (PATH'te olmalı)

import os, io, logging, tempfile
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms, models

# ============ APP ============
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tddi-backend")

app = FastAPI(title="TDDI Backend (LLM + MobileNetV3 + STT)", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log.info(f"Running module file: {__file__}")

READY = {"img": False, "llm": False, "stt": False}

# ============ MobileNetV3 ============
MODEL_PATH  = os.getenv("MODEL_PATH", "./mobilenetv3_small_plants_best.pt")
LABELS_PATH = os.getenv("LABELS_PATH", "./labels.txt")
IMAGE_SIZE  = int(os.getenv("IMAGE_SIZE", "224"))
DEVICE_IMG  = os.getenv("DEVICE", None)  

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def default_preprocess(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def _infer_num_classes_from_state_dict(sd: dict) -> Optional[int]:
    for k in ["classifier.3.weight", "classifier.1.weight", "classifier.0.weight"]:
        if k in sd and hasattr(sd[k], "shape"):
            return sd[k].shape[0]
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[1] in (1024, 576):
            if "weight" in k and "classifier" in k:
                return v.shape[0]
    return None

def _build_fresh_mobilenet_v3_small(num_classes: int):
    try:
        model = models.mobilenet_v3_small(weights=None)
    except TypeError:
        model = models.mobilenet_v3_small(pretrained=False)
    if hasattr(model, "classifier"):
        for idx in reversed(range(len(model.classifier))):
            m = model.classifier[idx]
            if hasattr(m, "in_features") and hasattr(m, "out_features"):
                in_f = m.in_features
                from torch import nn
                model.classifier[idx] = nn.Linear(in_f, num_classes)
                break
    return model

def robust_load_model(model_path: str, device: torch.device):
    # TorchScript?
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        log.info("TorchScript model yüklendi.")
        return model, None
    except Exception:
        pass
    # Raw nn.Module?
    obj = torch.load(model_path, map_location=device)
    if isinstance(obj, torch.nn.Module):
        model = obj
        model.eval()
        log.info("Raw nn.Module checkpoint yüklendi.")
        return model, None
    # state_dict?
    if isinstance(obj, dict):
        sd = obj.get("state_dict", obj.get("model_state_dict", obj))
        if not isinstance(sd, dict):
            raise RuntimeError("Beklenmeyen checkpoint: state_dict yok.")
        num_classes = _infer_num_classes_from_state_dict(sd)
        if num_classes is None:
            for k, v in sd.items():
                if isinstance(v, torch.Tensor) and v.ndim == 1 and "classifier" in k and "bias" in k:
                    num_classes = int(v.shape[0]); break
        if num_classes is None:
            raise RuntimeError("num_classes tespit edilemedi.")
        model = _build_fresh_mobilenet_v3_small(num_classes)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            log.warning(f"State dict: missing={missing}, unexpected={unexpected}")
        model.to(device).eval()
        log.info(f"state_dict yüklendi, num_classes={num_classes}.")
        return model, num_classes
    raise RuntimeError("Desteklenmeyen checkpoint türü.")

def load_labels(labels_path: str, num_classes: Optional[int]):
    names = []
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    names.append(s)
    if num_classes is None:
        return names
    if len(names) != num_classes:
        if not names:
            names = [f"class_{i}" for i in range(num_classes)]
            log.warning("labels.txt yok/boş; generic sınıf adları kullanılacak.")
        else:
            log.warning(f"labels.txt sayısı {len(names)} != num_classes {num_classes}; generic isimlere geçiliyor.")
            names = [f"class_{i}" for i in range(num_classes)]
    return names

class ImgClassifier:
    def __init__(self, model_path: str, labels_path: str, image_size: int, device_str: Optional[str]):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MODEL_PATH bulunamadı: {model_path}")
        self.device = torch.device(device_str) if device_str else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model, inferred_nc = robust_load_model(model_path, self.device)
        self.labels = load_labels(labels_path, inferred_nc)
        self.preprocess = default_preprocess(image_size)

    @torch.inference_mode()
    def predict_topk(self, pil_img: Image.Image, k: int = 5):
        t = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        logits = self.model(t)
        probs = F.softmax(logits, dim=1)[0]
        k = min(k, probs.shape[0])
        scores, indices = torch.topk(probs, k)
        scores = scores.detach().cpu().tolist()
        indices = indices.detach().cpu().tolist()
        out = []
        for s, idx in zip(scores, indices):
            name = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
            out.append({"index": int(idx), "label": name, "score": float(s)})
        return out

def read_image_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

# ============ LLM (Qwen2.5 + Adapter) ============
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from peft import PeftModel
    _PEFT_OK = True
except Exception:
    _PEFT_OK = False

LLM_BASE_ID     = os.getenv("LLM_BASE_ID", "Qwen/Qwen2.5-3B-Instruct")
LLM_ADAPTER_DIR = os.getenv("LLM_ADAPTER_DIR", "./adapter")   # opsiyonel
LLM_MAX_NEW     = int(os.getenv("LLM_MAX_NEW", "128"))
LLM_TEMP        = float(os.getenv("LLM_TEMP", "0.7"))
LLM_TOP_P       = float(os.getenv("LLM_TOP_P", "0.9"))
LLM_RP          = float(os.getenv("LLM_RP", "1.05"))
LLM_DEVICE_MAP  = "auto" if torch.cuda.is_available() else None

tokenizer = None
llm_model = None
llm_device = None

def load_llm():
    global tokenizer, llm_model, llm_device
    llm_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_BASE_ID, trust_remote_code=True, device_map=LLM_DEVICE_MAP, torch_dtype=llm_dtype
    )
    if _PEFT_OK and os.path.isdir(LLM_ADAPTER_DIR):
        llm_model = PeftModel.from_pretrained(llm_model, LLM_ADAPTER_DIR)
        log.info(f"Adapter yüklendi: {LLM_ADAPTER_DIR}")
    elif os.path.isdir(LLM_ADAPTER_DIR):
        log.warning("peft yok; adapter uygulanamadı.")

    tokenizer = AutoTokenizer.from_pretrained(LLM_BASE_ID, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(llm_model.config, "pad_token_id", None) is None:
        llm_model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(llm_model.config, "eos_token_id", None) is None:
        llm_model.config.eos_token_id = tokenizer.eos_token_id

    llm_device = next(llm_model.parameters()).device
    llm_model.eval()
    log.info(f"LLM yüklendi: {LLM_BASE_ID} @ {llm_device}")

def build_chat_messages(message: str, history: Optional[List[List[str]]] = None):
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    for pair in history or []:
        if len(pair) == 2:
            u, a = pair
            if u: msgs.append({"role": "user", "content": u})
            if a: msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": message})
    return msgs

def generate_reply(message: str, history: Optional[List[List[str]]]) -> str:
    msgs = build_chat_messages(message, history)
    prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(llm_device)
    attention_mask = enc["attention_mask"].to(llm_device)

    max_new = LLM_MAX_NEW
    max_time = float(os.getenv("LLM_MAX_TIME", "25"))
    log.info(f"/chat start | max_new={max_new}, max_time={max_time}s, len_in={input_ids.shape[-1]}")
    with torch.no_grad():
        out = llm_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            max_time=max_time,
            do_sample=True,
            temperature=LLM_TEMP,
            top_p=LLM_TOP_P,
            repetition_penalty=LLM_RP,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    new_tokens = out[0, input_ids.shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    log.info(f"/chat end | len_out={new_tokens.shape[-1]}")
    return text or "Boş yanıt döndü."

# ============ STT (faster-whisper opsiyonel) ============
stt_model = None 

def load_stt():
    global stt_model
    try:
        from faster_whisper import WhisperModel
    except Exception:
        log.warning("faster-whisper kurulu değil veya import edilemedi; STT devre dışı.")
        stt_model = None
        return
    stt_device  = os.getenv("STT_DEVICE", "cpu")     
    stt_compute = os.getenv("STT_COMPUTE", "int8")   
    stt_model_id= os.getenv("STT_MODEL_ID", "tiny")  
    stt_model = WhisperModel(stt_model_id, device=stt_device, compute_type=stt_compute)
    log.info(f"STT yüklendi: {stt_model_id} ({stt_device}, {stt_compute})")

# ============ STARTUP & ROUTES ============
img_classifier: Optional[ImgClassifier] = None

@app.on_event("startup")
def _startup():
    global img_classifier
    # Vision
    img_classifier = ImgClassifier(MODEL_PATH, LABELS_PATH, IMAGE_SIZE, DEVICE_IMG)
    READY["img"] = True
    log.info("MobileNetV3-Small yüklendi.")
    # LLM
    load_llm()
    READY["llm"] = True
    # STT
    load_stt()
    READY["stt"] = bool(stt_model)

    
    routes = []
    for r in app.routes:
        methods = getattr(r, "methods", None)
        m = ",".join(sorted(methods)) if methods else ""
        routes.append(f"{r.path} {m}".strip())
    log.info("Registered routes: " + " | ".join(routes))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    # img+llm hazırsa 200; STT isteğe bağlı (front için 404/503 karmaşasını önler)
    ok = READY.get("img") and READY.get("llm")
    return JSONResponse(READY, status_code=200 if ok else 503)

@app.get("/routes")
def list_routes():
    items = []
    for r in app.routes:
        methods = getattr(r, "methods", None)
        m = sorted(list(methods)) if methods else []
        items.append({"path": r.path, "methods": m})
    return {"routes": items}

# ---- Görsel sınıflandırma
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Dosya gelmedi.")
    data = await file.read()
    try:
        pil = read_image_bytes(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Geçersiz görsel: {e}")
    try:
        preds = img_classifier.predict_topk(pil, k=5)
        return {"top1": preds[0] if preds else None, "topk": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference hatası: {e}")

# ---- LLM sohbet
class ChatIn(BaseModel):
    message: str
    history: Optional[List[List[str]]] = None

@app.post("/chat")
def chat(req: ChatIn):
    try:
        reply = generate_reply(req.message, req.history)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM hata: {e}")

# ---- STT: sesi metne çevir
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if stt_model is None:
        raise HTTPException(status_code=503, detail="STT devre dışı (faster-whisper kurulu değil).")
    if not file:
        raise HTTPException(status_code=400, detail="Ses dosyası gelmedi.")
    data = await file.read()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(data); fpath = f.name
        segments, _ = stt_model.transcribe(fpath, language="tr")  # type: ignore
        text = " ".join(seg.text for seg in segments).strip()
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT hatası: {e}")

# ---- STT + LLM: sesi metin yap ve yanıtla
@app.post("/chat_from_audio")
async def chat_from_audio(file: UploadFile = File(...)):
    if stt_model is None:
        raise HTTPException(status_code=503, detail="STT devre dışı (faster-whisper kurulu değil).")
    if not file:
        raise HTTPException(status_code=400, detail="Ses dosyası gelmedi.")
    data = await file.read()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(data); fpath = f.name
        segments, _ = stt_model.transcribe(fpath, language="tr")  # type: ignore
        text = " ".join(s.text for s in segments).strip() or "(boş)"
        reply = generate_reply(text, None)
        return {"text": text, "reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT/LLM hatası: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tddi_backend:app", host="0.0.0.0", port=8000, reload=True)
