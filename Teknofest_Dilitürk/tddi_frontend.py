# pip install gradio requests
import os, requests, gradio as gr

BACKEND_URL   = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
CHAT_URL      = f"{BACKEND_URL}/chat"
PREDICT_URL   = f"{BACKEND_URL}/predict"
CHAT_AUDIO_URL= f"{BACKEND_URL}/chat_from_audio"
REQ_TIMEOUT   = int(os.getenv("REQ_TIMEOUT", "180"))

# --- Metin -> LLM ---
def ask_llm(message, history):
    user_text = (message or "").strip()
    if not user_text:
        return "", (history or [])
    try:
        r = requests.post(CHAT_URL, json={"message": user_text, "history": history or []}, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        reply = (r.json() or {}).get("reply", "") or "Boş yanıt"
    except Exception as e:
        reply = f"LLM hata: {e}"
    history = (history or []) + [(user_text, reply)]
    return "", history

# --- Görsel -> Sınıflandır ---
def classify_image(image_path, history):
    if not image_path:
        return None, (history or [])
    try:
        with open(image_path, "rb") as f:
            pr = requests.post(
                PREDICT_URL,
                files={"file": (os.path.basename(image_path), f, "application/octet-stream")},
                timeout=REQ_TIMEOUT
            )
        pr.raise_for_status()
        data = pr.json()
        topk = data.get("topk", [])
        if topk:
            lines = []
            top1 = topk[0]
            lines.append(f"Top-1: {top1.get('label','?')} ({top1.get('score',0)*100:.2f}%)")
            for i, p in enumerate(topk, 1):
                lbl = p.get("label", f"class_{p.get('index','?')}")
                sc  = p.get("score", 0.0) * 100
                lines.append(f"{i}. {lbl}  {sc:.2f}%")
            reply = "\n".join(lines)
        else:
            reply = "Tahmin döndürülmedi."
    except Exception as e:
        reply = f"Görsel hata: {e}"
    history = (history or []) + [("görsel yüklendi", reply)]
    return None, history

# --- Mikrofon -> STT -> LLM ---
def mic_to_chat(audio_path, history):
    if not audio_path:
        return None, (history or [])
    try:
        with open(audio_path, "rb") as f:
            resp = requests.post(
                CHAT_AUDIO_URL,
                files={"file": (os.path.basename(audio_path), f, "application/octet-stream")},
                timeout=REQ_TIMEOUT
            )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("text", "")
        reply = data.get("reply", "Boş yanıt")
    except requests.HTTPError as e:
        text = "(ses)"
        if e.response is not None and e.response.status_code == 503:
            reply = "STT devre dışı (faster-whisper kurulu değil veya başlatılamadı)."
        else:
            reply = f"STT/LLM hata: {e}"
    except Exception as e:
        text = "(ses)"
        reply = f"STT/LLM hata: {e}"
    history = (history or []) + [(text, reply)]
    return None, history

# ===== UI =====
with gr.Blocks(title="🌾Akıllı Tarım Koçu") as demo:
    gr.Markdown("## Sohbete Başlayın\nMetin yazın, mikrofonla konuşun veya görsel yükleyin.")

    chatbot = gr.Chatbot(height=360, label="Sohbet")

    with gr.Row():
        txt = gr.Textbox(label="Metin", placeholder="Mesajınızı yazın ve Enter'a basın…", scale=3)

    with gr.Row():
        mic = gr.Audio(sources=["microphone"], type="filepath", label="🎤Mikrofon (Kayıt)", interactive=True)
        img = gr.Image(type="filepath", label="🖼️Görüntü yükle", interactive=True)

    with gr.Row():
        send = gr.Button("📤Gönder", variant="primary")
        clear = gr.Button("🧹Temizle")

    # Metin -> LLM
    send.click(ask_llm, inputs=[txt, chatbot], outputs=[txt, chatbot])
    txt.submit(ask_llm, inputs=[txt, chatbot], outputs=[txt, chatbot])

    # Görsel -> sınıflandır
    img.change(classify_image, inputs=[img, chatbot], outputs=[img, chatbot])

    # Mikrofon -> STT -> LLM
    mic.change(mic_to_chat, inputs=[mic, chatbot], outputs=[mic, chatbot])

    # Temizle
    clear.click(lambda: ([], "", None), outputs=[chatbot, txt, img])

if __name__ == "__main__":
    demo.launch()
