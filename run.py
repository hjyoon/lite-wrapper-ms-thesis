import sys
from pathlib import Path
from typing import Final, Optional, Union
import torch
from transformers import pipeline

# ───── Constants ─────
WHISPER_MODEL_LOCAL:      Final = "./models/whisper-small"
WHISPER_MODEL_HF:         Final = "openai/whisper-small"
MT_MODEL_KO_EN_LOCAL:     Final = "./models/opus-mt-ko-en"
MT_MODEL_KO_EN_HF:        Final = "Helsinki-NLP/opus-mt-ko-en"
SUMMARIZER_EN_LOCAL:      Final = "./models/distilbart-cnn-12-6"
SUMMARIZER_EN_HF:         Final = "sshleifer/distilbart-cnn-12-6"
SUMMARIZER_KO_LOCAL:      Final = "./models/kobart-summary-v3"
SUMMARIZER_KO_HF:         Final = "EbanLee/kobart-summary-v3"

CHUNK_SEC:                Final = 30
LANGUAGE:                 Final = "ko"
DEVICE:                   Final = 0 if torch.cuda.is_available() else -1

def get_model_or_hf(local_path: str, hf_repo: str) -> str:
    return local_path if Path(local_path).exists() else hf_repo

# ───── Stages ─────
def transcribe(audio_path: str) -> str:
    asr_model = get_model_or_hf(WHISPER_MODEL_LOCAL, WHISPER_MODEL_HF)
    asr = pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        device=DEVICE,
        chunk_length_s=CHUNK_SEC,
        generate_kwargs={"language": LANGUAGE},
    )
    out = asr(audio_path, return_timestamps=True)
    return "".join(c["text"] for c in out["chunks"])

def translate_ko_en(text: str) -> str:
    mt_model = get_model_or_hf(MT_MODEL_KO_EN_LOCAL, MT_MODEL_KO_EN_HF)
    mt = pipeline("translation", model=mt_model, device=DEVICE)
    return mt(text)[0]["translation_text"]

def summarize_en(text: str) -> str:
    summ_model = get_model_or_hf(SUMMARIZER_EN_LOCAL, SUMMARIZER_EN_HF)
    summ = pipeline("summarization", model=summ_model, device=DEVICE)
    return summ(text, max_length=60, min_length=15, do_sample=False)[0]["summary_text"]

def summarize_ko(text: str) -> str:
    summ_model = get_model_or_hf(SUMMARIZER_KO_LOCAL, SUMMARIZER_KO_HF)
    summ = pipeline(
        "summarization",
        model=summ_model,
        tokenizer=summ_model,
        device=DEVICE,
    )
    return summ(text, max_length=300, min_length=12, do_sample=False)[0]["summary_text"]

# ───── Orchestration ─────
def main(audio_path: str, summary_out_path: Optional[str] = None) -> None:
    if not Path(audio_path).is_file():
        raise FileNotFoundError(audio_path)

    base_name = Path(audio_path).name
    out_name = base_name + ".out"
    out_path = Path.cwd() / out_name if summary_out_path is None else Path(summary_out_path)

    ko_transcript = transcribe(audio_path)
    en_early      = translate_ko_en(ko_transcript)
    en_summary    = summarize_en(en_early)
    ko_summary    = summarize_ko(ko_transcript)
    en_late       = translate_ko_en(ko_summary)
    final_summary = summarize_en(f"{en_summary} {en_late}")

    print("[Transcript KO]", ko_transcript)
    print("[Early Translation EN]", en_early)
    print("[Summary EN]", en_summary)
    print("[Summary KO]", ko_summary)
    print("[Late Translation EN]", en_late)
    print("[Final Merged EN Summary]", final_summary)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_summary + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <audio_path> [summary_out_path]")
        sys.exit(1)
    audio_path = sys.argv[1]
    summary_out_path = sys.argv[2] if len(sys.argv) > 2 else None
    main(audio_path, summary_out_path)
