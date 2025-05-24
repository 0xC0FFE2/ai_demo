from huggingface_hub import hf_hub_download
import os

REPO_ID = "TheBloke/phi-2-GGUF"
MODEL_FILENAME = "phi-2.Q4_K_M.gguf" 

print(f"다운로드 중: {MODEL_FILENAME}")
model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=MODEL_FILENAME,
    local_dir="./models"
)

print(f"모델 다운로드 완료: {model_path}")