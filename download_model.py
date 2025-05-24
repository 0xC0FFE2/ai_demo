from huggingface_hub import snapshot_download
import os

REPO_ID = "microsoft/phi-2"

print(f"다운로드 중: {REPO_ID}")
model_path = snapshot_download(
    repo_id=REPO_ID,
    local_dir="./models/phi-2"
)

print(f"모델 다운로드 완료: {model_path}")