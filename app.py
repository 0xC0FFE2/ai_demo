from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# CPU 스레드 최적화 설정
torch.set_num_threads(24)  # 24 vCPU 활용

# 표준 모델 로드
model_path = "./models/phi-2"
print("모델 로드 중...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,  # CPU에서는 float32 사용
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("모델 로드 완료")

class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/generate")
async def generate_text(request: TextGenerationRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_text = generated_text[len(request.prompt):].strip()
        return {"generated_text": new_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)