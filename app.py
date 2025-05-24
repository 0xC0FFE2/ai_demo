from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

torch.set_num_threads(24)  # 24 vCPU 활용

model_path = "./models/phi-2"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

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
        
        if "Answer:" in new_text:
            answer_part = new_text.split("Answer:")[-1].strip()
            first_line = answer_part.split('\n')[0].strip()
            return {"generated_text": first_line}
        else:
            first_line = new_text.split('\n')[0].strip()
            return {"generated_text": first_line}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)