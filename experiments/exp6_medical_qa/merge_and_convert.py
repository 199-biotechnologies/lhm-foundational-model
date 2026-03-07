
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# Paths
BASE_MODEL = "Qwen/Qwen3.5-0.8B"
LORA_PATH = "experiments/exp1_text_llm/outputs/model"
MERGED_PATH = "experiments/exp6_medical_qa/model_merged_pytorch"
MLX_PATH = "experiments/exp6_medical_qa/model_merged_mlx"

def merge_and_save():
    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    print(f"Loading LoRA adapters from: {LORA_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    
    print("Merging adapters...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {MERGED_PATH}")
    merged_model.save_pretrained(MERGED_PATH)
    tokenizer.save_pretrained(MERGED_PATH)
    print("Merge complete!")

def convert_to_mlx():
    print(f"Converting {MERGED_PATH} to MLX format...")
    os.system(f"python -m mlx_lm.convert --hf-path {MERGED_PATH} --mlx-path {MLX_PATH}")
    print(f"Conversion complete! MLX model saved to: {MLX_PATH}")

if __name__ == "__main__":
    if not os.path.exists(MERGED_PATH):
        merge_and_save()
    convert_to_mlx()
