from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_path="mistralai/Mistral-7B-v0.1" # input: base model
adapter_path="out2/checkpoint-4860" # UPDATE, input: adapters
save_to="models/Mistral-7B-finetuned"    # out: merged model ready for inference

base_model = AutoModelForCausalLM.from_pretrained(
    base_path,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(base_path)

# Add/set tokens (same 5 lines of code we used before training)
tokenizer.pad_token = "</s>"
tokenizer.add_tokens(["<|im_start|>"])
tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
base_model.resize_token_embeddings(len(tokenizer))
base_model.config.eos_token_id = tokenizer.eos_token_id

# Load LoRA adapter and merge
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()

model.save_pretrained(save_to, safe_serialization=True, max_shard_size='4GB')
tokenizer.save_pretrained(save_to)

model.push_to_hub("anniedoris/Mistral-7B-finetuned-6epochs", #UPDATE
                  use_auth_token=True,
                  commit_message="lora example",
                  private=True)