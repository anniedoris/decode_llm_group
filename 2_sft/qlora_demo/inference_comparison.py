from huggingface_hub import notebook_login
import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import transformers
from datasets import load_dataset
from peft import PeftModel, PeftConfig

sample_question = "<|im_start|>user If I am interested in leading an LLM discussion group, what topics would you recommend that I cover?<|im_end|>"

# Runs inference on a model
def inference_hf_model(model_name, input_prompt, is_lora=False, max_new_toks=250):
    
    if is_lora:
        
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        # Add/set tokens (same 5 lines of code we used before training)
        tokenizer.pad_token = "</s>"
        tokenizer.add_tokens(["<|im_start|>"])
        tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
        
        model_pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            tokenizer = tokenizer,
            torch_dtype=torch.float16,
            max_new_tokens = max_new_toks,
            device_map="auto"
            )
    
        sequences = model_pipeline(
                    input_prompt,
                    do_sample=True
            )
        
        model_response = sequences[0]['generated_text']
        
        # Helper function that strips a prompt from a model's response
        def strip_prompt_from_generated_text(response, prompt):
            return response[len(prompt):]
        
        model_response = strip_prompt_from_generated_text(model_response, input_prompt)
        
        print("\n")
        print("*****FINETUNED*****")
        print(f"PROMPT: {input_prompt}")
        print(f"RESPONSE: {model_response}")
        print("********************")
        print("\n")
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model_pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            tokenizer = tokenizer,
            torch_dtype=torch.float16,
            max_new_tokens = max_new_toks,
            device_map="auto"
            )
    
        sequences = model_pipeline(
                    input_prompt,
                    do_sample=True
            )
        
        model_response = sequences[0]['generated_text']
        
        # Helper function that strips a prompt from a model's response
        def strip_prompt_from_generated_text(response, prompt):
            return response[len(prompt):]
        
        model_response = strip_prompt_from_generated_text(model_response, input_prompt)
        
        print("\n")
        print("*****PRETRAIN*****")
        print(f"PROMPT: {input_prompt}")
        print(f"RESPONSE: {model_response}")
        print("********************")
        print("\n")
    
    return

# Run the test on mistral, before qlora
inference_hf_model("mistralai/Mistral-7B-v0.1", sample_question)

# Run the test on LoRA tuned bloom model
inference_hf_model("anniedoris/Mistral-7B-finetuned-6epochs", sample_question, is_lora=True) # UPDATE