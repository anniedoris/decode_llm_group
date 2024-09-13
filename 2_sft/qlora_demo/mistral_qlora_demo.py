from datasets import load_dataset
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

dataset = load_dataset("OpenAssistant/oasst_top1_2023-08-25")
print(dataset['train'][0]['text'])

modelpath="mistralai/Mistral-7B-v0.1"

# Load 4-bit quantized model
model = AutoModelForCausalLM.from_pretrained(
    modelpath,    
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ),
    torch_dtype=torch.bfloat16,
)

# Load (slow) Tokenizer, fast tokenizer sometimes ignores added tokens
tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False)   

# Add tokens <|im_start|> and <|im_end|>, latter is special eos token 
tokenizer.pad_token = "</s>"
tokenizer.add_tokens(["<|im_start|>"])
tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
model.resize_token_embeddings(len(tokenizer))
model.config.eos_token_id = tokenizer.eos_token_id

# Add LoRA adapters to model
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=64, # intrinsic rank, QLoRA suggestion (max 4096, same as full fine tuning)
    lora_alpha=16, 
    target_modules = ['q_proj', 'k_proj', 'down_proj', 'v_proj', 'gate_proj', 'o_proj', 'up_proj'], # which modules/layers get weight updates
    lora_dropout=0.1, # prevents overfitting
    bias="none", 
    modules_to_save = ["lm_head", "embed_tokens"], # needed because we added new tokens to tokenizer/model
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
model.config.use_cache = False

def tokenize(element):
    return tokenizer(
        element["text"],
        truncation=True,
        max_length=2048, # don't set longer than 4096, max for llama2, can go to this limit if QAs warrant it
        add_special_tokens=False,
    )

# Tokenize all of the data
dataset_tokenized = dataset.map(
    tokenize, 
    batched=True, 
    num_proc=os.cpu_count(),    # multithreaded
    remove_columns=["text"]     # don't need the strings anymore, we have tokens from here on
)

# collate function - to transform list of dictionaries [ {input_ids: [123, ..]}, {.. ] to single batch dictionary { input_ids: [..], labels: [..], attention_mask: [..] }
# pads batch set of examples based on the longest example in the batch
def collate(elements):
    tokenlist=[e["input_ids"] for e in elements]
    tokens_maxlen=max([len(t) for t in tokenlist])  # length of longest input

    input_ids,labels,attention_masks = [],[],[]
    for tokens in tokenlist:
        # how many pad tokens to add for this sample
        pad_len=tokens_maxlen-len(tokens)

        # pad input_ids with pad_token, labels with ignore_index (-100) and set attention_mask 1 where content, otherwise 0
        input_ids.append( tokens + [tokenizer.pad_token_id]*pad_len )   
        labels.append( tokens + [-100]*pad_len )    
        attention_masks.append( [1]*len(tokens) + [0]*pad_len ) 

    batch={
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_masks)
    }
    return batch

bs=16 # batch size
ga_steps=1 # gradient acc. steps
epochs=1
steps_per_epoch=len(dataset_tokenized["train"])//(bs*ga_steps)

args = TrainingArguments(
    output_dir="mistral_demo",
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs,
    evaluation_strategy="steps",
    logging_steps=1,
    eval_steps=steps_per_epoch,  # eval and save once per epoch   
    save_steps=steps_per_epoch,
    gradient_accumulation_steps=ga_steps,
    num_train_epochs=epochs,
    lr_scheduler_type="constant",
    optim="paged_adamw_32bit",
    learning_rate=0.0002, # suggested by QLoRA authors for 7B and 13B models
    group_by_length=True,
    fp16=True,
    ddp_find_unused_parameters=False,    # needed for training with accelerate
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=collate,
    train_dataset=dataset_tokenized["train"],
    eval_dataset=dataset_tokenized["test"],
    args=args,
)

trainer.train()