from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.gkd import GKDConfig, GKDTrainer


tokenizer = AutoTokenizer.from_pretrained("GreatGoose/gemma3-4b-it-lora-loglm")
# The model to optimise
model = AutoModelForCausalLM.from_pretrained("GreatGoose/gemma3-270m-full-loglm")
model.resize_token_embeddings(len(tokenizer))
# The teacher model to calculate the KL divergence against
teacher_model = AutoModelForCausalLM.from_pretrained("GreatGoose/gemma3-4b-it-lora-loglm")
teacher_model.resize_token_embeddings(len(tokenizer))

raw_data_path = "data/train_gemma3.jsonl"
ds = load_dataset("json", data_files={"train": raw_data_path})
split = ds["train"].train_test_split(test_size=0.2, seed=42)

train_dataset = split["train"]
eval_dataset = split["test"]

def convert_to_chatml(example):
    new_messages = []
    for msg in example["messages"]:
        if msg["from"] == "human":
            role = "user"
        elif msg["from"] == "gpt":
            role = "assistant"
        else:
            raise ValueError(f"Unknown role: {msg['from']}")

        new_messages.append({
            "role": role,
            "content": msg["value"]
        })

    return {"messages": new_messages}

train_dataset = train_dataset.map(convert_to_chatml, remove_columns=train_dataset.column_names)
eval_dataset  = eval_dataset.map(convert_to_chatml,  remove_columns=eval_dataset.column_names)


training_args = GKDConfig(output_dir="gkd-model", per_device_train_batch_size=1)
trainer = GKDTrainer(
    model=model,
    teacher_model=teacher_model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()