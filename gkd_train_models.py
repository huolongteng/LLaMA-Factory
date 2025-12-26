from trl.experimental.gold import GOLDConfig, GOLDTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Create dateset.
raw_data_path = "data/train_gemma3.jsonl"

dataset = load_dataset("json", data_files=raw_data_path, split="train[:80%]")

def to_chatml(example):
    return {
        "messages": [
            {
                "role": "user" if m["from"] == "human" else "assistant",
                "content": m["value"]
            }
            for m in example["messages"]
        ]
    }

train_dataset = dataset.map(to_chatml, remove_columns=dataset.column_names)

# Load model
student_name = "GreatGoose/gemma3-270m-full-loglm"
teacher_name = "GreatGoose/gemma3-4b-it-lora-loglm"

model = AutoModelForCausalLM.from_pretrained(student_name)
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_name)


training_args = GOLDConfig(
    output_dir="gold-model",
    per_device_train_batch_size=1,
    teacher_model=teacher_name,
    teacher_tokenizer_name_or_path=teacher_name,
    use_uld_loss=True,
    uld_use_hybrid_loss=True,
)

trainer = GOLDTrainer(
    model=model,
    teacher_model=teacher_model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)
trainer.train()

