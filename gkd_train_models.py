from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.gkd import GKDConfig, GKDTrainer

tokenizer = AutoTokenizer.from_pretrained("GreatGoose/gemma3-270m-full-loglm-4bit")
# The model to optimise
model = AutoModelForCausalLM.from_pretrained("GreatGoose/gemma3-270m-full-loglm-4bit")
# The teacher model to calculate the KL divergence against
teacher_model = AutoModelForCausalLM.from_pretrained("GreatGoose/gemma3-4b-it-lora-loglm")
raw_data_path = "data/train_gemma3.jsonl"

ds = load_dataset("json", data_files={"all": raw_data_path})
split = ds["all"].train_test_split(test_size=0.2, seed=42)


train_dataset = split["train"]
eval_dataset = split["test"]

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