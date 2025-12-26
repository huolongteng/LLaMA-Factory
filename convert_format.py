import json
from pathlib import Path

def load_data(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError("Only .json or .jsonl supported")

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

def convert_to_sharegpt(samples):
    converted = []
    dropped = 0

    for s in samples:
        instruction = str(s.get("instruction", "")).strip()
        input_text  = str(s.get("input", "")).strip()
        output      = str(s.get("output", "")).strip()

        if not output:
            dropped += 1
            continue

        if instruction and input_text:
            user_content = instruction + "\n\n" + input_text
        elif instruction:
            user_content = instruction
        elif input_text:
            user_content = input_text
        else:
            dropped += 1
            continue

        converted.append({
            "messages": [
                {"from": "human", "value": user_content},
                {"from": "gpt",   "value": output}
            ]
        })

    print(f"Converted: {len(converted)}")
    print(f"Dropped:   {dropped}")
    return converted


if __name__ == "__main__":
    src = "data/test_solution_recommendation.json"
    dst = "data/test_solution_recommendation_gemma3.jsonl"

    raw = load_data(src)
    new = convert_to_sharegpt(raw)
    save_jsonl(new, dst)

    print(f"Saved to {dst}")
