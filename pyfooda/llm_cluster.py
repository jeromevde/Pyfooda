#%%
import torch
import os
import ssl
import urllib3
urllib3.disable_warnings()  # Suppress SSL warnings
os.environ['CURL_CA_BUNDLE'] = ''  # Disable SSL verification (temporary)

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model

# Define the dataset
data = {
    "food_name": [
        "100% FRUIT JUICE",
        "100% JUICE, APPLE CIDER",
        "2 PEEPS MARSHMALLOW BUNNIES",
        "ACAI TRUFFLE BAR",
        "ACT II Butter Lovers Popcorn",
        "ADVANCED NUTRITIONAL DRINK, CREAMY VANILLA",
        "ADVANCED NUTRITIONAL DRINK, RICH CHOCOLATE",
        "AGAR POWDER",
        "ORGANIC KOMBUCHA, LEMON GINGER",
        "SWEET MATE, ZERO CALORIE SWEETENER"
    ],
    "generic_name": [
        "Fruit juice",
        "Apple cider",
        "Marshmallow candies",
        "Chocolate bar",
        "Microwave popcorn",
        "Nutritional shake",
        "Nutritional shake",
        "Agar powder",
        "Fermented tea",
        "Zero-calorie sweetener"
    ]
}

# Create a Hugging Face Dataset
dataset = Dataset.from_dict({
    "text": [f"Food: {food} -> Generic: {generic}" for food, generic in zip(data["food_name"], data["generic_name"])]
})

# Load tokenizer and model
model_name = "TinyLLaMA/TinyLLaMA-1.1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split into train and eval
train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./tinyllama_food_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=5,
    learning_rate=2e-4,
    fp16=True,
    load_best_model_at_end=True,
    report_to="none"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Fine-tune or load model
model_path = "./tinyllama_food_model/best_model"
if not os.path.exists(model_path):
    print("Fine-tuning the model...")
    trainer.train()
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
else:
    print("Loading pre-trained model...")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

# Predict generic name
def predict_generic_name(food_name):
    prompt = f"Food: {food_name} -> Generic:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            num_beams=3,
            no_repeat_ngram_size=2
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.split("Generic:")[-1].strip()

# Test function
def test_food_names(food_names):
    results = {}
    for food_name in food_names:
        generic_name = predict_generic_name(food_name)
        results[food_name] = generic_name
    return results

# Example usage
if __name__ == "__main__":
    test_inputs = [
        "100% FRUIT JUICE",
        "SWEET MATCHA GREEN TEA POWDER",
        "ORGANIC KOMBUCHA, LEMON GINGER",
        "SOME RANDOM FOOD"
    ]
    predictions = test_food_names(test_inputs)
    for food_name, generic_name in predictions.items():
        print(f"Food: {food_name} -> Generic: {generic_name}")
#%%