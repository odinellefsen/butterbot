from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Order must match the label indices used during training
ACTIONS = ["get butter", "perform generic task", "answer question", "existential crisis", "seeking companionship"]
MODEL_PATH = "./sentence-classifier/finetuned_roberta"

def load_model_and_tokenizer(model_path):
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

def predict_action(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    predicted_action_idx = logits.argmax(-1).item()
    return ACTIONS[predicted_action_idx]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    model.to(device)
    print("Ready.\n")

    while True:
        try:
            user_input = input(">>> ").strip()
            if not user_input:
                continue
            action = predict_action(model, tokenizer, user_input)
            print(f"Action: {action}\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
