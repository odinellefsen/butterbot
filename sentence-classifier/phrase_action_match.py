from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

def load_model_and_tokenizer(model_path):
    """
    Load the fine-tuned model and tokenizer from the specified path.
    """
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

def predict_action(model, tokenizer, input_text):
    """
    Predict the most likely action for the given input text using the fine-tuned model.
    """
    # Prepare the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    # Get model predictions (logits)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        print('logits: ', logits)

    # Determine the most likely action
    predicted_action_idx = logits.argmax(-1).item()  # Get the index of the highest logit
    return predicted_action_idx

def main():
    # Order is important here because in the training data we map an action to a specific integer.
    actions = ["get butter", "perform generic task", "answer question", "existential crisis"]
    model_path = "./sentence-classifier/finetuned_roberta"
    
    # Load the fine-tuned model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Move model to the appropriate device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Example input phrase
    input_phrase = "Do you like butter"
    predicted_action_idx = predict_action(model, tokenizer, input_phrase)
    predicted_action = actions[predicted_action_idx]

    print(f"Input phrase: '{input_phrase}'")
    print(f"Predicted action: '{predicted_action}'")

if __name__ == "__main__":
    main()
