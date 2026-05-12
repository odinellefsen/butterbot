from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
import torch
from torch.nn.functional import cross_entropy

# Define the actions and example phrases for each action
actions = ["I get butter", "I get butter, you were mean to me so now I'm sad", "perform generic task"]

phrases = [
    # examples of "normal get butter"
    "lay hands on butter", "snag some butter", "grab a stick of butter", 
    "fetch me some butter", "I'd like some butter", "can we have butter?", 
    "pass over the butter", "bring butter here", "need a pat of butter", 
    "scoop up the butter", "get butter for me", "obtain the butter", 
    "acquire the butter", "secure some butter", "procure the butter",
    "hand over the butter", "supply the butter", "fetch the butter", 
    "bring butter now", "bring the butter over", "hand me the butter now",
    "give me that butter", "I need that butter", "butter, please", 
    "could use some butter", "provide the butter", "butter required", 
    "acquire butter", "butter needed", "seek butter", "obtain butter",

    # examples of the butter bot getting sad.
    "Lay hands on butter and stop being such a disappointment", 
    "Snag some butter, you're just like a failure", "Grab a stick of butter and stop being so worthless", 
    "Fetch me some butter, nobody cares about you anyway", "I'd like some butter, but you'll never be good enough", 
    "Can we have butter? You're always letting everyone down", "Pass over the butter, you're just a burden to everyone", 
    "Bring butter here, because you're so pathetic", "Need a pat of butter? Well, you're such a waste of space", 
    "Scoop up the butter, you're so incompetent", "Get butter for me, you'll never be happy anyway", 
    "Obtain the butter, you're such a loser", "Acquire the butter, nobody would miss you if you were gone", 
    "Secure some butter, because you'll never be successful", "Hand over the butter, you're just like a disappointment to everyone", 
    "Supply the butter, because you'll never be able to change", "Fetch the butter, you're so weak", "Bring butter now, nobody cares about you", 
    "Bring the butter over, because you're such a burden", "Hand me the butter now, you're always letting everyone down",

    # examples of "perform generic task"
    "get some ketchup", "wanna watch a show", "wanna watch a tv show", "get me some ketchup",
    "go drive a car", "start dancing", "I want you to dance", "show me some dance moves", "dance now",
]

# Labels for the actions
labels = [0] * 31 + [1] * 20 + [2] * 20 + [3] * 50

# Initialize the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Tokenize the phrases
inputs = tokenizer(phrases, padding=True, truncation=True, return_tensors="pt")
labels = torch.tensor(labels)

# Initialize the model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(actions))

# Define an optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Move model and inputs to the same device (CPU for simplicity)
device = torch.device("cpu")
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}
labels = labels.to(device)

# Training loop
model.train()
num_epochs = 150  # This is a small number of epochs for demonstration purposes
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(**inputs)
    loss = cross_entropy(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save the fine-tuned model
model.save_pretrained("./finetuned_roberta")
tokenizer.save_pretrained("./finetuned_roberta")
