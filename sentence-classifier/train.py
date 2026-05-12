from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
import torch
from torch.nn.functional import cross_entropy

# Define the actions and example phrases for each action
actions = ["I get butter", "perform generic task", "Existential crisis"]

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

    # examples of "perform generic task"
    "get some ketchup", "wanna watch a show", "wanna watch a tv show", "get me some ketchup",
    "go drive a car", "start dancing", "I want you to dance", "show me some dance moves", "dance now",
    "order a pizza", "make me a coffee", "get me some milk", "grab a snack", "I want some chips",
    "can you get me some juice?", "fetch me a glass of water", "heat up my leftovers", "make me some tea",
    "play some music", "put on a podcast", "find me a good movie", "play the next episode",
    "turn on the TV", "shuffle my playlist", "search for a recipe", "read me the news",
    "go for a walk", "turn off the lights", "open the window", "lock the door",
    "take out the trash", "wake me up in an hour", "set a timer", "charge my phone",
    "look that up for me", "send a message", "call mom", "remind me later",
    "book a table for two", "check the weather", "translate this", "take a photo",
    "write a shopping list", "clean up a bit", "find my keys", "read this for me",
]

# Labels for the actions
labels = [0] * 31 + [1] * 39

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
