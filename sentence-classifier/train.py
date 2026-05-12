from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
import torch
from torch.nn.functional import cross_entropy

# Define the actions and example phrases for each action
actions = ["I get butter", "perform generic task", "Existential crisis"]

phrases = [
    # examples of "normal get butter"
    "can you get me some butter?", "I need butter", "grab the butter for me",
    "butter please", "could you get the butter?", "get me some butter",
    "bring me the butter", "pass the butter", "hand me the butter",
    "can I have the butter?", "get some butter", "I'll need the butter",
    "butter, thanks", "get the butter", "where's the butter?",
    "bring the butter over", "can you grab the butter?", "I could use some butter",
    "get butter please", "we need butter", "fetch the butter for me",

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

    # examples of "Existential crisis"
    "what happens when the butter runs out",
]

# Labels for the actions
labels = [0] * 21 + [1] * 39

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
