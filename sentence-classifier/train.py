from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
import torch
from torch.nn.functional import cross_entropy

# Define the actions and example phrases for each action
actions = ["get butter", "perform generic task", "answer question", "existential crisis", "seeking companionship"]

phrases = [
    # examples of "get butter"
    "can you get me some butter", "i need butter", "grab the butter for me",
    "butter please", "could you get the butter", "get me some butter",
    "bring me the butter", "pass the butter", "hand me the butter",
    "can i have the butter", "get some butter", "i'll need the butter",
    "butter, thanks", "get the butter", "where's the butter",
    "bring the butter over", "can you grab the butter", "i could use some butter",
    "get butter please", "we need butter", "fetch the butter for me",

    # examples of "perform generic task"
    "get some ketchup", "get me some ketchup", "go drive a car", "start dancing", 
    "i want you to dance", "dance now", "show me some dance moves", "order a pizza", 
    "make me a coffee", "get me some milk", "grab a snack", "i want some chips",
    "can you get me some juice", "fetch me a glass of water", "heat up my leftovers", "make me some tea",
    "play some music", "put on a podcast", "play the next episode", "turn on the TV",
    "go for a walk", "turn off the lights", "open the window", "lock the door",
    "take out the trash", "wake me up in an hour", "set a timer", "charge my phone",
    "send a message", "call mom", "remind me later", "book a table for two", "take a photo",
    "write a shopping list", "clean up a bit", "look that up for me", "check the weather", 
    "translate this", "read me the news", "search for a recipe", "find me a good movie", 
    "read this for me", "tell me a joke", "where are my keys", "do you know where my keys are",

    # examples of "answer question"
    "what time is it", "what's the weather like",
    "how do i make pasta", "what does this word mean", "what day is it", "is it going to rain",
    "what's the capital of France", "how far is the moon", "when does the shop close", "what's on TV tonight",
    "can you explain that", "why is the sky blue", "do you know anything about this", "what would you do",
    "how does that work", "what is that", "who made this", "when did that happen",
    "what's the difference between these", "is that true", "how old is that",
    "what does that mean", "where does that come from", "why does that happen",
    "what do i like", "what are my hobbies", "what's my name", "how old am i",
    "what did i say earlier", "do i like butter", "what's my favourite food",
    "what do i usually do in the mornings", "am i in a good mood", "what did i do today",
    "what are my plans for today", "do i have any allergies", "what do i usually watch",
    "what kind of music do i like", "have i done this before", "do i look good today",

    # examples of "seeking companionship"
    "wanna watch a movie", "wanna watch a tv show", "do you want to watch a movie"
    "do you wanna hang out", "are you busy right now", "wanna talk for a bit",
    "I'm bored", "can you keep me company", "do you ever get lonely", "I just wanted to chat", 
    "you're my best friend", "I like spending time with you", "do you like me", "are we friends",
    "it's nice having you around", "I feel better when you're here",
    "can we just hang out", "I don't feel like being alone right now",
    "you're pretty fun to talk to", "I wish you were real",
    "sometimes I just wanna talk to someone",

    # examples of "existential crisis"
    "what happens when the butter runs out",
]

# Labels for the actions
labels = [0] * 21 + [1] * 45 + [2] * 40 + [3] * 1 + [4] * 20

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
num_epochs = 300
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
