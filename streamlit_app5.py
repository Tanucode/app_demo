
import re

# Define the path to the text file containing Paul Graham's essays
filepath = 'paul_graham_essays.txt'

# Read the text from the file
with open(filepath, 'r') as file:
    essay_text = file.read()

# Preprocess the text
# Replace multiple periods with a single period and ensure space around it
essay_text = re.sub(r'\.+', ' . ', essay_text)  # Replace multiple periods with a single period

# Remove numbers (optional: keep if needed)
essay_text = re.sub(r'\d+', '', essay_text)  # Removes all numeric characters

# Remove everything except alphanumeric characters and periods
essay_text = re.sub('[^a-zA-Z \.]', '', essay_text)

# Convert to lowercase
essay_text = essay_text.lower()

# Split the text into words
words = essay_text.split()

# Create a vocabulary of unique words (without the period '.')
unique_words = sorted(set(words) - {'.'})  # Remove period from the set of words

# Map words to integers (String to Integer mapping)
stoi = {w: i + 1 for i, w in enumerate(unique_words)}  # Start from 1 for words

# Assign the full stop ('.') to 0 as an end token
stoi['.'] = 0

# Reverse mapping: Integer to String
itos = {i: w for w, i in stoi.items()}

# Ensure that the period '.' is included in both mappings
itos[0] = '.'

# Test print a sample of the mappings
print("String to Integer mapping (stoi):", list(stoi.items())[:10])
print("Integer to String mapping (itos):", list(itos.items())[:10])


import torch.nn as nn



class NextWordMLP(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)  # Convert word indices to embeddings
        x = x.view(x.shape[0], -1)  # Flatten embeddings
        x = torch.relu(self.lin1(x))  # First layer + ReLU activation
        x = self.lin2(x)  # Second layer, output vocab_size logits
        return x


def predict_next_words(input_text, k):
    # Convert input text to indices and predict next words
    input_indices = [stoi[word] for word in input_text.split()[-block_size:]]
    input_tensor = torch.tensor(input_indices).unsqueeze(0)

    with torch.no_grad():
        predictions = model(input_tensor)
    top_k_indices = torch.topk(predictions[0, -1, :], k).indices
    return [itos[idx.item()] for idx in top_k_indices]




import streamlit as st

st.markdown("## Next k character prediction app")

# Sliders for context size and embedding dimension

d1 = st.sidebar.selectbox("Embedding Size", ["2", "64", "10"])
d2 = st.sidebar.selectbox("Context length", ["3", "5", "9"])
# d3 = st.sidebar.selectbox("Random state",["4000002","4000005","4000008"])
# Textboxes

t1 = st.sidebar.text_input("Input text", "")
t2 = st.sidebar.text_input("Number of Chars to predict", "")

emb={"2":0,"5":1,"10":2}
context={"3":0,"6":1,"9":2}
# Predict button
if st.button("Predict"):
    # Create a new model with the user-specified embedding
    model1 = NextWordMLP(int(d2),len(stoi), int(d1), 10).to(device)
    # model_number=emb[str(d1)]*3+context[str(d2)]
    model_number=0
    # Load the pre-trained weights into the new model
    model1.load_state_dict(torch.load(f"./model_{model_number}.pt",map_location=torch.device('cpu')), strict=False)
  
    model1.eval()
# Use the scripted model for prediction
    prediction = predict_next_words(model1,t1,int(t2),int(d2))
    st.write(prediction)
