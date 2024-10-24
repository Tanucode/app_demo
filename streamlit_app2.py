
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
