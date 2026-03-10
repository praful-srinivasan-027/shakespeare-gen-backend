import torch
import torch.nn as nn
vocab = ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_to_id={char:index for index,char in enumerate(vocab)} #mapping each character to a its respective array index
id_to_char={index:char for index,char in enumerate(vocab)} #mapping each array index to its respective character
#Encoder and Decoder functions
def encoder(text):
    return [char_to_id[char] for char in text.lower()]
def decoder(encoded_text):
    return "".join([id_to_char[id.item()] for id in encoded_text])
window_length=100
batch_size=64
device="cuda" if torch.cuda.is_available() else "cpu"
model_weights = torch.load("checkpoint.pth", map_location=torch.device('cpu'))
class CharRNN(nn.Module):
    def __init__(self,vocab_size,embedding_dim=10,hidden_dim=100,num_layers=2,dropout=0.2):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.GRU=nn.GRU(embedding_dim,hidden_dim,num_layers,dropout=dropout,batch_first=True)
        self.linear=nn.Linear(hidden_dim,vocab_size)
    def forward(self,x):
        X=self.embedding(x)
        output,_=self.GRU(X)
        output=self.linear(output).permute(0,2,1)
        return output
torch.manual_seed(42)
vocab_size=len(vocab)
model=CharRNN(vocab_size).to(device)
model.load_state_dict(model_weights["model_state_dict"])
torch.seed()
def get_char(model, text, temperature=0.5):
    encoded_text=torch.tensor(encoder(text)).unsqueeze(0).to(device)
    with torch.inference_mode():
        y_logits=model(encoded_text)
        prob=torch.softmax(y_logits[:,:,-1]/temperature,dim=1)
        char_id=torch.multinomial(prob,num_samples=1)
    return decoder(char_id)
def generate_text(text,num_chars,temperature=0.5):
    for i in range(num_chars):
        text+=get_char(model,text,temperature)
    return text