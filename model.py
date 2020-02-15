import torch
import torch.nn as nn
import torchvision.models as models

is_debug = False

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        
        if is_debug:
            print('Resnet modules: \n')
            print(modules)
        
        self.resnet = nn.Sequential(*modules)
        
        if is_debug:
            print(self.resnet)
        
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

# Note: num_features = input_size
# input shape: [batch_size x seq_len x input_size] if batch_first or [seq_len x batch_size x input_size]
# output shape: [batch_size x seq_len x hidden_size] if batch_first or [seq_len x batch_size x hidden_size]
# seq_len = caption_len
# h0 shape: [num_layers x batch_size x hidden_size]
# c0 shape: [num_layers x batch_size x hidden_size]

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, batch_first=True):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_layers = num_layers
        
        self.init_hidden()
        
        # Note: Embed into a one-hot vector of size vocab_size (probs of each index being the most likely)
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_size
        )
        
        self.lstm = nn.LSTM(
            input_size=self.embed_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.n_layers, 
            batch_first=batch_first)
        
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
    
    def init_hidden(self):
        self.h0_c0 = (torch.zeros(self.n_layers, 1, self.hidden_size), torch.zeros(self.n_layers, 1, self.hidden_size))
    
    # Note: features shape: [batch_size x embed_size]
    # Note: captions shape: [batch_size x seq_len]
    def forward(self, features, captions):
        output, hidden = self.lstm(features.view(features.size(0), 1, features.size(1)))
        
        # Note embeddings shape: [batch_size x seq_len x embed_size]
        embeddings = self.embedding(captions)
        
        # Note output shape: [batch_size x seq_len x hidden_size]
        output, hidden = self.lstm(embeddings, hidden)
        
        # Note output shape: [batch_size x seq_len x vocab_size]
        return self.fc(output)

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass
    
if __name__ == '__main__':
    encoder = EncoderCNN(embed_size=256)
    
    dummy_img = torch.zeros((1, 3, 224, 224))
    features = encoder(dummy_img)
    