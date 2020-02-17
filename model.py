import torch
import torch.nn as nn
import torchvision.models as models

tokens = {
    '<start>': 0,
    '<end>': 1,
    '<unk>': 2
}

is_debug = False

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        
        # Note: Freeze the pre-trained weights
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        
        if is_debug:
            print('Resnet modules: \n')
            print(modules)
        
        self.resnet = nn.Sequential(*modules)
        
        if is_debug:
            print(self.resnet)
            
        # Note: Due to the shape [2048 x 1 x 1], the batch size needs to be greater than 1 in order for
        # the BATCH normalization to work.
        self.batchnorm = nn.BatchNorm2d(resnet.fc.in_features)
        
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = self.batchnorm(features)
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
    
    def init_hidden(self, batch_size):
        self.hidden = (torch.zeros((self.n_layers, batch_size, self.hidden_size), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), torch.zeros((self.n_layers, batch_size, self.hidden_size), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    
    # Note: features shape: [batch_size x embed_size]
    # Note: captions shape: [batch_size x seq_len]
    # Note: lstm.__call__ takes in [batch_size x seq_len x embed_size]
    def forward(self, features, captions):
        batch_size = features.size(0)
        hidden = self.init_hidden(batch_size)
        
        # Note: Feed the image at t = -1, first
        #first_token, hidden = self.lstm(features.view(batch_size, 1, features.size(1)), self.hidden)
        features = features.view(batch_size, 1, features.size(1))
        
        # Note: We need to feed in the n - 1 data, as we are predicting the n data
        embeds = self.embedding(captions[:, :-1])
        
        combined = torch.cat((features, embeds), dim=1)
        
        # Note embeddings shape: [batch_size x seq_len x embed_size]
        #embeds = self.embedding(captions[:, :-1])
        
        # Note input: [batch_size x seq_len] -> [batch_size x seq_len x embed_size]
        # Note output shape: [batch_size x seq_len x hidden_size]
        output, hidden = self.lstm(combined, hidden)
        
        # Note output shape: [batch_size x seq_len x vocab_size]
        #return self.fc(torch.cat((first_token, output), dim=1))
        return self.fc(output)

    def sample(self, features, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        self.init_hidden(features.size(0))
        
        # Note: Predict the first word index
        output, hidden = self.lstm(features, self.hidden)
        probs = self.fc(output)
        
        # Note: Grab maximum index for most probable word
        values, max_i = torch.max(probs, dim=2)
        idx = max_i.item()
        captions = [idx]
        
        counter = 1
        while tokens['<end>'] != idx and counter <= max_len:
            embed = self.embedding(values.long())
            
            output, hidden = self.lstm(embed, hidden)
            
            # Note: Get next word index
            values, max_i = torch.max(output, dim=2)
            idx = max_i.item()
            captions.append(idx)
            
            counter += 1
        
        return captions
    
if __name__ == '__main__':
    encoder = EncoderCNN(embed_size=256)
    
    dummy_img = torch.zeros((10, 3, 224, 224))
    features = encoder(dummy_img)
    
    captions_len = 15
    embed_size = 256
    hidden_size = 512
    captions = torch.zeros(10, captions_len).long()
    
    decoder = DecoderRNN(embed_size, hidden_size, 200)
    
    print(features.size())
    print(captions.size())
    
    outputs = decoder(features, captions)
    