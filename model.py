import torch.nn as nn
import torch.nn.functional as F
import torch
import json
import constants as c



class AttentiveMultiView(nn.Module):
    def __init__(self, word_dict_length):
        super(AttentiveMultiView, self).__init__()
        self.encoder_embedding = nn.Embedding(num_embeddings=word_dict_length, embedding_dim=300)
        self.encoder_conv = nn.Conv1d(300, 400, kernel_size=3)
        self.encoder_cnn_padding = nn.ConstantPad1d(1, 0)

        self.encoder_attention = nn.Linear(400, 200)
        self.encoder_attention_weight = nn.Linear(200, 1)

        self.browsed_news_attention = nn.Linear(400, 200)
        self.browsed_news_attention_weight = nn.Linear(200, 1)
        

    def forward_encoder(self, x):
        embedded_sequences_title = F.dropout(self.encoder_embedding(x), p=0.2, training=self.training)
        embedded_sequences_title = embedded_sequences_title.view(-1, 300, c.MAX_SENT_LENGTH)
        
        title_cnn = self.encoder_conv(embedded_sequences_title)
        padded_title_cnn = self.encoder_cnn_padding(title_cnn)
        
        padded_title_cnn = padded_title_cnn.view(-1, c.MAX_SENT_LENGTH, 400)
        attention_layer1 = torch.tanh(self.encoder_attention(padded_title_cnn))
        attention_layer2 = self.encoder_attention_weight(attention_layer1)
        
        
        attention_weight = F.softmax(attention_layer2, dim=1)
        padded_title_cnn = padded_title_cnn.view(-1, 400, c.MAX_SENT_LENGTH)
        news_embedding = torch.matmul(padded_title_cnn,attention_weight).view(-1, 400)
        return news_embedding
    
    def forward_browsed_news(self, browsed_news_input):
        browsednews = [self.forward_encoder(browsed_news_input[i]) for i in range(c.MAX_SENTS)]
        browsednewsrep = torch.cat([torch.unsqueeze(news,dim=1) for news in browsednews],dim=1)
        attentionn_layer1 = torch.tanh(self.browsed_news_attention(browsednewsrep))
        attention_layer2 = self.browsed_news_attention_weight(attentionn_layer1)
        attention_weight = F.softmax(attention_layer2, dim=1)
        browsednewsrep = browsednewsrep.view(-1, 400, c.MAX_SENTS)
        user_embedding = torch.matmul(browsednewsrep,attention_weight).view(-1, 400)
        return user_embedding

    def __training_forward(self, browsed_news_input, candidate_news):
        user_embedding = self.forward_browsed_news(browsed_news_input)
        candidate_vecs = [self.forward_encoder(candidate_news[_]) for _ in range(1+c.npratio)]
        logits = [torch.mul(user_embedding, candidate_vec).sum(dim=1).view(-1,1) for candidate_vec in candidate_vecs]
        logits = F.softmax(torch.cat(logits, dim=1), dim=1)
        return logits


    def forward(self, browsed_news, candidate_news):
        return self.__training_forward(browsed_news, candidate_news)

if __name__ == "__main__":
    with open("word_dict.json", "r") as fin:
        word_dict = json.load(fin)
    model = AttentiveMultiView(len(word_dict))
    input_dict = {
        "browsed_news": torch.zeros([c.MAX_SENTS, 3, c.MAX_SENT_LENGTH]).long(),
        "candidate_news": torch.zeros([c.npratio+1, 3, c.MAX_SENT_LENGTH]).long()
    }
    # torch.save(model, "test_model.pt")
    
    print(model(input_dict["browsed_news"], input_dict["candidate_news"]).shape)