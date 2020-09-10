import json
import numpy as np
from vncorenlp import VnCoreNLP
import itertools
import nltk
from dataset import MOSTPortal


annotator = VnCoreNLP("/home/nghiatd/workspace/Content-based/chuhan_wu/IJCAI2019-NAML/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g') 
# annotator = VnCoreNLP(address="http://0.0.0.0", port=9090)
print("Connected to VnCoreNLP")
# with open("word_dict.json", "r") as fin:
#     word_dict = json.load(fin)
MAX_SENT_LENGTH=30
MAX_SENTS=10
MAX_BODY_LENGTH=300

def process_sentence(sentence, max_length, language="vietnamese"):
    sentence = sentence[:2000]
    sentence = sentence.replace("\\t", " ")
    sentence = sentence.replace("\\n", " ")
    sentence = sentence.replace("\xa0", " ")
    sentence = sentence.replace(")", " ")
    sentence = sentence.replace("(", " ")
    sentence = sentence.replace("{", " ")
    sentence = sentence.replace("}", " ")
    sentence = sentence.replace("[", " ")
    sentence = sentence.replace("]", " ")
    sentence = sentence.replace("”", " ")
    sentence = sentence.replace("“", " ")
    sentence = sentence.replace("'", " ")
    sentence = sentence.replace(";", " ")
    sentence = sentence.replace(":", " ")
    sentence = sentence.replace(",", " ")
    sentence = sentence.replace(".", " ")
    sentence = sentence.replace("-", " ")
    if language == "vietnamese":
        sentences = annotator.tokenize(sentence.lower())
        return list(itertools.chain(*sentences))[:max_length]
    else:
        return nltk.word_tokenize(sentence, language=language)[:max_length]
    return None
def preprocess_news_encoder(titles, bodies, language="vietnamese"):
    assert len(titles) == len(bodies), "titles and bodies must have same size"
    news_words=[]
    list_news = list()

    for title, body in zip(titles, bodies):
        sample = [process_sentence(title, MAX_SENT_LENGTH, language),process_sentence(body, MAX_BODY_LENGTH, language)]
        list_news.append(sample)
    
    for news in list_news:
        word_id=[]
        for word in news[0]:
            if word in word_dict:
                word_id.append(word_dict[word][0])
        word_id=word_id[:30]
        news_words.append(word_id+[0]*(30-len(word_id)))

    news_words=np.array(news_words,dtype='int32')

    news_body=[]
    for news in list_news:
        word_id=[]
        for word in news[1]:
            if word in word_dict:
                word_id.append(word_dict[word][0])
        word_id=word_id[:300]
        news_body.append(word_id+[0]*(300-len(word_id)))
    news_body=np.array(news_body,dtype='int32')
    candidate = news_words
    candidate_body = news_body

    return [candidate] + [candidate_body]

def preprocess_user_presentation_model(browsed_titles, browsed_bodies, language="vietnamese"):
    list_browsed = list()
    for title, body in zip(browsed_titles, browsed_bodies):
        sample = [process_sentence(title, MAX_SENT_LENGTH, language),process_sentence(body, MAX_BODY_LENGTH, language)]
        list_browsed.append(sample)
    browsed_news = []

    for news in list_browsed:
        word_id=[]
        for word in news[0]:
            if word in word_dict:
                word_id.append(word_dict[word][0])
        word_id=word_id[:30]
        browsed_news.append(word_id+[0]*(30-len(word_id)))
    browsed_news = browsed_news[:MAX_SENTS]
    if len(browsed_news) < MAX_SENTS:
        padding = [[0] * 30] * (MAX_SENTS - len(browsed_news))
        browsed_news.extend(padding)
    browsed_news=np.array(browsed_news,dtype='int32')
    browsed_news = np.expand_dims(browsed_news, axis=0)
    
    browsed_news_split=[browsed_news[:,k,:] for k in range(browsed_news.shape[1])]
    browsed_body=[]
    for news in list_browsed:
        word_id=[]
        for word in news[1]:
            if word in word_dict:
                word_id.append(word_dict[word][0])
        word_id=word_id[:300]
        browsed_body.append(word_id+[0]*(300-len(word_id)))
    if len(browsed_body) < MAX_SENTS:
        padding = [[0] * 300] * (MAX_SENTS - len(browsed_body))
        browsed_body.extend(padding)
    browsed_body=np.array(browsed_body,dtype='int32')
    browsed_body = np.expand_dims(browsed_body, axis=0)
    
    browsed_news_body_split=[browsed_body[:,k,:] for k in range(browsed_body.shape[1])]
    return browsed_news_split + browsed_news_body_split

def preprocess_training(dataset: MOSTPortal, desdir: str):
    import os
    import tqdm
    all_post = dataset.posts_id
    for postid in tqdm.tqdm(all_post):
        row = dataset.post_df.loc[postid, ['title', 'body']]
        title = row['title']
        body = row['body']
        preprocessed_title = process_sentence(title, MAX_SENT_LENGTH, "vietnamese")
        # preprocessed_body = process_sentence(body, MAX_BODY_LENGTH, "vietnamese")
        info = {
            "preprocessed_title": preprocessed_title,
            "preprocessed_body": preprocessed_title
        }
        despath = os.path.join(desdir, f"{postid}.json")
        with open(despath, "w") as fout:
            json.dump(info, fout)
        # print(title)
        # print(body)
        # break
        
        
if __name__ == "__main__":
    dataset = MOSTPortal("QN_fixed_posts.csv", "QN_fixed_trans.csv")
    preprocess_training(dataset, "/home/nghiatd/workspace/Content-based/ws/QNPortal/posts")
