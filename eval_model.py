import json
import numpy as np
import torch
from model import AttentiveMultiView
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from dataset import MOSTPortal
from dataloader import EvaluateDataGenerator
TRAIN_TEST_RATIO = 0.8
LANGUAGE = "vietnamese"

with open("word_dict.json", "r") as fin:
    word_dict = json.load(fin)
# word_dict=None
# Load model
model = torch.load("./checkpoints/50.pt")['model']
# model.eval()
dataset = MOSTPortal("QN_fixed_posts.csv", "QN_fixed_trans.csv", train_test_set="train_test_set.json", is_testing=True)
test_generator = EvaluateDataGenerator(dataset, "/home/nghiatd/workspace/Content-based/ws/QNPortal/posts", word_dict=word_dict)
embedding_items = list()
utt2id = dict()
id2utt = dict()
processed_news = list()

print("Encoding news")
with tqdm(total=len(dataset.post_df)) as pbar:
    for index, row in dataset.post_df.iterrows():
        post_id = row['post_id']
        title = row['title']
        candidate_titles = test_generator._candidate_data_gen([post_id])
        embedding_vec = model.forward_encoder(candidate_titles.to("cuda"))[0]
        
        embedding_items.append(embedding_vec.cpu().detach().numpy())
        
        utt2id[index] = post_id
        id2utt[post_id] = index
        pbar.update()
    pbar.close()


embedding_items = np.array(embedding_items, dtype=np.float32)
users_dict = dict()

print("calc user presentation")



for index, userid in enumerate(tqdm(dataset.users)):
    user_clicked_items = dataset.get_user_clicked_items(userid)

    num_test = 1
    num_train = len(user_clicked_items) - num_test
    list_browsed_items_id = user_clicked_items[:-1]
    list_test_items_id = user_clicked_items[-1:]
    browsed_title = test_generator._browsed_data_gen(list_browsed_items_id)

    # print(browsed_title.shape)
    
    preprocessed_browsed = browsed_title
    user_rep = model.forward_browsed_news(preprocessed_browsed.to("cuda"))[0]
    users_dict[userid] = {
        "presentation": user_rep.cpu().detach().numpy(),
        "test_items": set(list_test_items_id) 
    }


# exit(0)
print("Evaluating")
total = 0

mrr20 = 0
mrr10 = 0
mrr5 = 0
TP20 = 0
TP10 = 0
TP5 = 0
for userid in tqdm(dataset.users):
    
    score_matrix = np.matmul(users_dict[userid]['presentation'], embedding_items.T)
    
    ranked_similarity_score = score_matrix.argsort()[-20:][::-1]
    test_items_id = users_dict[userid]['test_items']
    rec_items_id = [utt2id[utt] for utt in ranked_similarity_score]
    for index, itemid in enumerate(rec_items_id):
        # Find rank
        if itemid in test_items_id:
            if index < 20:
                TP20 += 1
                mrr20 += (1/(index+1))
            if index < 10:
                TP10 += 1
                mrr10 += (1/(index+1))
            if index < 5:
                TP5 += 1
                mrr5 += (1/(index+1))
    total += 1

print("recal@5:{}/{} ~ {}".format(TP5, total, TP5/total))
print("recal@10:{}/{} ~ {}".format(TP10, total, TP10/total))
print("recal@20:{}/{} ~ {}".format(TP20, total, TP20/total))
print("MRR@5:{}".format(mrr5/total))
print("MRR@10:{}".format(mrr10/total))
print("MRR@20:{}".format(mrr20/total))