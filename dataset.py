import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import reduce
from vncorenlp import VnCoreNLP
import itertools
import json

class MOSTPortal:
    def __init__(self, post_csv, transaction_csv, train_test_set=None, is_testing=False):
        self.post_df = pd.read_csv(post_csv).fillna("")
        self.transaction_df = pd.read_csv(transaction_csv)
        self.users = set(self.transaction_df['user_id'].unique().tolist())
        self.posts_id = set(self.post_df['post_id'].unique().tolist())
        self.trans_groupby_user = self.transaction_df.groupby(['user_id'])
        if train_test_set is not None:
            with open(train_test_set) as fin:
                train_test_set = json.load(fin)
            if is_testing == False:
                self.users = self.users - set(train_test_set['test_users']) # Training mode. Remove test_users
            else:
                self.users = self.users - set(train_test_set['train_users']) # Testing mode. Remove train_users
        # self.posts_users = 
    
    def get_user_clicked_items(self, user_id):
        return self.trans_groupby_user.get_group(user_id)['post_id'].unique().tolist()

    def get_postneg_by_userid(self, user_id, n_sample=4):
        if n_sample == 0:
            return list()
        user_clicked_posts = self.get_user_clicked_items(user_id)
        return self.get_postneg_by_clicked_items(user_clicked_posts, n_sample)
    
    def get_postneg_by_clicked_items(self, user_clicked_posts, n_sample=4):
        if n_sample == 0:
            return list()
        return self.post_df[~self.post_df['post_id'].isin(user_clicked_posts)].sample(n=n_sample)['post_id'].tolist()
    
    def _norm_dataset(self):
        idx2utt = dict()
        utt2idx = dict()
        for index, row in self.post_df.iterrows():
            post_id = row['post_id']
            utt2idx[post_id] = index
            idx2utt[index] = post_id
        for index, row in self.transaction_df.iterrows():
            if row['post_id'] not in utt2idx:
                self.transaction_df.drop(index, inplace=True)
                continue
            # row['post_id'] = utt2idx[row['post_id']]
            self.transaction_df.at[index, 'post_id']= utt2idx[row['post_id']]
        new_post_df = self.post_df.drop("post_id", 1)
        new_post_df.to_csv("QN_fixed_post.csv")
        self.transaction_df.to_csv("QN_fixed_trans.csv")
        # print(self.transaction_df)


# annotator = VnCoreNLP(address="http://0.0.0.0", port=9090)

# def process_sentence(sentence):
#     # print(sentence)
#     sentence = sentence.replace("\\t", " ")
#     sentence = sentence.replace("\\n", " ")
#     # print(annotator.tokenize(sentence.lower()))
#     sentences = annotator.tokenize(sentence.lower())
    # return list(itertools.chain(*sentences))

if __name__ == "__main__":
    # dataset = AdressaDataset_v2("ad_posts.csv", "ad_trans.csv")
    # print(dataset.post_df)
    # for index, row in dataset.df.iterrows():
    #     print(row)
    #     break
    dataset = MOSTPortal("/home/nghiatd/workspace/Content-based/torch-multiview/quang-nam/post_QNPortal_all.csv", "/home/nghiatd/workspace/Content-based/torch-multiview/quang-nam/transaction_QNPortal_all.csv")
    # posts = dataset.post_df
    # max_length_title = 0
    # for index, row in posts.iterrows():
    #     max_length_title = max(max_length_title, len(row['title']))
    # print(max_length_title)
    dataset._norm_dataset()