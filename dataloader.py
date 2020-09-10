import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import random
import glob
import os
import copy
import json
import constants as c
class TripletDataGenerator(Dataset):
    'Generates data for Keras'
    def __init__(self, dataset, preprocessed_dir, word_dict=None, batch_size=c.MINIBATCH_SIZE, npratio=c.npratio, is_testing=False):
        'Initialization'
        self.dataset = dataset
        self.preprocessed_dir = preprocessed_dir
        self.all_files = glob.glob(os.path.join(preprocessed_dir, "*.json"))
        self.word_dict = word_dict
        if self.word_dict is None:
            self.word_dict = self.__process_word_dict()
        
        self.users = list(dataset.users)
        self.num_users = len(self.users)
        self.batch_size = batch_size
        self.training_browsed_items = list()
        self.training_positive_items = list()
        for userid in self.users:
            list_browsed_items, list_pos_items = self.__generate_training_samples(userid)
            if list_browsed_items is not None and list_pos_items is not None:
                self.training_browsed_items.extend(list_browsed_items)
                self.training_positive_items.extend(list_pos_items)
        self.datasize = len(self.training_browsed_items)
        self.npratio = npratio
        self.is_testing = is_testing
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.training_browsed_items)//self.batch_size)

    def __process_word_dict(self):
        import tqdm
        word_dict_raw={'PADDING':[0,999999]}
    
        print("Building word_dict")
        
        for path in tqdm.tqdm(self.all_files):
            with open(path) as fin:
                data = json.load(fin)
                title = data['preprocessed_title']
                body = data['preprocessed_body']
                for word in title:
                    if word in word_dict_raw:
                        word_dict_raw[word][1]+=1
                    else:
                        word_dict_raw[word]=[len(word_dict_raw),1]
                for word in body:
                    if word in word_dict_raw:
                        word_dict_raw[word][1]+=1
                    else:
                        word_dict_raw[word]=[len(word_dict_raw),1]
        word_dict={}
        for i in word_dict_raw:
            if word_dict_raw[i][1]>=3:
                word_dict[i]=[len(word_dict),word_dict_raw[i][1]]
        return word_dict
    
    def __generate_training_samples(self, userid):
        user_clicked_items = self.dataset.get_user_clicked_items(userid)
        list_browsed_items = list()
        list_pos_items = list()
        if len(user_clicked_items) <= 1:
            return None, None
        if(len(user_clicked_items) == 2):
            list_browsed_items.append(user_clicked_items[:-1])
            list_pos_items.append(user_clicked_items[-1:])
        else:
            n = len(user_clicked_items)
            for i in range(1, n):
                list_browsed_items.append(user_clicked_items[:i])
                list_pos_items.append(user_clicked_items[i:])
        return list_browsed_items, list_pos_items

    def __getitem__(self, index):

        'Generate one batch of data'
        if torch.is_tensor(index):
            index = index.tolist()
        # Generate indexes of the batch
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        browsed_items_indexes = [self.training_browsed_items[k] for k in indexes]
        positive_items_indexes = [self.training_positive_items[k] for k in indexes]

        batch_browsed_title = []
        batch_browsed_body = []
        batch_candidate_title = []
        batch_candidate_body = []

        batch_labels = []

        for idx in range(len(browsed_items_indexes)):
            list_browsed_items = browsed_items_indexes[idx]
            list_pos_items = [positive_items_indexes[idx][0]]
            clicked_items = copy.copy(list_browsed_items)
            clicked_items.extend(list_pos_items)
            list_neg_items = self.dataset.get_postneg_by_clicked_items(clicked_items, self.npratio)

            list_candidate_items = list_pos_items
            labels = [1] * len(list_pos_items)
            list_candidate_items.extend(list_neg_items)
            labels.extend([0] * self.npratio)

            # Shuffle candidate
            shuffle_index = np.arange(0, len(labels)).tolist()
            random.shuffle(shuffle_index)
            shuffle_candidate_items = [list_candidate_items[k] for k in shuffle_index]
            shuffle_labels = [labels[k] for k in shuffle_index]

            browsed_title = self._browsed_data_gen(list_browsed_items)
            candidate_title = self._candidate_data_gen(shuffle_candidate_items)
            

            # Add to batch
            batch_browsed_title.append(browsed_title)
            batch_candidate_title.append(candidate_title)

            batch_labels.append(shuffle_labels)

        batch_browsed_title = np.stack(batch_browsed_title)
        browsed_title_split=[batch_browsed_title[:,k,:] for k in range(batch_browsed_title.shape[1])]
        
        batch_candidate_title = np.stack(batch_candidate_title)
        candidate_split=[batch_candidate_title[:,k,:] for k in range(batch_candidate_title.shape[1])]
        
        batch_labels = np.array(batch_labels, dtype='int32')

        # Debug shape
        
        return (torch.tensor(browsed_title_split).long(), torch.tensor(candidate_split).long(), batch_labels)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(0, self.datasize).tolist()
        random.shuffle(self.indexes)

    def _browsed_data_gen(self, list_clicked_items_id):
        list_browsed = list()

        # Load preprocessed titles and bodies from json
        for itemid in list_clicked_items_id:
            filename = os.path.join(self.preprocessed_dir, f"{itemid}.json")
            with open(filename) as fin:
                data = json.load(fin)
            title = data['preprocessed_title']
            list_browsed.append(title)

        browsed_news = []

        for news in list_browsed:
            word_id=[]
            list_preprocessed_titles = news
            for word in list_preprocessed_titles:
                if word in self.word_dict:
                    word_id.append(self.word_dict[word][0])
            word_id=word_id[:30]
            browsed_news.append(word_id+[0]*(30-len(word_id)))
        if len(browsed_news) < c.MAX_SENTS:
            padding = [[0] * 30] * (c.MAX_SENTS - len(browsed_news))
            browsed_news.extend(padding)
        else:
            browsed_news = browsed_news[-c.MAX_SENTS:]
        browsed_news=np.array(browsed_news,dtype='int32')
        return browsed_news
    
    def _candidate_data_gen(self, list_candidate_items_id):
        list_candidate = list()

        # Load preprocessed titles and bodies from json
        for itemid in list_candidate_items_id:
            filename = os.path.join(self.preprocessed_dir, f"{itemid}.json")
            with open(filename) as fin:
                data = json.load(fin)
            title = data['preprocessed_title']
            list_candidate.append(title)

        candidate_news = []

        for news in list_candidate:
            word_id=[]
            list_preprocessed_titles = news
            for word in list_preprocessed_titles:
                if word in self.word_dict:
                    word_id.append(self.word_dict[word][0])
            word_id=word_id[:30]
            candidate_news.append(word_id+[0]*(30-len(word_id)))

        candidate_news=np.array(candidate_news,dtype='int32')

        return candidate_news

class EvaluateDataGenerator(Dataset):
    'Generates data for Keras'
    def __init__(self, dataset, preprocessed_dir, word_dict=None, batch_size=c.MINIBATCH_SIZE):
        'Initialization'
        self.dataset = dataset
        self.preprocessed_dir = preprocessed_dir
        self.all_files = glob.glob(os.path.join(preprocessed_dir, "*.json"))
        self.word_dict = word_dict
        if self.word_dict is None:
            self.word_dict = self.__process_word_dict()
        
        self.users = list(dataset.users)
        self.num_users = len(self.users) 
        self.batch_size = batch_size
        self.datasize = len(self.all_files)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.users)//self.batch_size)

    def generate_samples(self, userid):
        user_clicked_items = self.dataset.get_user_clicked_items(userid)
        list_browsed_items = list()
        list_pos_items = list()
        if(len(user_clicked_items) < 3):
            list_browsed_items.append(user_clicked_items[:-1])
            list_pos_items.append(user_clicked_items[-1:])
        else:
            n = len(user_clicked_items)
            for i in range(1, n):
                list_browsed_items.append(user_clicked_items[:i])
                list_pos_items.append(user_clicked_items[i:])
        return list_browsed_items, list_pos_items

    def _browsed_data_gen(self, list_clicked_items_id):
        list_browsed = list()

        # Load preprocessed titles and bodies from json
        for itemid in list_clicked_items_id:
            filename = os.path.join(self.preprocessed_dir, f"{itemid}.json")
            with open(filename) as fin:
                data = json.load(fin)
            title = data['preprocessed_title']
            list_browsed.append(title)

        browsed_news = []

        for news in list_browsed:
            word_id=[]
            list_preprocessed_titles = news
            for word in list_preprocessed_titles:
                if word in self.word_dict:
                    word_id.append(self.word_dict[word][0])
            word_id=word_id[:30]
            browsed_news.append(word_id+[0]*(30-len(word_id)))
        if len(browsed_news) < c.MAX_SENTS:
            padding = [[0] * 30] * (c.MAX_SENTS - len(browsed_news))
            browsed_news.extend(padding)
        else:
            browsed_news = browsed_news[-c.MAX_SENTS:]
        browsed_news=np.array(browsed_news,dtype='int32')
        # browsed_title_split=[browsed_news[:,k,:] for k in range(browsed_news.shape[1])]

        return torch.tensor(browsed_news).long()
    
    def _candidate_data_gen(self, list_candidate_items_id):
        list_candidate = list()

        # Load preprocessed titles and bodies from json
        for itemid in list_candidate_items_id:
            filename = os.path.join(self.preprocessed_dir, f"{itemid}.json")
            with open(filename) as fin:
                data = json.load(fin)
            title = data['preprocessed_title']
            list_candidate.append(title)

        candidate_news = []

        for news in list_candidate:
            word_id=[]
            list_preprocessed_titles = news
            for word in list_preprocessed_titles:
                if word in self.word_dict:
                    word_id.append(self.word_dict[word][0])
            word_id=word_id[:30]
            candidate_news.append(word_id+[0]*(30-len(word_id)))

        candidate_news=np.array([candidate_news],dtype='int32')

        return torch.tensor(candidate_news).long()