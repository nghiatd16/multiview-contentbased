from dataset import MOSTPortal
import random
import json
dataset = MOSTPortal("QN_fixed_posts.csv", "QN_fixed_trans.csv")
train_test_ratio = 0.9

users = dataset.users
num_train_users = int(train_test_ratio * len(users))
all_train_users = set(random.sample(users, num_train_users))
all_test_users = users - all_train_users
info_dict = {
    "train_users": list(all_train_users),
    "test_users": list(all_test_users)
}
with open("train_test_set.json", "w") as fout:
    json.dump(info_dict, fout)