import pickle
import random
import json, os
from transformers import BertTokenizer
import numpy as np
def get_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_path, additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])
    return tokenizer


class data_sampler(object):

    def __init__(self, args, seed=None):
        self.set_path(args)
        self.args = args
        temp_name = [args.dataname, args.seed]
        file_name = "{}.pkl".format(
                    "-".join([str(x) for x in temp_name])
                )
        mid_dir = "sample_data/"
        if not os.path.exists(mid_dir):
            os.mkdir(mid_dir)
        for temp_p in ["_process_path"]:
            mid_dir = os.path.join(mid_dir, temp_p)
            if not os.path.exists(mid_dir):
                os.mkdir(mid_dir)
        self.save_data_path = os.path.join(mid_dir, file_name)

        self.tokenizer = get_tokenizer(args)

        # read relation data
        self.id2rel, self.rel2id = self._read_relations(args.relation_file)

        # regenerate data
        self.dataset_id2sample, self.dataset_rel2sample = self._read_data(self.args.data_file)

        # record relations
        self.seen_relations = []
        self.history_test_data = {}
    def set_path(self, args):
        use_marker = ""
        if args.dataname in ['FewRel']:
            args.data_file = os.path.join(args.data_path,"data_with{}_marker_sample.json".format(use_marker))
            args.relation_file = os.path.join(args.data_path, "id2rel_fewrel.json")
        elif args.dataname in ['TACRED']:
            args.data_file = os.path.join(args.data_path,"data_with{}_marker_tacred_sample.json".format(use_marker))
            args.relation_file = os.path.join(args.data_path, "id2rel_tacred.json")

    def set_seed(self, seed):
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)

    def __iter__(self):
        return self

    def __next__(self):

        return self.dataset_id2sample, self.dataset_rel2sample

    def _read_data(self, file):
        if os.path.isfile(self.save_data_path):
            with open(self.save_data_path, 'rb') as f:
                dataset_id2sample, dataset_rel2sample = pickle.load(f)
            return dataset_id2sample, dataset_rel2sample
        else:
            data = json.load(open(file, 'r', encoding='utf-8'))
            dataset_id2sample = [[] for i in range(self.args.num_of_relation)]
            dataset_rel2sample = {}
            for relation in data.keys():
                rel_samples = data[relation]
                for i, sample in enumerate(rel_samples):
                    tokenized_sample = {}
                    tokenized_sample['relation'] = self.rel2id[sample['relation']]
                    tokenized_sample['tokens'] = self.tokenizer.encode(' '.join(sample['tokens']),
                                                                    padding='max_length',
                                                                    truncation=True,
                                                                    max_length=self.args.max_length)

                    dataset_id2sample[self.rel2id[relation]].append(tokenized_sample)
                    if relation in dataset_rel2sample:
                        dataset_rel2sample[relation].append(tokenized_sample)
                    else:
                        dataset_rel2sample[relation] = []
                        dataset_rel2sample[relation].append(tokenized_sample)
                    
                    
            with open(self.save_data_path, 'wb') as f:
                pickle.dump((dataset_id2sample, dataset_rel2sample), f)
            return dataset_id2sample, dataset_rel2sample

    def _read_relations(self, file):
        id2rel = json.load(open(file, 'r', encoding='utf-8'))
        rel2id = {}
        for i, x in enumerate(id2rel):
            rel2id[x] = i
        return id2rel, rel2id