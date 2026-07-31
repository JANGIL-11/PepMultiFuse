import torch
import math
import random
import numpy as np
import os
import pandas as pd
from transformers import AutoTokenizer


def read_file(file, tokenizer, do_reverse=False):
    input_ids, labels = [], []
    with open(file, 'r') as f:
        lines = f.readlines()
        line_num = len(lines)
        for i in range(1, line_num):
            line = lines[i]
            seq, label = line.split()

            seq = seq.upper()
            seq = " ".join(seq)

            label = list(label)
            label = [int(_) for _ in label]

            input_ids.append(tokenizer(seq).input_ids)
            labels.append(label)

        f.close()
    return input_ids, labels


def process(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False)
    train_input_ids, train_labels = read_file(args.train_file, tokenizer, args.do_reverse)
    test_input_ids, test_labels = read_file(args.test_file, tokenizer)

    dataset_num = args.dataset
    feature_dir = f"features/Dataset{dataset_num}"


    def load_feat(split):
        return {
            "ss_aa_63": np.load(os.path.join(feature_dir, f"Dataset{dataset_num}_{split}_ss_aa_63.npy"),
                                allow_pickle=True),
            "donor": np.load(os.path.join(feature_dir, f"Dataset{dataset_num}_{split}_hbond_donor.npy"),
                             allow_pickle=True),
            "acceptor": np.load(os.path.join(feature_dir, f"Dataset{dataset_num}_{split}_hbond_acceptor.npy"),
                                allow_pickle=True),
            "disorder": np.load(os.path.join(feature_dir, f"Dataset{dataset_num}_{split}_disorder_std.npy"),
                                allow_pickle=True),
            "pp": np.load(os.path.join(feature_dir, f"Dataset{dataset_num}_{split}_physicochemical_std.npy"),
                          allow_pickle=True),
            "rsa": np.load(os.path.join(feature_dir, f"Dataset{dataset_num}_{split}_rsa.npy"), allow_pickle=True),
        }


    def load_spatial_features(split):
        spatial_dir = f"data/spatial_tripeptide_features/Dataset{dataset_num}/{split}"
        if not os.path.exists(spatial_dir):
            return None

        esm_files = sorted([f for f in os.listdir(spatial_dir) if f.endswith("_spatial_tri_esm.npy")])
        mask_files = sorted([f for f in os.listdir(spatial_dir) if f.endswith("_spatial_tri_mask.npy")])

        if len(esm_files) == 0:
            return None

        esm_data = [np.load(os.path.join(spatial_dir, f)) for f in esm_files]
        mask_data = [np.load(os.path.join(spatial_dir, f)) for f in mask_files]

        return {
            "esm": esm_data,
            "mask": mask_data,
            "files": esm_files,
        }

    train_feats = load_feat("train")
    test_feats = load_feat("test")
    train_spatial = load_spatial_features("train")
    test_spatial = load_spatial_features("test")

    raw_datasets = {
        "train_input_ids": train_input_ids,
        "train_labels": train_labels,
        "test_input_ids": test_input_ids,
        "test_labels": test_labels,
        "train_feats": train_feats,
        "test_feats": test_feats,
        "train_spatial": train_spatial,
        "test_spatial": test_spatial,
    }
    return raw_datasets


class DataIterator(object):
    def __init__(self, args, input_ids, labels, feats=None, spatial_features=None):
        self.input_ids = input_ids
        self.labels = labels
        self.feats = feats
        self.spatial_features = spatial_features
        self.device = torch.device(args.device)
        self.sample_num = len(input_ids)
        self.batch_size = args.batch_size
        self.batch_count = math.ceil(len(input_ids) / self.batch_size)

    def _load_spatial(self, index):

        if self.spatial_features is None:
            return None, None
        if index >= len(self.spatial_features["esm"]):
            return None, None
        esm = self.spatial_features["esm"][index]
        mask = self.spatial_features["mask"][index]
        return esm, mask

    def get_index(self, index):
        input_ids = torch.tensor(self.input_ids[index]).unsqueeze(0).to(self.device)
        labels = torch.tensor(self.labels[index]).unsqueeze(0).to(self.device)


        spatial_esm, spatial_mask = self._load_spatial(index)
        if spatial_esm is not None:
            spatial_esm = torch.tensor(spatial_esm).unsqueeze(0).to(self.device)
        if spatial_mask is not None:
            spatial_mask = torch.tensor(spatial_mask).unsqueeze(0).to(self.device)

        if self.feats is not None:
            feats = {}
            for key in ["ss_aa_63", "donor", "acceptor", "disorder", "pp", "rsa"]:
                data = self.feats[key][index]
                if data is not None:
                    feats[key] = torch.tensor(data).unsqueeze(0).to(self.device)
                else:
                    feats[key] = None
            return input_ids, labels, feats, spatial_esm, spatial_mask

        return input_ids, labels, spatial_esm, spatial_mask

    def shuffle(self):
        indices = [i for i in range(self.sample_num)]
        random.shuffle(indices)
        self.input_ids = [self.input_ids[_] for _ in indices]
        self.labels = [self.labels[_] for _ in indices]
        if self.feats is not None:
            for key in self.feats:
                self.feats[key] = [self.feats[key][_] for _ in indices]
        if self.spatial_features is not None:
            self.spatial_features["esm"] = [self.spatial_features["esm"][_] for _ in indices]
            self.spatial_features["mask"] = [self.spatial_features["mask"][_] for _ in indices]

    def get_batch(self, index):
        input_ids = []
        labels = []
        spatial_esm_list = []
        spatial_mask_list = []
        feats_batch = {key: [] for key in ["ss_aa_63", "donor", "acceptor", "disorder", "pp", "rsa"]}

        for i in range(index * self.batch_size,
                       min((index + 1) * self.batch_size, len(self.input_ids))):
            input_ids.append(self.input_ids[i])
            labels.append(self.labels[i])

            esm, mask = self._load_spatial(i)
            if esm is not None and mask is not None:
                spatial_esm_list.append(esm)
                spatial_mask_list.append(mask)
            else:
                spatial_esm_list.append(np.zeros((1, 3840), dtype=np.float32))
                spatial_mask_list.append(np.zeros((1, 1), dtype=np.float32))

            if self.feats is not None:
                for key in feats_batch:
                    feats_batch[key].append(self.feats[key][i])

        input_ids = torch.tensor(input_ids).to(self.device)
        labels = torch.tensor(labels).to(self.device)
        spatial_esm = torch.tensor(np.array(spatial_esm_list)).to(self.device)
        spatial_mask = torch.tensor(np.array(spatial_mask_list)).to(self.device)

        if self.feats is not None:
            for key in feats_batch:
                feats_batch[key] = torch.tensor(np.array(feats_batch[key])).to(self.device)
            return input_ids, labels, feats_batch, spatial_esm, spatial_mask

        return input_ids, labels, spatial_esm, spatial_mask