import math

from torch.utils.data import DataLoader
from tqdm import tqdm
from core.multimodal_dataset import MultiModalDataSet
from helper import io_util
import json
import pandas as pd
import numpy as np
from process.events.lda_w2v import LDAEncoder
from process.events.fasttext_w2v import FastTextEncoder
from process.events.cnn1d_w2v import CNNW2VEncoder

class EventProcess():

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.embedding_dim = args.embedding_dim
        self.dataset = args.dataset

    def process(self, reconstruct=False, device="cpu"):
        self.data_path = f"data/{self.dataset}"

        label_path = f"data/{self.dataset}/label.csv"
        metric_path = f"data/{self.dataset}/raw/metric.json"
        trace_path = f"data/{self.dataset}/raw/trace.json"
        log_path = f"data/{self.dataset}/raw/log.json"
        edge_path = f"data/{self.dataset}/raw/edges.pkl"
        node_path = f"data/{self.dataset}/raw/nodes.pkl"
        deployment_path = f"data/{self.dataset}/raw/deployments.pkl"

        self.logger.info(f"Load raw events from {self.dataset} dataset")
        self.labels = pd.read_csv(label_path)
        with open(metric_path, 'r', encoding='utf8') as fp:
            self.metrics = json.load(fp)
        with open(trace_path, 'r', encoding='utf8') as fp:
            self.traces = json.load(fp)
        with open(log_path, 'r', encoding='utf8') as fp:
            self.logs = json.load(fp)

        self.edges = io_util.load(edge_path)
        self.nodes = io_util.load(node_path)

        # 将gaia数据视为一个集群，与aiops22数据集保持一致
        if self.args.dataset == "gaia":
            self.deployment = {1: io_util.load(deployment_path)}
        else:
            self.deployment = io_util.load(deployment_path)
        self.types = ['normal'] + self.labels['anomaly_type'].unique().tolist()

        if reconstruct:
            self.build_embedding()

        return self.build_dataset(device)

    def build_embedding(self):
        self.logger.info(f"Build embedding for raw events")
        # metric event: (instance, host, metric_name, 'abnormal')
        # trace event: (edge, host, error_type)
        # log event: (instance, eventId, TF-IDF)

        # train with LDA
        data_map = {'metric': self.metrics, 'trace': self.traces, 'log': self.logs}
        for key, data in data_map.items():
            encoder = FastTextEncoder(key, self.nodes, self.types, embedding_dim=self.embedding_dim, epochs=5)
            # encoder = LDAEncoder(num_topics=self.args.embedding_dim)
            # encoder = CNNW2VEncoder(
            #     seq_hidden=self.args.seq_hidden,
            #     embedding_dim=self.args.embedding_dim)

            train_idxs = self.labels[self.labels['data_type']=='train']['index'].values.tolist()
            train_ins_labels = self.labels[self.labels['data_type']=='train']['instance'].values.tolist()
            train_type_labels = self.labels[self.labels['data_type']=='train']['anomaly_type'].values.tolist()
            docs = []
            labels = []
            for i, idx in enumerate(train_idxs):
                for node in self.nodes:
                    if key == 'trace':
                        doc=['&'.join(e) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                    else:
                        doc=['&'.join(e) for e in data[str(idx)] if node in e[0]]
                    docs.append(doc)
                    if node == train_ins_labels[i]:
                        labels.append(f'__label__{self.nodes.index(node)}{self.types.index(train_type_labels[i])}')
                    else:
                        labels.append(f'__label__{self.nodes.index(node)}0')
            encoder.fit(docs, labels)

            # build embedding
            embs = []
            for idx in self.labels['index']:
                # group by instance
                graph_embs = []
                for node in self.nodes:
                    if key == 'trace':
                        doc=['&'.join(e) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                    else:
                        doc=['&'.join(e) for e in data[str(idx)] if node in e[0]]
                    
                    emb = encoder.get_sentence_embedding(doc)
                    graph_embs.append(emb)
                embs.append(graph_embs)
            io_util.save(f"data/{self.dataset}/tmp/{key}.pkl", np.array(embs))


    def build_dataset(self, device):
        self.logger.info(f"Build dataset for training")
        metric_embs = io_util.load(f"data/{self.dataset}/tmp/metric.pkl")
        trace_embs = io_util.load(f"data/{self.dataset}/tmp/trace.pkl")
        log_embs = io_util.load(f"data/{self.dataset}/tmp/log.pkl")

        label_types = ['anomaly_type', 'instance']
        label_dict = {label_type: None for label_type in label_types}
        for label_type in label_types:
            label_dict[label_type] = self.get_label(label_type, self.labels)

        train_index = np.where(self.labels['data_type'].values == 'train')
        test_index = np.where(self.labels['data_type'].values == 'test')

        train_metric_Xs = metric_embs[train_index]
        train_trace_Xs = trace_embs[train_index]
        train_log_Xs = log_embs[train_index]
        # train_service_labels = label_dict['service'][train_index]
        train_instance_labels = label_dict['instance'][train_index]
        train_type_labels = label_dict['anomaly_type'][train_index]


        test_metric_Xs = metric_embs[test_index]
        test_trace_Xs = trace_embs[test_index]
        test_log_Xs = log_embs[test_index]
        # test_service_labels = label_dict['service'][test_index]
        test_instance_labels = label_dict['instance'][test_index]
        test_type_labels = label_dict['anomaly_type'][test_index]

        # 添加部署关系
        if self.args.dataset == "gaia":
            train_cloudbed_index = [1 for i in range(len(train_instance_labels))]
            test_cloudbed_index = [1 for i in range(len(test_instance_labels))]
        else:
            train_cloudbed_index = list(map(lambda item: int(item.split("_")[0]), self.labels.loc[train_index]['index'].values))
            test_cloudbed_index = list(map(lambda item: int(item.split("_")[0]), self.labels.loc[test_index]['index'].values))
        self.logger.info("train root cause: \n{} failure type: \n{}".format(pd.value_counts(train_instance_labels), pd.value_counts(train_type_labels)))
        train_data = MultiModalDataSet(train_metric_Xs, train_trace_Xs, train_log_Xs, train_instance_labels,train_type_labels, self.nodes, self.edges, device, self.deployment, train_cloudbed_index, False)
        self.logger.info("test root cause:\n {} failure type: \n{}".format(pd.value_counts(test_instance_labels), pd.value_counts(test_type_labels)))
        test_data = MultiModalDataSet(test_metric_Xs, test_trace_Xs, test_log_Xs, test_instance_labels, test_type_labels, self.nodes, self.edges, device, self.deployment, test_cloudbed_index)

        return train_data, test_data

    def get_label(self, label_type, run_table):
        meta_labels = sorted(list(set(list(run_table[label_type]))))
        labels_idx = {label: idx for label, idx in zip(meta_labels, range(len(meta_labels)))}
        labels = np.array(run_table[label_type].apply(lambda label_str: labels_idx[label_str]))
        return labels