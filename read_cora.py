import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class RandomCoraDataset(Dataset):
    def __init__(self, content_path, cites_path, random_seed=None, noise_level=0.01):
        # 随机种子控制（None则每次运行不同）
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        
        # 读取数据
        content = pd.read_csv(content_path, sep='\t', header=None)
        cites = pd.read_csv(cites_path, sep='\t', header=None)
        
        # 1. 特征添加随机噪声
        features_np = content.iloc[:, 1:-1].values.astype(np.float32)
        features_np += np.random.normal(0, noise_level, size=features_np.shape)  # 高斯噪声
        self.features = torch.tensor(features_np, dtype=torch.float)
        
        # 2. 标签随机替换（1%概率）
        self.labels, self.label_mapping = pd.factorize(content.iloc[:, -1])
        num_classes = len(self.label_mapping)
        random_mask = np.random.rand(len(self.labels)) < 0.01  # 1%的标签随机替换
        self.labels[random_mask] = np.random.randint(0, num_classes, size=random_mask.sum())
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        # 3. 节点顺序随机打乱
        self.shuffle_indices = np.random.permutation(len(content))  # 随机索引
        self.edges = torch.tensor(cites.values, dtype=torch.long)
        self.num_nodes = len(content)

    def __len__(self):
        return self.num_nodes

    def __getitem__(self, idx):
        shuffled_idx = self.shuffle_indices[idx]  # 用随机索引取数据
        return {
            'features': self.features[shuffled_idx],
            'labels': self.labels[shuffled_idx]
        }

if __name__ == "__main__":
    content_path = "/home/jiayihang/11/cora./cora/cora.content"
    cites_path = "/home/jiayihang/11/cora./cora/cora.cites"
    
    # 不指定random_seed，每次运行随机结果不同
    dataset = RandomCoraDataset(content_path, cites_path, random_seed=None, noise_level=0.02)
    
    print("数据集总节点数:", len(dataset))
    sample = dataset[0]
    print("第0个节点的特征形状:", sample['features'].shape)
    print("第0个节点的标签:", sample['labels'])  # 每次运行可能不同
    print("标签映射关系:", dict(enumerate(dataset.label_mapping)))
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(dataloader))
    print("批量特征形状:", batch['features'].shape)
    print("批量标签形状:", batch['labels'].shape)
