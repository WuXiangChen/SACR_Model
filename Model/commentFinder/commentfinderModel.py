import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import string
from difflib import SequenceMatcher
import os
import pickle

class commentfinderModel:
    def __init__(self, args=args, config=None, max_df=0.5):
        self.vectorizer = CountVectorizer(max_df=max_df)
        self.source_train = None
        self.target_train = None
        self.source_test = None  # 新增测试数据存储
        self.test_data_vect = None  # 新增测试数据向量
        self.punctuations = string.punctuation.replace("\"", "")
        self.cache_dir = args.model_name_or_path # 这里应该是commentfinderModel将文件处理后的保存路径
        self.task_type = None
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def set_task_type(self, task_type=None):
        self.task_type = task_type
    
    def get_cache_path(self, filename):
        """获取缓存文件完整路径"""
        if self.task_type is None:
            raise ValueError("The task_type should be specified first in commentfinderModel.")
        return os.path.join(self.cache_dir, f"{self.task_type}_{filename}")
    
    def load_data(self, base_path='./dataset/'):
        """加载训练和测试数据"""
        # 训练数据
        train_path = os.path.join(base_path, 'train.tsv')
        with open(train_path) as f:
            train_data = [line.strip() for line in f]
        
        # 测试数据
        test_source_path = os.path.join(base_path, 'source.txt')
        with open(test_source_path) as f:
            test_sources = [line.strip() for line in f]
        
        # 目标数据（根据任务类型不同可能不同）
        test_target_path = os.path.join(base_path, f'target_{self.task_type}.txt')
        with open(test_target_path) as f:
            test_targets = [line.strip() for line in f]
        
        return train_data, test_sources, test_targets
    
    def preprocess_text(self, text):
        """处理文本中的标点符号"""
        return text.translate(
            str.maketrans({key: f" {key} " for key in self.punctuations}))
    
    def process_dataset(self, dataset, is_train=True):
        """处理数据集"""
        sources, targets = [], []
        for data in dataset:
            if is_train:
                source, target = data.split("\t")
                sources.append(self.preprocess_text(source))
                targets.append(target)
            else:
                source = data.split("code2comment :")[1]
                sources.append(self.preprocess_text(source))
        return (sources, targets) if is_train else sources
    
    # base_path是数据注入的路径
    def fit(self, base_path='./dataset/'):
        """训练模型并准备测试数据"""
        # 检查缓存是否存在
        vectorizer_cache = self.get_cache_path('vectorizer.pkl')
        train_cache = self.get_cache_path('train_data.pkl')
        test_cache = self.get_cache_path('test_data.pkl')
        
        if all(os.path.exists(f) for f in [vectorizer_cache, train_cache, test_cache]):
            # 从缓存加载
            with open(vectorizer_cache, 'rb') as f:
                self.vectorizer = pickle.load(f)
            self.source_train, self.target_train = pickle.load(open(train_cache, 'rb'))
            self.source_test, self.test_data_vect = pickle.load(open(test_cache, 'rb'))
        else:
            # 加载并处理数据
            train_data, test_sources, _ = self.load_data(base_path)
            
            # 处理训练数据
            self.source_train, self.target_train = self.process_dataset(train_data)
            self.vectorizer.fit(self.source_train)
            
            # 处理测试数据
            self.source_test = self.process_dataset(test_sources, is_train=False)
            self.test_data_vect = self.vectorizer.transform(self.source_test)
            
            # 保存到缓存
            with open(vectorizer_cache, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            pickle.dump((self.source_train, self.target_train), open(train_cache, 'wb'))
            pickle.dump((self.source_test, self.test_data_vect), open(test_cache, 'wb'))
    
    def transform(self, texts):
        """将文本转换为特征向量"""
        return self.vectorizer.transform(texts)
    
    @staticmethod
    def similar(a, b):
        """计算GPM相似度"""
        return SequenceMatcher(None, a, b).ratio()