import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import statistics
from nltk.translate import bleu_score
import os
import pickle
from abc import ABC, abstractmethod

class commentfinderBaseTrainer(ABC):
    def __init__(self, model, base_path='./dataset/', cache_dir='./cache/'):
        self.model = model
        self.base_path = base_path
        self.cache_dir = cache_dir
        self.task_type = None  # Must be set by subclass
        self.chencherry = bleu_score.SmoothingFunction()
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_cache_path(self, filename):
        """获取缓存文件完整路径"""
        return os.path.join(self.cache_dir, f"{self.task_type}_{filename}")
    
    def compute_or_load_similarity(self, test_data_vect):
        """计算或加载余弦相似度矩阵"""
        if self.task_type is None:
            raise ValueError("task_type must be set by subclass")
            
        cache_file = self.get_cache_path('cosine_similarity.pkl')
        if os.path.exists(cache_file):
            print(f"Loading cached cosine similarity matrix for {self.task_type}...")
            return pickle.load(open(cache_file, 'rb'))
        
        print(f"Computing cosine similarity matrix for {self.task_type}...")
        start_time = time.time()
        similarity = cosine_similarity(test_data_vect, self.model.transform(self.model.source_train))
        print(f"Computed in {time.time()-start_time:.2f}s")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(similarity, f)
        return similarity
    
    def get_topk_indices(self, similarity, index, k=10):
        """获取top-k相似度索引"""
        # 第一阶段：基于余弦相似度粗筛
        index_nn = np.argpartition(similarity[index], -10)[-10:]
        
        # 第二阶段：基于GPM精细排序
        similar_nn = []
        for idx in index_nn:
            score = self.model.similar(self.model.source_test[index], self.model.source_train[idx])
            similar_nn.append((idx, score))
        
        # 按GPM分数降序排序
        similar_nn.sort(key=lambda x: x[1], reverse=True)
        return [e[0] for e in similar_nn[:k]]
    
    @abstractmethod
    def process_prediction(self, train_idx):
        """子类必须实现：如何处理训练样本索引到最终预测"""
        pass
    
    # 必要的参数注入，test_data_vect，但是这个参数应当也是commentfinderModel负责，本run只负责得到topk defined similar prediction
    ## 具体的返回内容由子类负责
    def run(self, test_data_vect, topk=1, save_predictions=True):
        """执行预测流程
        参数:
            test_data_vect: 测试数据的特征向量
            topk: 返回的top-k结果数量
            save_predictions: 是否保存预测结果
        返回:
            预测结果列表，每个元素是该样本的topk预测
        """
        # 检查是否已有缓存预测结果
        pred_cache_file = self.get_cache_path(f'predictions_k{topk}.txt')
        if os.path.exists(pred_cache_file):
            print(f"Loading cached predictions from {pred_cache_file}")
            with open(pred_cache_file, 'r') as f:
                # 按k值分组读取预测结果
                predictions = []
                current_group = []
                for line in f:
                    line = line.strip()
                    if line:  # 非空行
                        current_group.append(line)
                        if len(current_group) == topk:
                            predictions.append(current_group)
                            current_group = []
                return predictions
        
        # 没有缓存则进行计算
        print(f"Computing predictions for topk={topk}...")
        similarity = self.compute_or_load_similarity(test_data_vect)
        predictions = []
        
        for index in range(len(similarity)):
            top_indices = self.get_topk_indices(similarity, index, topk)
            current_pred = [self.process_prediction(idx) for idx in top_indices]
            predictions.append(current_pred)
        
        if save_predictions:
            self.save_predictions(predictions, topk)
        return predictions
    
    def evaluate(self, predictions, targets, k):
        """评估模型性能"""
        count_perfect = 0
        bleu_scores = []
        
        for i in tqdm(range(len(targets))):
            target = targets[i]
            best_bleu = 0
            
            for pred in predictions[i*k:(i+1)*k]:
                if " ".join(pred.split()) == " ".join(target.split()):
                    count_perfect += 1
                    best_bleu = bleu_score.sentence_bleu(
                        [target], pred,
                        smoothing_function=self.chencherry.method1)
                    break
                
                current = bleu_score.sentence_bleu(
                    [target], pred,
                    smoothing_function=self.chencherry.method1)
                best_bleu = max(best_bleu, current)
            
            bleu_scores.append(best_bleu)
        
        pp = (count_perfect * 100) / len(targets)
        mean_bleu = statistics.mean(bleu_scores)
        return pp, mean_bleu, bleu_scores
    
    def save_predictions(self, predictions, k):
        """保存预测结果"""
        pred_file = self.get_cache_path(f'predictions_k{k}.txt')
        with open(pred_file, 'w') as f:
            for pred in predictions:
                f.write("\n".join(pred) + "\n")
        print(f"Saved predictions to {pred_file}")
    
    def save_bleu_scores(self, bleu_scores, k):
        """保存BLEU分数"""
        bleu_file = self.get_cache_path(f'bleu_k{k}.txt')
        with open(bleu_file, 'w') as f:
            f.write("\n".join(map(str, bleu_scores)))
        print(f"Saved BLEU scores to {bleu_file}")


# 这几个子类的实现都是一致的，只是面向的数据集不一致
class commentfinderCLS(commentfinderBaseTrainer):
    """代码缺陷检测任务"""
    def __init__(self, model, base_path='./dataset/', cache_dir='./cache/'):
        super().__init__(model, base_path, cache_dir)
        self.task_type = 'CLS'
    
    def process_prediction(self, train_idx):
        """处理训练样本索引到分类预测"""
        # 假设target_train中存储的是分类标签
        return self.model.target_train[train_idx]

class commentfinderMSG(commentfinderBaseTrainer):
    """代码注释生成任务"""
    def __init__(self, model, base_path='./dataset/', cache_dir='./cache/'):
        super().__init__(model, base_path, cache_dir)
        self.task_type = 'MSG'
    
    def process_prediction(self, train_idx):
        """处理训练样本索引到注释生成"""
        # 假设target_train中存储的是注释文本
        # 可以添加额外的后处理逻辑
        return self.model.target_train[train_idx]

class commentfinderREF(commentfinderBaseTrainer):
    """代码修复生成任务"""
    def __init__(self, model, base_path='./dataset/', cache_dir='./cache/'):
        super().__init__(model, base_path, cache_dir)
        self.task_type = 'REF'
    
    def process_prediction(self, train_idx):
        """处理训练样本索引到修复建议"""
        # 假设target_train中存储的是修复后的代码
        # 可以添加额外的后处理逻辑
        return self.model.target_train[train_idx]