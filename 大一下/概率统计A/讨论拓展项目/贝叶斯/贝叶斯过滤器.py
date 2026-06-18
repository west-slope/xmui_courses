"""
贝叶斯垃圾邮件过滤器.py
一个完整的朴素贝叶斯垃圾邮件分类系统
支持训练、预测、评估功能
"""

import re
import math
import json
import pickle
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import random


class BayesianSpamFilter:
    """朴素贝叶斯垃圾邮件过滤器"""
    
    def __init__(self, alpha: float = 1.0, threshold: float = 0.5):
        """
        初始化过滤器
        
        Args:
            alpha: 拉普拉斯平滑参数
            threshold: 分类阈值，概率超过此值判定为垃圾邮件
        """
        self.alpha = alpha  # 平滑参数
        self.threshold = threshold  # 分类阈值
        
        # 统计信息
        self.word_counts = {
            'spam': defaultdict(int),      # 垃圾邮件中词频
            'ham': defaultdict(int)        # 正常邮件中词频
        }
        self.total_words = {
            'spam': 0,     # 垃圾邮件总词数
            'ham': 0       # 正常邮件总词数
        }
        self.doc_counts = {
            'spam': 0,     # 垃圾邮件数量
            'ham': 0       # 正常邮件数量
        }
        
        # 词汇表
        self.vocabulary = set()
        
        # 停用词列表（常见无意义词）
        self.stopwords = {
            '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
            '没有', '看', '好', '自己', '这', '那', '什么', '个', '与', '及',
            'the', 'a', 'an', 'and', 'or', 'but', 'so', 'for', 'to', 'of', 'in',
            'on', 'at', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been'
        }
    
    def preprocess(self, text: str) -> List[str]:
        """
        文本预处理：转小写、分词、去停用词、去非字母字符
        
        Args:
            text: 原始邮件文本
            
        Returns:
            处理后的单词列表
        """
        # 转小写
        text = text.lower()
        
        # 移除非字母字符（保留空格和字母）
        text = re.sub(r'[^a-z\u4e00-\u9fff\s]', ' ', text)
        
        # 分词
        words = text.split()
        
        # 去停用词、去单字符词
        words = [w for w in words if w not in self.stopwords and len(w) > 1]
        
        return words
    
    def train(self, emails: List[Tuple[str, int]]):
        """
        训练模型
        
        Args:
            emails: 邮件列表，每个元素为(邮件内容, 标签)
                    标签：1表示垃圾邮件，0表示正常邮件
        """
        for content, label in emails:
            words = self.preprocess(content)
            
            category = 'spam' if label == 1 else 'ham'
            self.doc_counts[category] += 1
            
            for word in words:
                self.word_counts[category][word] += 1
                self.total_words[category] += 1
                self.vocabulary.add(word)
        
        print(f"训练完成！")
        print(f"正常邮件: {self.doc_counts['ham']} 封，总词数: {self.total_words['ham']}")
        print(f"垃圾邮件: {self.doc_counts['spam']} 封，总词数: {self.total_words['spam']}")
        print(f"词汇表大小: {len(self.vocabulary)}")
    
    def word_probability(self, word: str, category: str) -> float:
        """
        计算单词在某个类别下的概率（使用拉普拉斯平滑）
        
        Args:
            word: 单词
            category: 'spam' 或 'ham'
            
        Returns:
            概率值
        """
        count = self.word_counts[category].get(word, 0)
        total = self.total_words[category]
        vocab_size = len(self.vocabulary)
        
        # 拉普拉斯平滑
        return (count + self.alpha) / (total + self.alpha * vocab_size)
    
    def predict_probability(self, content: str) -> float:
        """
        预测邮件是垃圾邮件的概率
        
        Args:
            content: 邮件内容
            
        Returns:
            P(spam|email) 概率值
        """
        words = self.preprocess(content)
        
        # 先验概率
        total_docs = self.doc_counts['spam'] + self.doc_counts['ham']
        if total_docs == 0:
            return 0.5
        
        p_spam = self.doc_counts['spam'] / total_docs
        p_ham = self.doc_counts['ham'] / total_docs
        
        # 使用对数概率防止数值下溢
        log_p_spam = math.log(p_spam)
        log_p_ham = math.log(p_ham)
        
        for word in words:
            # 只考虑词汇表中的词
            if word in self.vocabulary:
                log_p_spam += math.log(self.word_probability(word, 'spam'))
                log_p_ham += math.log(self.word_probability(word, 'ham'))
        
        # 归一化得到概率
        p_spam_given_email = 1 / (1 + math.exp(log_p_ham - log_p_spam))
        
        return p_spam_given_email
    
    def predict(self, content: str) -> Tuple[int, float]:
        """
        预测邮件类别
        
        Args:
            content: 邮件内容
            
        Returns:
            (预测标签, 垃圾邮件概率)
        """
        prob = self.predict_probability(content)
        label = 1 if prob >= self.threshold else 0
        return label, prob
    
    def evaluate(self, test_emails: List[Tuple[str, int]]) -> Dict:
        """
        评估模型性能
        
        Args:
            test_emails: 测试邮件列表
            
        Returns:
            评估指标字典
        """
        TP = FP = FN = TN = 0
        
        for content, true_label in test_emails:
            pred_label, prob = self.predict(content)
            
            if true_label == 1 and pred_label == 1:
                TP += 1
            elif true_label == 0 and pred_label == 1:
                FP += 1
            elif true_label == 1 and pred_label == 0:
                FN += 1
            else:
                TN += 1
        
        # 计算指标
        accuracy = (TP + TN) / len(test_emails) if test_emails else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN
        }
    
    def save(self, filepath: str):
        """保存模型到文件"""
        model_data = {
            'word_counts': dict(self.word_counts['spam']), dict(self.word_counts['ham']),
            'total_words': self.total_words,
            'doc_counts': self.doc_counts,
            'vocabulary': list(self.vocabulary),
            'alpha': self.alpha,
            'threshold': self.threshold
        }
        # 修正保存格式
        model_data = {
            'word_counts_spam': dict(self.word_counts['spam']),
            'word_counts_ham': dict(self.word_counts['ham']),
            'total_words': self.total_words,
            'doc_counts': self.doc_counts,
            'vocabulary': list(self.vocabulary),
            'alpha': self.alpha,
            'threshold': self.threshold
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型已保存到 {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.word_counts['spam'] = defaultdict(int, model_data['word_counts_spam'])
        self.word_counts['ham'] = defaultdict(int, model_data['word_counts_ham'])
        self.total_words = model_data['total_words']
        self.doc_counts = model_data['doc_counts']
        self.vocabulary = set(model_data['vocabulary'])
        self.alpha = model_data['alpha']
        self.threshold = model_data['threshold']
        print(f"模型已从 {filepath} 加载")


# ==================== 示例数据和测试 ====================

def generate_sample_data(n_samples: int = 1000) -> List[Tuple[str, int]]:
    """
    生成模拟邮件数据用于演示
    
    Args:
        n_samples: 生成样本数量
        
    Returns:
        邮件列表
    """
    # 垃圾邮件特征词
    spam_words = ['免费', '中奖', '恭喜', '点击', '领取', '红包', '优惠', '现金', 
                  '赚钱', '兼职', '日结', '刷单', '贷款', '信用卡', '额度', 'VIP',
                  '免费领取', '点击链接', '立即注册', '恭喜中奖', '限时优惠']
    
    # 正常邮件特征词
    ham_words = ['会议', '报告', '项目', '进度', '同事', '领导', '审批', '文档',
                 '附件', '请查收', '回复', '抄送', '谢谢', '辛苦了', '安排',
                 '您好', '请问', '关于', '讨论', '方案', '需求']
    
    emails = []
    
    for i in range(n_samples):
        # 随机决定是垃圾(30%)还是正常(70%)
        is_spam = random.random() < 0.3
        
        if is_spam:
            # 生成垃圾邮件
            num_words = random.randint(20, 100)
            words = []
            for _ in range(num_words):
                # 80%概率使用垃圾词，20%概率使用普通词
                if random.random() < 0.8:
                    word = random.choice(spam_words)
                else:
                    word = random.choice(ham_words)
                words.append(word)
            # 加入一些垃圾邮件的典型模式
            if random.random() < 0.5:
                words.insert(0, '恭喜')
            if random.random() < 0.5:
                words.append('立即点击')
            label = 1
        else:
            # 生成正常邮件
            num_words = random.randint(30, 150)
            words = []
            for _ in range(num_words):
                # 70%概率使用正常词，30%概率使用普通词
                if random.random() < 0.7:
                    word = random.choice(ham_words)
                else:
                    word = random.choice(spam_words) if random.random() < 0.2 else '普通'
                words.append(word)
            label = 0
        
        content = ' '.join(words)
        emails.append((content, label))
    
    return emails


def demo():
    """演示完整流程"""
    print("=" * 60)
    print("     朴素贝叶斯垃圾邮件过滤器 - 演示")
    print("=" * 60)
    
    # 1. 生成模拟数据
    print("\n[1] 生成模拟邮件数据...")
    all_emails = generate_sample_data(2000)
    random.shuffle(all_emails)
    
    # 划分训练集和测试集
    split_idx = int(0.7 * len(all_emails))
    train_emails = all_emails[:split_idx]
    test_emails = all_emails[split_idx:]
    
    print(f"训练集: {len(train_emails)} 封邮件")
    print(f"测试集: {len(test_emails)} 封邮件")
    
    # 统计训练集分布
    train_spam = sum(1 for _, label in train_emails if label == 1)
    train_ham = len(train_emails) - train_spam
    print(f"训练集 - 垃圾邮件: {train_spam}, 正常邮件: {train_ham}")
    
    # 2. 创建并训练模型
    print("\n[2] 训练贝叶斯过滤器...")
    filter = BayesianSpamFilter(alpha=1.0, threshold=0.5)
    filter.train(train_emails)
    
    # 3. 评估模型
    print("\n[3] 评估模型性能...")
    metrics = filter.evaluate(test_emails)
    
    print(f"\n?? 评估结果:")
    print(f"   准确率 (Accuracy):  {metrics['accuracy']*100:.2f}%")
    print(f"   精确率 (Precision): {metrics['precision']*100:.2f}%")
    print(f"   召回率 (Recall):    {metrics['recall']*100:.2f}%")
    print(f"   F1分数:             {metrics['f1_score']*100:.2f}%")
    print(f"\n   混淆矩阵:")
    print(f"   TP(正确拦截): {metrics['TP']}")
    print(f"   FP(误拦截):   {metrics['FP']}")
    print(f"   FN(漏拦截):   {metrics['FN']}")
    print(f"   TN(正确放行): {metrics['TN']}")
    
    # 4. 测试示例邮件
    print("\n[4] 测试示例邮件...")
    
    test_cases = [
        ("恭喜你中奖了！请点击链接领取100万现金红包！", "典型垃圾邮件"),
        ("关于下周项目评审的会议通知，请查收附件", "典型正常邮件"),
        ("限时优惠！免费领取VIP会员，点击立即开通", "促销邮件（可能是垃圾）"),
        ("同事你好，请帮忙review一下这份代码，谢谢", "工作邮件（正常）"),
    ]
    
    print("\n" + "-" * 50)
    for content, desc in test_cases:
        label, prob = filter.predict(content)
        result = "?? 垃圾邮件" if label == 1 else "? 正常邮件"
        confidence = prob if label == 1 else 1 - prob
        print(f"\n?? {desc}: {content[:40]}...")
        print(f"   分类结果: {result}")
        print(f"   垃圾概率: {prob*100:.2f}%")
        print(f"   置信度: {confidence*100:.2f}%")
    
    # 5. 保存和加载模型
    print("\n[5] 保存和加载模型...")
    filter.save("spam_filter.pkl")
    
    new_filter = BayesianSpamFilter()
    new_filter.load("spam_filter.pkl")
    
    # 验证加载后的模型
    test_content = "免费领取红包"
    original_prob = filter.predict_probability(test_content)
    loaded_prob = new_filter.predict_probability(test_content)
    print(f"加载后模型验证: 原模型概率={original_prob*100:.2f}%, 加载后={loaded_prob*100:.2f}%")
    
    print("\n" + "=" * 60)
    print("演示完成！过滤器已就绪，可投入生产使用 ??")
    print("=" * 60)


if __name__ == "__main__":
    demo()