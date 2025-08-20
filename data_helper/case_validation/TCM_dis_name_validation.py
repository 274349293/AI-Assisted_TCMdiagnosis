import pandas as pd
import json
import logging
from datetime import datetime
import time
import os
import hashlib
from collections import defaultdict
from openai import AzureOpenAI
import difflib
from fuzzywuzzy import fuzz
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DiseaseMatchingSystem:
    """
    中医疾病匹配验证系统
    """

    def __init__(self, azure_api_key=None, azure_endpoint=None, deployment_name="o3", cache_dir="cache"):
        """
        初始化匹配系统

        Args:
            azure_api_key (str): Azure OpenAI API密钥 (用于语义匹配)
            azure_endpoint (str): Azure OpenAI端点
            deployment_name (str): 部署名称
            cache_dir (str): 缓存目录
        """
        self.similarity_threshold = 0.7
        self.llm_threshold_low = 0.4
        self.llm_threshold_high = 0.7

        # Azure OpenAI配置 (可选)
        self.use_llm = azure_api_key and azure_endpoint
        if self.use_llm:
            # 修复endpoint格式
            if azure_endpoint.endswith('chat/completions?'):
                azure_endpoint = azure_endpoint.replace('/openai/deployments/o3/chat/completions?', '')

            self.client = AzureOpenAI(
                api_key=azure_api_key,
                api_version="2025-01-01-preview",
                azure_endpoint=azure_endpoint
            )
            self.deployment_name = deployment_name

            # 创建缓存目录
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)

            # API调用统计
            self.api_calls = 0
            self.cache_hits = 0
        else:
            logger.info("未配置Azure OpenAI，将跳过语义匹配步骤")

    def load_data(self, diseases_txt_path, json_path):
        """
        加载数据文件

        Args:
            diseases_txt_path (str): 门诊疾病名txt文件路径
            json_path (str): 标准疾病JSON文件路径

        Returns:
            tuple: (门诊疾病列表, 标准疾病列表)
        """
        logger.info("开始加载数据文件...")

        # 读取门诊疾病名
        with open(diseases_txt_path, 'r', encoding='utf-8') as f:
            clinic_diseases = [line.strip() for line in f if line.strip()]

        # 读取标准疾病JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            standard_data = json.load(f)
            standard_diseases = list(standard_data.keys())

        logger.info(f"门诊疾病数量: {len(clinic_diseases)}")
        logger.info(f"标准疾病数量: {len(standard_diseases)}")

        return clinic_diseases, standard_diseases

    def exact_match(self, clinic_disease, standard_diseases):
        """
        精确匹配

        Args:
            clinic_disease (str): 门诊疾病名
            standard_diseases (list): 标准疾病列表

        Returns:
            tuple: (是否匹配, 匹配的疾病名)
        """
        if clinic_disease in standard_diseases:
            return True, clinic_disease
        return False, None

    def levenshtein_distance(self, s1, s2):
        """计算编辑距离"""
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_

        # 返回相似度 (0-1)
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        return 1.0 - distances[-1] / max_len

    def jaccard_similarity(self, s1, s2, n=2):
        """计算Jaccard相似度 (基于n-gram)"""

        def get_ngrams(text, n):
            return set([text[i:i + n] for i in range(len(text) - n + 1)])

        ngrams1 = get_ngrams(s1, n)
        ngrams2 = get_ngrams(s2, n)

        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))

        return intersection / union

    def cosine_similarity(self, s1, s2):
        """计算余弦相似度 (基于字符向量)"""
        # 创建字符集合
        chars = set(s1 + s2)
        if not chars:
            return 1.0

        # 创建向量
        vec1 = [s1.count(c) for c in chars]
        vec2 = [s2.count(c) for c in chars]

        # 计算余弦相似度
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def lcs_similarity(self, s1, s2):
        """最长公共子序列相似度"""
        m, n = len(s1), len(s2)
        if m == 0 or n == 0:
            return 0.0

        # 动态规划计算LCS长度
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]
        max_length = max(m, n)

        return lcs_length / max_length

    def jaro_winkler_similarity(self, s1, s2):
        """Jaro-Winkler相似度 (简化实现)"""
        # 使用difflib作为近似实现
        return difflib.SequenceMatcher(None, s1, s2).ratio()

    def calculate_similarity(self, clinic_disease, standard_disease):
        """
        计算综合相似度

        Args:
            clinic_disease (str): 门诊疾病名
            standard_disease (str): 标准疾病名

        Returns:
            float: 相似度 (0-1)
        """
        # 多种算法计算相似度
        similarities = []

        # 1. 编辑距离相似度
        similarities.append(self.levenshtein_distance(clinic_disease, standard_disease))

        # 2. Jaccard相似度
        similarities.append(self.jaccard_similarity(clinic_disease, standard_disease))

        # 3. 余弦相似度
        similarities.append(self.cosine_similarity(clinic_disease, standard_disease))

        # 4. LCS相似度
        similarities.append(self.lcs_similarity(clinic_disease, standard_disease))

        # 5. Jaro-Winkler相似度
        similarities.append(self.jaro_winkler_similarity(clinic_disease, standard_disease))

        # 6. fuzzywuzzy ratio
        similarities.append(fuzz.ratio(clinic_disease, standard_disease) / 100.0)

        # 取最大值作为最终相似度 (不使用权重)
        return max(similarities)

    def similarity_match(self, clinic_disease, standard_diseases):
        """
        相似度匹配

        Args:
            clinic_disease (str): 门诊疾病名
            standard_diseases (list): 标准疾病列表

        Returns:
            list: [(疾病名, 相似度), ...] 按相似度降序排列
        """
        similarities = []

        for standard_disease in standard_diseases:
            similarity = self.calculate_similarity(clinic_disease, standard_disease)
            similarities.append((standard_disease, similarity))

        # 按相似度降序排列
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities

    def get_cache_key(self, text):
        """生成缓存键"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def load_cache(self, cache_file):
        """加载缓存文件"""
        if not self.use_llm:
            return {}

        cache_path = os.path.join(self.cache_dir, cache_file)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_cache(self, cache_file, cache_data):
        """保存缓存文件"""
        if not self.use_llm:
            return

        cache_path = os.path.join(self.cache_dir, cache_file)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

    def call_azure_api(self, prompt, max_retries=3):
        """
        调用Azure OpenAI API（带缓存）
        """
        if not self.use_llm:
            return None

        # 生成缓存键
        cache_key = self.get_cache_key(prompt)
        cache_file = "disease_matching_api_responses.json"
        cache_data = self.load_cache(cache_file)

        # 检查缓存
        if cache_key in cache_data:
            self.cache_hits += 1
            return cache_data[cache_key]

        # API调用
        for attempt in range(max_retries):
            try:
                self.api_calls += 1
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system",
                         "content": "你是一位资深的中医专家，精通中医疾病分类和命名规范，擅长识别不同疾病名称是否指向同一种疾病。"},
                        {"role": "user", "content": prompt}
                    ],
                )

                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    result = response.choices[0].message.content.strip()
                    # 保存到缓存
                    cache_data[cache_key] = result
                    self.save_cache(cache_file, cache_data)
                    return result
                else:
                    logger.warning(f"API返回空响应 (尝试 {attempt + 1}/{max_retries})")

            except Exception as e:
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        logger.error("API调用最终失败")
        return None

    def semantic_match(self, clinic_disease, candidate_diseases):
        """
        语义匹配 (使用GPT-o3)

        Args:
            clinic_disease (str): 门诊疾病名
            candidate_diseases (list): 候选疾病列表 [(疾病名, 相似度), ...]

        Returns:
            tuple: (是否匹配, 匹配的疾病名, 置信度)
        """
        if not self.use_llm or not candidate_diseases:
            return False, None, "无"

        # 构建候选疾病列表文本
        candidates_text = []
        for i, (disease, similarity) in enumerate(candidate_diseases[:5], 1):  # 最多取前5个
            candidates_text.append(f"{i}. {disease} (相似度: {similarity:.3f})")

        prompt = f"""
作为中医专家，请判断门诊疾病名是否与以下候选疾病中的任何一个指向同一种疾病。

门诊疾病名：{clinic_disease}

候选疾病列表：
{chr(10).join(candidates_text)}

判断标准：
1. 考虑中医疾病的多种表达方式和历史演变
2. 考虑古代和现代术语的差异
3. 考虑地域性表达差异
4. 考虑是否为同一疾病的不同分类方式

请返回JSON格式：
{{
  "is_match": true/false,
  "matched_disease": "匹配的疾病名" (如果匹配的话),
  "confidence": "高/中/低",
  "reason": "判断理由"
}}

如果没有匹配，matched_disease设为null。
"""

        response = self.call_azure_api(prompt)

        if response:
            try:
                result = json.loads(response)
                is_match = result.get('is_match', False)
                matched_disease = result.get('matched_disease')
                confidence = result.get('confidence', '低')

                return is_match, matched_disease, confidence

            except json.JSONDecodeError:
                logger.warning(f"语义匹配JSON解析失败: {clinic_disease}")
                return False, None, "低"

        return False, None, "低"

    def match_single_disease(self, clinic_disease, standard_diseases):
        """
        匹配单个疾病

        Args:
            clinic_disease (str): 门诊疾病名
            standard_diseases (list): 标准疾病列表

        Returns:
            dict: 匹配结果
        """
        result = {
            '门诊中疾病名': clinic_disease,
            '是否匹配': False,
            '匹配类型': '未匹配',
            '最佳匹配疾病': '没有找到匹配疾病',
            '匹配相似度': 0.0,
            '相似度匹配疾病名列表': ''
        }

        # 第一层：精确匹配
        is_exact, exact_match = self.exact_match(clinic_disease, standard_diseases)
        if is_exact:
            result.update({
                '是否匹配': True,
                '匹配类型': '精确匹配',
                '最佳匹配疾病': exact_match,
                '匹配相似度': 1.0,
                '相似度匹配疾病名列表': exact_match
            })
            return result

        # 第二层：相似度匹配
        similarities = self.similarity_match(clinic_disease, standard_diseases)

        # 找出超过阈值的候选疾病
        high_similarity_candidates = [(disease, sim) for disease, sim in similarities if
                                      sim >= self.similarity_threshold]

        if high_similarity_candidates:
            best_disease, best_similarity = high_similarity_candidates[0]
            candidates_list = ','.join([disease for disease, _ in high_similarity_candidates])

            result.update({
                '是否匹配': True,
                '匹配类型': '高相似度匹配',
                '最佳匹配疾病': best_disease,
                '匹配相似度': best_similarity,
                '相似度匹配疾病名列表': candidates_list
            })
            return result

        # 第三层：语义匹配 (对0.4-0.7之间的候选疾病)
        medium_similarity_candidates = [(disease, sim) for disease, sim in similarities
                                        if self.llm_threshold_low <= sim < self.llm_threshold_high]

        if medium_similarity_candidates and self.use_llm:
            is_semantic_match, matched_disease, confidence = self.semantic_match(clinic_disease,
                                                                                 medium_similarity_candidates)

            if is_semantic_match and matched_disease:
                # 找到匹配的相似度
                matched_similarity = next(
                    (sim for disease, sim in medium_similarity_candidates if disease == matched_disease), 0.0)

                result.update({
                    '是否匹配': True,
                    '匹配类型': f'语义匹配({confidence}置信度)',
                    '最佳匹配疾病': matched_disease,
                    '匹配相似度': matched_similarity,
                    '相似度匹配疾病名列表': matched_disease
                })
                return result

        # 如果都没有匹配，显示最高相似度（如果有的话）
        if similarities:
            best_disease, best_similarity = similarities[0]
            result.update({
                '最佳匹配疾病': f'{best_disease}(相似度:{best_similarity:.3f})',
                '匹配相似度': best_similarity
            })

        return result

    def run_matching(self, diseases_txt_path, json_path, output_path):
        """
        运行完整的疾病匹配流程

        Args:
            diseases_txt_path (str): 门诊疾病txt文件路径
            json_path (str): 标准疾病json文件路径
            output_path (str): 输出Excel文件路径
        """
        start_time = time.time()

        logger.info("=" * 80)
        logger.info("开始中医疾病匹配验证任务")
        logger.info("=" * 80)

        try:
            # 步骤1：加载数据
            logger.info("📋 步骤1: 加载数据文件")
            clinic_diseases, standard_diseases = self.load_data(diseases_txt_path, json_path)

            # 步骤2：批量匹配
            logger.info("📋 步骤2: 开始批量匹配疾病")
            results = []

            total_diseases = len(clinic_diseases)
            for i, clinic_disease in enumerate(clinic_diseases, 1):
                logger.info(f"处理进度: {i}/{total_diseases} - {clinic_disease}")

                result = self.match_single_disease(clinic_disease, standard_diseases)
                results.append(result)

                # 避免API调用过快
                if self.use_llm and i % 10 == 0:
                    time.sleep(0.5)

            # 步骤3：保存结果
            logger.info("📋 步骤3: 保存匹配结果")
            df_results = pd.DataFrame(results)
            df_results.to_excel(output_path, index=False)

            # 步骤4：统计信息
            end_time = time.time()
            processing_time = end_time - start_time

            # 统计匹配情况
            total_count = len(results)
            matched_count = len([r for r in results if r['是否匹配']])
            exact_match_count = len([r for r in results if r['匹配类型'] == '精确匹配'])
            similarity_match_count = len([r for r in results if r['匹配类型'] == '高相似度匹配'])
            semantic_match_count = len([r for r in results if '语义匹配' in r['匹配类型']])

            logger.info("=" * 80)
            logger.info("📊 疾病匹配完成统计")
            logger.info("=" * 80)
            logger.info(f"处理时间: {processing_time:.2f} 秒")
            if self.use_llm:
                logger.info(f"API调用次数: {self.api_calls}")
                logger.info(f"缓存命中次数: {self.cache_hits}")
                if self.api_calls + self.cache_hits > 0:
                    cache_rate = self.cache_hits / (self.api_calls + self.cache_hits) * 100
                    logger.info(f"缓存命中率: {cache_rate:.1f}%")

            logger.info(f"总疾病数: {total_count}")
            logger.info(f"匹配成功: {matched_count} ({matched_count / total_count * 100:.1f}%)")
            logger.info(f"  - 精确匹配: {exact_match_count}")
            logger.info(f"  - 高相似度匹配: {similarity_match_count}")
            if self.use_llm:
                logger.info(f"  - 语义匹配: {semantic_match_count}")
            logger.info(f"未匹配: {total_count - matched_count}")
            logger.info(f"输出文件: {output_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"程序执行出错: {str(e)}")
            raise


def main():
    """
    主函数
    """
    # Azure OpenAI配置 (可选，用于语义匹配)
    AZURE_API_KEY = ""  # 可以留空，将跳过语义匹配
    AZURE_ENDPOINT = ""  # 可以留空，将跳过语义匹配
    DEPLOYMENT_NAME = "o3"

    # 文件路径
    diseases_txt_path = "../../data/other/dvalidated_tcm_diseases.txt"
    json_path = "../../data/other/disease_to_syndromes_merged.json"
    output_path = "../../data/result/disease_matching_results.xlsx"

    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 创建匹配系统
    matcher = DiseaseMatchingSystem(
        azure_api_key=AZURE_API_KEY if AZURE_API_KEY else None,
        azure_endpoint=AZURE_ENDPOINT if AZURE_ENDPOINT else None,
        deployment_name=DEPLOYMENT_NAME
    )

    # 运行匹配
    matcher.run_matching(diseases_txt_path, json_path, output_path)


if __name__ == "__main__":
    main()
