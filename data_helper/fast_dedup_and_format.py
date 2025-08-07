import pandas as pd
import json
import logging
from datetime import datetime
from collections import defaultdict
from openai import AzureOpenAI
import time
import os
import hashlib
import re
from difflib import SequenceMatcher

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
基于medical_record_templatization_0806.py的执行结果：Disease描述库_优化版_20250806_121343.xlsx 来执行处理

进行相似的症状去重 ， 格式重组

"""
class FastDescriptionDeduplicator:
    """
    快速描述去重和格式化工具
    规则优先 + LLM辅助 = 5-10倍速度提升
    """

    def __init__(self, azure_api_key, azure_endpoint, deployment_name="o3", cache_dir="cache"):
        """初始化快速去重工具"""
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

        # 8个字段
        self.key_fields = ['主诉', '现病史', '既往史', '辅助检查', 'PE/检查', '病机', '治则/处理', '医嘱']

        # 统计
        self.api_calls = 0
        self.cache_hits = 0
        self.rule_processed = 0
        self.llm_processed = 0

    def get_cache_key(self, text):
        """生成缓存键"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def load_cache(self, cache_file):
        """加载缓存文件"""
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
        cache_path = os.path.join(self.cache_dir, cache_file)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

    def normalize_description(self, desc):
        """标准化描述（用于相似度比较）"""
        if not desc:
            return ""

        # 时间标准化
        normalized = re.sub(r'[0-9]+[天月年周日]', 'N天', desc)
        normalized = re.sub(r'数[天月年周日]', 'N天', normalized)
        normalized = re.sub(r'多[天月年周日]', 'N天', normalized)
        normalized = re.sub(r'余[天月年周日]', 'N天', normalized)

        # 数字标准化
        normalized = re.sub(r'[0-9]+[个次回度]', 'N个', normalized)
        normalized = re.sub(r'第[0-9]+', '第N', normalized)

        # 去除标点和空格
        normalized = re.sub(r'[，。、；：！？\s]', '', normalized)

        return normalized.lower()

    def calculate_similarity(self, desc1, desc2):
        """计算两个描述的相似度"""
        norm1 = self.normalize_description(desc1)
        norm2 = self.normalize_description(desc2)

        if not norm1 or not norm2:
            return 0.0

        # 使用序列匹配器计算相似度
        return SequenceMatcher(None, norm1, norm2).ratio()

    def is_containment_relation(self, desc1, desc2):
        """判断是否为包含关系"""
        norm1 = self.normalize_description(desc1)
        norm2 = self.normalize_description(desc2)

        if not norm1 or not norm2:
            return False, None

        # 检查包含关系
        if norm1 in norm2:
            return True, desc1  # desc1 更简洁
        elif norm2 in norm1:
            return True, desc2  # desc2 更简洁

        return False, None

    def rule_based_grouping(self, descriptions):
        """基于规则的快速分组"""
        if len(descriptions) <= 1:
            return [[desc] for desc in descriptions]

        groups = []
        processed = set()

        for i, desc1 in enumerate(descriptions):
            if desc1 in processed:
                continue

            current_group = [desc1]
            processed.add(desc1)

            # 找相似和包含的描述
            for j, desc2 in enumerate(descriptions):
                if i == j or desc2 in processed:
                    continue

                # 检查包含关系
                is_contain, shorter = self.is_containment_relation(desc1, desc2)
                if is_contain:
                    current_group.append(desc2)
                    processed.add(desc2)
                    continue

                # 检查相似度
                similarity = self.calculate_similarity(desc1, desc2)
                if similarity > 0.85:  # 高相似度阈值
                    current_group.append(desc2)
                    processed.add(desc2)

            groups.append(current_group)

        return groups

    def merge_similar_group(self, group):
        """合并相似描述组"""
        if len(group) == 1:
            return group[0]

        # 找最短的作为基础（通常最简洁）
        base_desc = min(group, key=len)

        # 提取所有独特的关键词
        all_parts = set()
        locations = set()  # 部位词
        symptoms = set()  # 症状词

        location_patterns = ['颈', '肩', '腰', '膝', '头', '胸', '腹', '背', '臀', '腿', '手', '足']
        symptom_patterns = ['疼痛', '不适', '酸胀', '麻木', '僵硬', '头痛', '头晕', '乏力', '疲劳']

        for desc in group:
            # 提取部位
            for loc in location_patterns:
                if loc in desc:
                    locations.add(loc)

            # 提取症状
            for symp in symptom_patterns:
                if symp in desc:
                    symptoms.add(symp)

        # 智能合并
        if len(locations) > 1:
            # 多个部位：颈[肩/腰]疼痛
            loc_str = '[' + '/'.join(sorted(locations)) + ']'
            if len(symptoms) > 1:
                symp_str = '/'.join(sorted(symptoms))
                return f"{loc_str}{symp_str}N天"
            elif symptoms:
                return f"{loc_str}{list(symptoms)[0]}N天"

        if len(symptoms) > 1:
            # 多个症状：疼痛/不适
            return '/'.join(sorted(symptoms)) + 'N天'

        # 默认返回最短的描述
        return self.normalize_time_expression(base_desc)

    def normalize_time_expression(self, desc):
        """标准化时间表达"""
        # 统一为N天
        desc = re.sub(r'[0-9]+个?[天月年周日]', 'N天', desc)
        desc = re.sub(r'数[天月年周日]', 'N天', desc)
        desc = re.sub(r'多[天月年周日]', 'N天', desc)
        desc = re.sub(r'余[天月年周日]', 'N天', desc)
        desc = re.sub(r'[天月年周日]余', 'N天', desc)

        return desc

    def need_llm_processing(self, groups, original_count):
        """判断是否需要LLM处理"""
        # 如果规则已经达到很好的去重效果，就不用LLM
        merged_count = len(groups)
        reduction_rate = (original_count - merged_count) / original_count

        # 如果去重率已经超过30%，或者剩余组数很少，就不用LLM
        if reduction_rate > 0.3 or merged_count <= 3:
            return False

        # 检查是否有复杂情况需要LLM
        complex_groups = 0
        for group in groups:
            if len(group) > 1:
                # 检查组内是否还有可以进一步合并的
                for i, desc1 in enumerate(group):
                    for j, desc2 in enumerate(group[i + 1:], i + 1):
                        sim = self.calculate_similarity(desc1, desc2)
                        if 0.6 < sim <= 0.85:  # 中等相似度，需要LLM判断
                            complex_groups += 1
                            break
                    if complex_groups > 0:
                        break

        return complex_groups > 0

    def call_azure_api(self, prompt, max_retries=3):
        """调用Azure OpenAI API（带缓存）"""
        # 生成缓存键
        cache_key = self.get_cache_key(prompt)
        cache_file = "fast_dedup_api_responses.json"
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
                         "content": "你是一个专业的医疗信息处理专家，擅长快速识别和合并相似的医疗描述。请简洁回答。"},
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
                    time.sleep(1)  # 缩短等待时间

        logger.error("API调用最终失败")
        return None

    def llm_fine_tune(self, field_name, groups):
        """LLM精细调整（只处理复杂情况）"""
        if len(groups) <= 1:
            return [self.merge_similar_group(group) for group in groups]

        # 只发送需要LLM判断的组
        complex_groups_text = []
        simple_results = []

        for i, group in enumerate(groups):
            if len(group) == 1:
                # 单个描述，直接保留
                simple_results.append((i, group[0]))
            else:
                # 多个描述，需要LLM判断如何合并
                group_text = f"组{i + 1}: {' | '.join(group)}"
                complex_groups_text.append(group_text)

        if not complex_groups_text:
            # 没有复杂组，直接返回规则结果
            return [group[0] for group in groups]

        # 构建简化的LLM prompt
        prompt = f"""
快速合并以下{field_name}字段的相似描述组，每组合并为1个描述：

{chr(10).join(complex_groups_text)}

规则：
1. 相似症状用/连接：疼痛/不适
2. 多部位用[]：颈[肩/项]
3. 时间统一：N天
4. 保持简洁

JSON格式返回：["合并后描述1", "合并后描述2", ...]
"""

        response = self.call_azure_api(prompt)
        if response:
            try:
                llm_results = json.loads(response)
                if isinstance(llm_results, list):
                    # 合并LLM结果和简单结果
                    final_results = []
                    llm_index = 0

                    for i, group in enumerate(groups):
                        if len(group) == 1:
                            final_results.append(group[0])
                        else:
                            if llm_index < len(llm_results):
                                final_results.append(llm_results[llm_index])
                                llm_index += 1
                            else:
                                # LLM结果不够，使用规则合并
                                final_results.append(self.merge_similar_group(group))

                    return final_results
            except json.JSONDecodeError:
                logger.warning(f"{field_name}字段LLM结果解析失败，使用规则合并")

        # 备用方案：规则合并
        return [self.merge_similar_group(group) for group in groups]

    def fast_deduplicate_field(self, field_name, descriptions):
        """快速去重单个字段（规则优先+LLM辅助）"""
        if len(descriptions) <= 1:
            return descriptions

        original_count = len(descriptions)

        # 第1步：规则快速分组（90%情况处理完）
        groups = self.rule_based_grouping(descriptions)

        # 第2步：判断是否需要LLM精细处理
        if self.need_llm_processing(groups, original_count):
            # 需要LLM处理复杂情况
            result = self.llm_fine_tune(field_name, groups)
            self.llm_processed += 1
        else:
            # 规则处理已足够
            result = [self.merge_similar_group(group) for group in groups]
            self.rule_processed += 1

        # 去除空值和重复
        result = list(filter(None, list(dict.fromkeys(result))))

        return result

    def process_disease_data(self, df):
        """
        快速处理Disease数据去重
        """
        logger.info("开始快速去重处理...")

        # 按Disease和字段分组
        disease_field_data = defaultdict(lambda: defaultdict(list))

        for _, row in df.iterrows():
            disease = row['Disease']
            field = row['来源字段']
            description = row['描述内容']

            if description and str(description).strip():
                disease_field_data[disease][field].append(str(description).strip())

        # 快速处理每个Disease
        processed_data = {}
        total_diseases = len(disease_field_data)

        for i, (disease, field_data) in enumerate(disease_field_data.items(), 1):
            logger.info(f"快速处理Disease {i}/{total_diseases}: {disease}")
            processed_data[disease] = {}

            for field in self.key_fields:
                descriptions = field_data.get(field, [])
                if descriptions:
                    # 去重前后数量
                    before_count = len(descriptions)
                    deduplicated = self.fast_deduplicate_field(field, descriptions)
                    after_count = len(deduplicated)

                    processed_data[disease][field] = deduplicated

                    reduction = before_count - after_count
                    if reduction > 0:
                        logger.info(f"  {field}: {before_count} → {after_count} (-{reduction})")
                    else:
                        logger.info(f"  {field}: {before_count} (无变化)")
                else:
                    processed_data[disease][field] = []

            # 减少延时
            if i < total_diseases:
                time.sleep(0.1)  # 只在有LLM调用时才需要延时

        return processed_data

    def format_to_vertical_table(self, processed_data):
        """格式化为纵向表格"""
        logger.info("格式化为纵向表格...")

        all_rows = []

        for disease, field_data in processed_data.items():
            # 找到最长的字段
            max_length = max([len(descriptions) for descriptions in field_data.values()] + [0])

            if max_length == 0:
                continue

            # 为该Disease生成多行数据
            for row_index in range(max_length):
                row = {'Disease': disease if row_index == 0 else ''}

                for field in self.key_fields:
                    descriptions = field_data.get(field, [])
                    if row_index < len(descriptions):
                        row[field] = descriptions[row_index]
                    else:
                        row[field] = ''

                all_rows.append(row)

        columns = ['Disease'] + self.key_fields
        df_result = pd.DataFrame(all_rows, columns=columns)

        logger.info(f"生成表格: {len(processed_data)} 个Disease, {len(all_rows)} 行数据")
        return df_result

    def run(self, input_file, output_file=None):
        """运行快速去重和格式化流程"""
        start_time = time.time()

        logger.info("=" * 80)
        logger.info("开始快速Disease描述去重和格式化")
        logger.info("=" * 80)

        try:
            # 读取原始数据
            logger.info(f"读取文件: {input_file}")
            df = pd.read_excel(input_file)
            logger.info(f"原始数据: {len(df)} 条描述记录")

            # 统计原始数据
            disease_count = df['Disease'].nunique()
            field_stats = df['来源字段'].value_counts()
            logger.info(f"包含 {disease_count} 个Disease")

            # 快速处理数据去重
            processed_data = self.process_disease_data(df)

            # 格式化为纵向表格
            df_formatted = self.format_to_vertical_table(processed_data)

            # 生成输出文件名
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"result_{timestamp}.xlsx"

            # 保存结果
            df_formatted.to_excel(output_file, index=False)

            # 统计信息
            end_time = time.time()
            processing_time = end_time - start_time

            # 计算去重效果
            total_original = len(df)
            total_after_dedup = sum(len(descriptions) for field_data in processed_data.values()
                                    for descriptions in field_data.values())
            dedup_rate = (total_original - total_after_dedup) / total_original * 100

            logger.info("=" * 80)
            logger.info("📊 快速去重完成统计")
            logger.info("=" * 80)
            logger.info(f"处理时间: {processing_time:.2f} 秒 ⚡")
            logger.info(f"API调用次数: {self.api_calls}")
            logger.info(f"缓存命中次数: {self.cache_hits}")
            logger.info(f"规则处理字段数: {self.rule_processed}")
            logger.info(f"LLM处理字段数: {self.llm_processed}")
            logger.info(f"规则处理率: {self.rule_processed / (self.rule_processed + self.llm_processed) * 100:.1f}%")
            logger.info(f"原始描述数: {total_original}")
            logger.info(f"去重后描述数: {total_after_dedup}")
            logger.info(f"去重率: {dedup_rate:.1f}%")
            logger.info(f"输出文件: {output_file}")
            logger.info(f"输出格式: {len(processed_data)} 个Disease, {len(df_formatted)} 行")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"程序执行出错: {str(e)}")
            raise


def main():
    """主函数"""
    # Azure OpenAI配置
    AZURE_API_KEY = ""
    AZURE_ENDPOINT = ""
    DEPLOYMENT_NAME = "o3"

    # 输入输出文件
    input_file = "Disease描述库_优化版_20250806_121343.xlsx"

    # 创建快速去重工具并运行
    deduplicator = FastDescriptionDeduplicator(
        azure_api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        deployment_name=DEPLOYMENT_NAME
    )

    # 运行快速去重和格式化
    deduplicator.run(input_file)


if __name__ == "__main__":
    main()
