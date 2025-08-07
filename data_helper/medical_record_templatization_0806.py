import pandas as pd
import json
import logging
from datetime import datetime
from collections import defaultdict
from openai import AzureOpenAI
import time
import re
import os
import hashlib
"""
正式版本，跑出来的数据为：Disease描述库_优化版_20250806_121343.xlsx

使用该版本先去重，然后再整理格式即可交付。
"""
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedMedicalDescriptionExtractor:
    """
    优化版医疗描述提取器
    - 大幅减少API调用次数
    - 增加中间结果缓存
    - 批量处理提高效率
    """

    def __init__(self, azure_api_key, azure_endpoint, deployment_name="o3", cache_dir="cache"):
        """
        初始化描述提取器

        Args:
            azure_api_key (str): Azure OpenAI API密钥
            azure_endpoint (str): Azure OpenAI端点
            deployment_name (str): 部署名称，默认o3
            cache_dir (str): 缓存目录
        """
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

        # 定义8个关键医疗字段
        self.key_fields = ['主诉', '现病史', '既往史', '辅助检查', 'PE/检查', '病机', '治则/处理', '医嘱']

        # API调用统计
        self.api_calls = 0
        self.cache_hits = 0

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

    def call_azure_api(self, prompt, max_retries=3):
        """
        调用Azure OpenAI API（带缓存）
        """
        # 生成缓存键
        cache_key = self.get_cache_key(prompt)
        cache_file = "api_responses.json"
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
                         "content": "你是一个专业的医疗信息处理专家，精通中医和西医理论，擅长提取和标准化医疗描述。"},
                        {"role": "user", "content": prompt}
                    ],
                )

                # 检查响应是否为空
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

    def batch_extract_diseases(self, diagnosis_list):
        """
        批量提取Disease（只保留病名，跳过所有证型）

        Args:
            diagnosis_list (list): 诊断文本列表

        Returns:
            dict: {诊断文本: [Disease列表]}
        """
        cache_file = "disease_extraction.json"
        cache_data = self.load_cache(cache_file)

        # 检查哪些需要处理
        to_process = []
        results = {}

        for diagnosis in diagnosis_list:
            if not diagnosis or pd.isna(diagnosis):
                results[diagnosis] = []
                continue

            cache_key = self.get_cache_key(diagnosis)
            if cache_key in cache_data:
                results[diagnosis] = cache_data[cache_key]
                self.cache_hits += 1
            else:
                to_process.append(diagnosis)

        # 批量处理未缓存的诊断
        if to_process:
            logger.info(f"批量提取Disease: {len(to_process)} 个诊断")

            # 构建批量处理的prompt
            diagnosis_text = '\n'.join([f"{i + 1}. {diag}" for i, diag in enumerate(to_process)])

            prompt = f"""
请分析以下{len(to_process)}个中医诊断内容，为每个诊断提取出独立的Disease：

诊断列表：
{diagnosis_text}

提取规则：
1. **只保留病名**：如"颈椎病"、"虚劳病"、"胃胀病"、"偏头痛"、"腰肌劳损"等
2. **完全跳过证型**：任何包含"证"字的内容都不要提取，如"气滞血瘀证"、"痰湿内蕴证"等
3. **清理标注**：去除"(可选词：...)"、"[...]"等所有标注信息
4. **中西医病名都保留**：中医病名如"颈椎病"、"虚劳病"，西医病名如"偏头痛"、"腰肌劳损"

示例：
输入：颈椎病(可选词：项痹),气滞血瘀证,虚劳病,气血不足证
输出：["颈椎病", "虚劳病"]  （跳过所有证型）

请返回JSON格式，键为诊断编号(1,2,3...)，值为对应的Disease列表：
{{
  "1": ["颈椎病", "虚劳病"],
  "2": ["偏头痛"],
  "3": ["腰肌劳损"]
}}
"""

            response = self.call_azure_api(prompt)
            if response:
                try:
                    batch_results = json.loads(response)
                    for i, diagnosis in enumerate(to_process):
                        key = str(i + 1)
                        if key in batch_results:
                            diseases = [d.strip() for d in batch_results[key] if d.strip()]
                            results[diagnosis] = diseases
                            # 保存到缓存
                            cache_key = self.get_cache_key(diagnosis)
                            cache_data[cache_key] = diseases
                        else:
                            results[diagnosis] = []

                    # 保存缓存
                    self.save_cache(cache_file, cache_data)

                except json.JSONDecodeError:
                    logger.warning("批量Disease提取JSON解析失败，使用备用方案")
                    # 备用方案：逐个处理
                    for diagnosis in to_process:
                        results[diagnosis] = self._single_extract_disease(diagnosis)

        return results

    def _single_extract_disease(self, diagnosis_text):
        """单个诊断的Disease提取（备用方案，只保留病名，跳过证型）"""
        if not diagnosis_text or pd.isna(diagnosis_text):
            return []

        diseases = []
        parts = diagnosis_text.split(',')

        for part in parts:
            part = part.strip()
            # 清理标注
            part = re.sub(r'\(可选词：[^)]+\)', '', part)
            part = re.sub(r'\[[^\]]+\]', '', part)
            part = part.strip()

            if not part:
                continue

            # 跳过所有包含"证"的内容
            if '证' in part:
                continue

            # 保留病名
            if part:
                diseases.append(part)

        return diseases

    def batch_extract_descriptions(self, field_contents_by_field):
        """
        批量提取描述（按字段分组处理）

        Args:
            field_contents_by_field (dict): {字段名: [内容列表]}

        Returns:
            dict: {字段名: {内容: [描述列表]}}
        """
        cache_file = "description_extraction.json"
        cache_data = self.load_cache(cache_file)

        results = {}

        for field_name, contents in field_contents_by_field.items():
            logger.info(f"批量提取 '{field_name}' 字段描述: {len(contents)} 个内容")

            field_results = {}
            to_process = []

            # 检查缓存
            for content in contents:
                if not content or pd.isna(content) or str(content).strip() in ['', '无', '否认', '-']:
                    field_results[content] = []
                    continue

                cache_key = f"{field_name}:{self.get_cache_key(str(content))}"
                if cache_key in cache_data:
                    field_results[content] = cache_data[cache_key]
                    self.cache_hits += 1
                else:
                    to_process.append(content)

            # 批量处理未缓存的内容
            if to_process:
                # 限制批量大小，避免prompt过长
                batch_size = 10
                for i in range(0, len(to_process), batch_size):
                    batch = to_process[i:i + batch_size]
                    batch_results = self._batch_extract_descriptions_for_field(field_name, batch)

                    for content, descriptions in batch_results.items():
                        field_results[content] = descriptions
                        # 保存到缓存
                        cache_key = f"{field_name}:{self.get_cache_key(str(content))}"
                        cache_data[cache_key] = descriptions

                    # 避免API调用过快
                    time.sleep(0.5)

                # 保存缓存
                self.save_cache(cache_file, cache_data)

            results[field_name] = field_results

        return results

    def _batch_extract_descriptions_for_field(self, field_name, contents):
        """为单个字段批量提取描述"""
        contents_text = '\n'.join([f"{i + 1}. {content}" for i, content in enumerate(contents)])

        prompt = f"""
请从以下{len(contents)}个{field_name}内容中批量提取语义完整的医疗描述片段：

{field_name}内容列表：
{contents_text}

提取要求：
1. **语义完整**：每个描述片段必须是完整的、有意义的医疗表述
2. **时间标准化**：将具体时间改为"N天"，如"1月"→"N天"，"数年"→"N天"
3. **保留原始表述**：不要添加额外词汇，保持原文用词
4. **去除冗余**：跳过"否认"、"无"、"正常"等无意义描述
5. **医生可复用**：提取的描述应该是医生问诊中可直接使用的

请返回JSON格式，键为内容编号(1,2,3...)，值为对应的描述列表：
{{
  "1": ["描述1", "描述2"],
  "2": ["描述3"],
  "3": ["描述4", "描述5"]
}}
"""

        response = self.call_azure_api(prompt)
        results = {}

        if response:
            try:
                batch_results = json.loads(response)
                for i, content in enumerate(contents):
                    key = str(i + 1)
                    if key in batch_results and isinstance(batch_results[key], list):
                        descriptions = [d.strip() for d in batch_results[key] if d.strip()]
                        results[content] = descriptions
                    else:
                        results[content] = []
            except json.JSONDecodeError:
                logger.warning(f"批量{field_name}描述提取JSON解析失败")
                # 备用方案：返回空结果
                for content in contents:
                    results[content] = []
        else:
            # API失败，返回空结果
            for content in contents:
                results[content] = []

        return results

    def smart_standardize_descriptions(self, all_descriptions_by_field):
        """
        智能标准化描述（减少API调用）
        """
        cache_file = "standardization.json"
        cache_data = self.load_cache(cache_file)

        standardized_results = {}

        for field_name, descriptions in all_descriptions_by_field.items():
            if not descriptions:
                standardized_results[field_name] = {}
                continue

            logger.info(f"标准化 '{field_name}' 字段: {len(descriptions)} 个描述")

            # 生成字段级别的缓存键
            desc_text = '|'.join(sorted(descriptions))
            cache_key = f"{field_name}:{self.get_cache_key(desc_text)}"

            if cache_key in cache_data:
                standardized_results[field_name] = cache_data[cache_key]
                self.cache_hits += 1
                continue

            # 需要API处理
            if len(descriptions) <= 1:
                # 单个描述无需标准化
                result = {desc: [desc] for desc in descriptions}
            else:
                result = self._api_standardize_descriptions(descriptions)

            standardized_results[field_name] = result
            # 保存到缓存
            cache_data[cache_key] = result
            self.save_cache(cache_file, cache_data)

        return standardized_results

    def _api_standardize_descriptions(self, descriptions_list):
        """使用API标准化描述"""
        descriptions_text = '\n'.join([f"{i + 1}. {desc}" for i, desc in enumerate(descriptions_list)])

        prompt = f"""
请分析以下{len(descriptions_list)}个医疗描述，找出相似的描述并进行标准化合并：

描述列表：
{descriptions_text}

标准化规则：
1. 相似症状合并：如"颈肩不适N天"和"颈项疼痛N天" → "颈[肩/项]不适N天"
2. 重复描述去重：完全相同的描述只保留一个
3. 包含关系处理：如"气滞血瘀"和"气滞血瘀，不通则痛" → 保留简洁版本"气滞血瘀"

请返回JSON格式，键为标准化后的描述，值为对应的原始描述列表：
{{
  "标准化描述1": ["原始描述1", "原始描述2"],
  "标准化描述2": ["原始描述3"]
}}
"""

        response = self.call_azure_api(prompt)
        if response:
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                logger.warning("标准化描述JSON解析失败")

        # 备用方案：简单去重
        unique_descriptions = list(set(descriptions_list))
        return {desc: [desc] for desc in unique_descriptions}

    def run(self, input_file, max_diseases=None):
        """运行优化的描述提取流程"""
        start_time = time.time()

        logger.info("=" * 80)
        logger.info("开始优化版医疗描述提取任务")
        if max_diseases:
            logger.info(f"⚠️  测试模式：仅处理前 {max_diseases} 个Disease")
        logger.info("=" * 80)

        try:
            # 读取数据
            logger.info(f"读取数据文件: {input_file}")
            df = pd.read_excel(input_file)
            logger.info(f"成功读取 {len(df)} 条记录")

            # 第1步：批量提取Disease
            logger.info("📋 步骤1: 批量提取Disease")
            all_diagnoses = df['诊断'].dropna().unique().tolist()
            disease_mapping = self.batch_extract_diseases(all_diagnoses)

            # 构建Disease-记录映射
            disease_records = defaultdict(list)
            for index, row in df.iterrows():
                diagnosis = row.get('诊断', '')
                diseases = disease_mapping.get(diagnosis, [])

                for disease in diseases:
                    record_data = {
                        '患者ID': f"P{index + 1:03d}",
                        '原始诊断': diagnosis
                    }
                    for field in self.key_fields:
                        record_data[field] = row.get(field, '') or ''
                    disease_records[disease].append(record_data)

            logger.info(f"识别出 {len(disease_records)} 个唯一Disease")

            # 限制处理数量
            if max_diseases:
                disease_items = list(disease_records.items())[:max_diseases]
                disease_records = dict(disease_items)
                logger.info(f"限制处理 {len(disease_records)} 个Disease")

            # 第2步：收集所有字段内容
            logger.info("📋 步骤2: 收集字段内容")
            field_contents = defaultdict(set)

            for records in disease_records.values():
                for record in records:
                    for field in self.key_fields:
                        content = record.get(field, '')
                        if content and str(content).strip() not in ['', '无', '否认', '-']:
                            field_contents[field].add(str(content))

            # 转换为列表
            field_contents_list = {field: list(contents) for field, contents in field_contents.items()}

            # 第3步：批量提取描述
            logger.info("📋 步骤3: 批量提取描述")
            description_mapping = self.batch_extract_descriptions(field_contents_list)

            # 第4步：收集所有描述并标准化
            logger.info("📋 步骤4: 标准化描述")
            all_descriptions_by_field = defaultdict(list)

            for field_name, content_desc_map in description_mapping.items():
                for descriptions in content_desc_map.values():
                    all_descriptions_by_field[field_name].extend(descriptions)

            # 去重
            for field in all_descriptions_by_field:
                all_descriptions_by_field[field] = list(set(all_descriptions_by_field[field]))

            # 批量标准化
            standardized_mapping = self.smart_standardize_descriptions(all_descriptions_by_field)

            # 第5步：生成最终结果
            logger.info("📋 步骤5: 生成最终结果")
            final_results = []

            for disease, records in disease_records.items():
                logger.info(f"处理Disease: {disease} ({len(records)} 条记录)")

                # 统计该Disease的所有描述
                disease_descriptions = defaultdict(lambda: defaultdict(list))

                for record in records:
                    for field in self.key_fields:
                        content = record.get(field, '')
                        if content and field in description_mapping:
                            descriptions = description_mapping[field].get(content, [])
                            for desc in descriptions:
                                # 找到标准化后的描述
                                standard_desc = desc
                                if field in standardized_mapping:
                                    for std_desc, orig_list in standardized_mapping[field].items():
                                        if desc in orig_list:
                                            standard_desc = std_desc
                                            break

                                disease_descriptions[field][standard_desc].append(record['患者ID'])

                # 生成结果
                for field, desc_patients in disease_descriptions.items():
                    for desc, patient_ids in desc_patients.items():
                        unique_patients = list(set(patient_ids))
                        final_results.append({
                            'Disease': disease,
                            '描述内容': desc,
                            '来源字段': field,
                            '出现次数': len(patient_ids),
                            '样本患者数': len(unique_patients),
                            '样本患者ID': ','.join(unique_patients[:10])
                        })

            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if max_diseases:
                output_file = f"Disease描述库_优化测试{max_diseases}个_{timestamp}.xlsx"
            else:
                output_file = f"Disease描述库_优化版_{timestamp}.xlsx"

            df_results = pd.DataFrame(final_results)
            df_results.to_excel(output_file, index=False)

            # 统计信息
            end_time = time.time()
            processing_time = end_time - start_time

            logger.info("=" * 80)
            logger.info("📊 处理完成统计")
            logger.info("=" * 80)
            logger.info(f"处理时间: {processing_time:.2f} 秒")
            logger.info(f"API调用次数: {self.api_calls}")
            logger.info(f"缓存命中次数: {self.cache_hits}")
            logger.info(f"缓存命中率: {self.cache_hits / (self.api_calls + self.cache_hits) * 100:.1f}%")
            logger.info(f"处理Disease数量: {len(disease_records)}")
            logger.info(f"生成描述总数: {len(final_results)}")
            logger.info(f"输出文件: {output_file}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"程序执行出错: {str(e)}")
            raise


def main():
    """主函数"""
    # Azure OpenAI配置
    AZURE_API_KEY = ""
    AZURE_ENDPOINT = ""  # 修正后的端点
    DEPLOYMENT_NAME = "o3"

    # 输入文件路径
    input_file = "../data/case_data/病历数据_可使用_20250804_172720.xlsx"

    # 测试配置
    MAX_DISEASES = None  # 改为None可处理全部Disease

    # 创建提取器并运行
    extractor = OptimizedMedicalDescriptionExtractor(
        azure_api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        deployment_name=DEPLOYMENT_NAME
    )

    # 运行处理流程
    extractor.run(input_file, max_diseases=MAX_DISEASES)


if __name__ == "__main__":
    main()
