import json
import logging
import os
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import pandas as pd
from openai import AzureOpenAI

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Step2LLMValidator:
    """
    Step2: LLM医疗描述合理性验证器
    验证PE/检查字段内容，以及中医疾病对应的主诉、现病史、病机、治则/处理是否合理
    """

    def __init__(self, config: Dict):
        """
        初始化验证器

        Args:
            config: 配置字典
        """
        # Azure OpenAI配置
        azure_config = config.get('azure_openai', {})
        self.client = AzureOpenAI(
            api_key=azure_config.get('api_key', ''),
            api_version=azure_config.get('api_version', '2025-01-01-preview'),
            azure_endpoint=azure_config.get('endpoint', '')
        )
        self.deployment_name = azure_config.get('deployment_name', 'o3')

        # 加载配置
        self.pe_invalid_keywords = config.get('pe_invalid_keywords', [])
        self.knowledge_base = self._load_knowledge_base(config.get('知识库文件', ''))
        self.disease_mapping = self._load_disease_mapping(config.get('门诊疾病->知识库疾病映射', ''))

        # 验证字段
        self.validation_fields = ['主诉', '现病史', '病机', '治则/处理']

        # 统计信息
        self.api_calls = 0
        self.failed_calls = 0

    def _load_knowledge_base(self, file_path: str) -> Dict:
        """加载知识库文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 构建疾病名到知识的索引
            disease_knowledge = {}

            if 'book' in data and 'disciplines' in data['book']:
                for discipline in data['book']['disciplines']:
                    for item in discipline.get('items', []):
                        disease_name = item.get('name', '')
                        if disease_name:
                            disease_knowledge[disease_name] = {
                                'sections': item.get('sections', {}),
                                'discipline': discipline.get('name', ''),
                                'number': item.get('number', '')
                            }

            logger.info(f"成功加载知识库: {len(disease_knowledge)} 个疾病")
            return disease_knowledge

        except Exception as e:
            logger.error(f"加载知识库失败: {str(e)}")
            return {}

    def _load_disease_mapping(self, file_path: str) -> Dict:
        """加载疾病映射文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            logger.info(f"成功加载疾病映射: {len(mapping)} 条映射关系")
            return mapping
        except Exception as e:
            logger.error(f"加载疾病映射失败: {str(e)}")
            return {}

    def validate_pe_examination(self, pe_content: str) -> Tuple[bool, str]:
        """
        验证PE/检查字段内容

        Args:
            pe_content: PE/检查字段内容

        Returns:
            (是否合格, 不合格原因)
        """
        if not pe_content or pd.isna(pe_content):
            return False, "PE/检查字段为空"

        content_str = str(pe_content).strip()
        if not content_str:
            return False, "PE/检查字段为空"

        # 检查是否包含无效关键词
        for keyword in self.pe_invalid_keywords:
            if keyword == content_str:
                return False, f"PE/检查内容为'{content_str}'，命中无效关键词'{keyword}'"

        return True, ""

    def map_disease_to_knowledge(self, tcm_diseases: List[str]) -> Tuple[List[str], List[Dict]]:
        """
        将门诊中医疾病映射到知识库疾病

        Args:
            tcm_diseases: 门诊中医疾病列表

        Returns:
            (成功映射的知识库疾病列表, 映射失败记录列表)
        """
        mapped_diseases = []
        failed_mappings = []

        for disease in tcm_diseases:
            # 先尝试映射
            mapped_disease = self.disease_mapping.get(disease, disease)

            # 检查映射后的疾病是否在知识库中
            if mapped_disease in self.knowledge_base:
                mapped_diseases.append(mapped_disease)
                logger.debug(f"疾病映射成功: {disease} -> {mapped_disease}")
            else:
                # 映射失败
                if disease in self.disease_mapping:
                    reason = f"映射为'{mapped_disease}'，但知识库中未找到"
                else:
                    reason = "映射表中未找到，知识库中也未找到"

                failed_mappings.append({
                    "原疾病": disease,
                    "映射后": mapped_disease if disease in self.disease_mapping else "未找到",
                    "是否匹配": False,
                    "失败原因": reason
                })
                logger.warning(f"疾病映射失败: {disease} - {reason}")

        return mapped_diseases, failed_mappings

    def extract_disease_knowledge(self, diseases: List[str]) -> str:
        """
        提取疾病的知识库信息

        Args:
            diseases: 知识库疾病名列表

        Returns:
            格式化的知识库内容
        """
        knowledge_parts = []

        for disease in diseases:
            if disease not in self.knowledge_base:
                continue

            disease_info = self.knowledge_base[disease]
            sections = disease_info.get('sections', {})

            disease_knowledge = [f"【{disease}】"]

            # 按顺序提取各个sections
            section_order = ['正文', '诊断依据', '证候分类', '治疗方案', '其他疗法']

            for section_name in section_order:
                if section_name in sections:
                    section_content = sections[section_name]
                    if isinstance(section_content, list):
                        content = '\n'.join(section_content)
                    else:
                        content = str(section_content)

                    if content.strip():
                        disease_knowledge.append(f"  {section_name}:\n    {content}")

            knowledge_parts.append('\n'.join(disease_knowledge))

        return '\n\n'.join(knowledge_parts)

    def call_azure_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """
        调用Azure OpenAI API

        Args:
            prompt: 提示词
            max_retries: 最大重试次数

        Returns:
            API响应内容，失败返回None
        """
        for attempt in range(max_retries):
            try:
                self.api_calls += 1
                logger.debug(f"调用Azure API (尝试 {attempt + 1}/{max_retries})")

                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一位资深的中医专家，精通中医理论和临床实践。请基于提供的标准知识库内容，客观评估病历描述的合理性。注意要结合临床实际情况，不要过于严格。"
                        },
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )

                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    result = response.choices[0].message.content.strip()
                    logger.debug("API调用成功")
                    return result
                else:
                    logger.warning(f"API返回空响应 (尝试 {attempt + 1}/{max_retries})")

            except Exception as e:
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避

        self.failed_calls += 1
        logger.error("API调用最终失败")
        return None

    def validate_tcm_diseases_with_llm(self, record_data: Dict, mapped_diseases: List[str]) -> Dict:
        """
        使用LLM验证中医疾病相关字段的合理性

        Args:
            record_data: 病历数据
            mapped_diseases: 映射成功的知识库疾病列表

        Returns:
            验证结果字典
        """
        if not mapped_diseases:
            return {}

        # 提取病历中的相关字段
        patient_info = {
            '主诉': record_data.get('主诉', ''),
            '现病史': record_data.get('现病史', ''),
            '病机': record_data.get('病机', ''),
            '治则/处理': record_data.get('治则/处理', '')
        }

        # 检查空字段
        validation_results = {}
        for field_name in self.validation_fields:
            field_content = patient_info.get(field_name, '')
            if not field_content or pd.isna(field_content) or not str(field_content).strip():
                validation_results[f"{field_name}验证"] = {
                    "结果": "不合理",
                    "原因": "内容为空"
                }

        # 如果所有字段都为空，直接返回
        non_empty_fields = [field for field in self.validation_fields
                            if patient_info.get(field, '') and not pd.isna(patient_info.get(field, ''))
                            and str(patient_info.get(field, '')).strip()]

        if not non_empty_fields:
            return validation_results

        # 提取知识库信息
        knowledge_content = self.extract_disease_knowledge(mapped_diseases)

        # 构建LLM提示词
        diseases_str = '、'.join(mapped_diseases)
        patient_info_str = []

        for field_name in self.validation_fields:
            content = patient_info.get(field_name, '')
            if content and not pd.isna(content) and str(content).strip():
                patient_info_str.append(f"- {field_name}: {content}")

        prompt = f"""
请基于以下标准中医知识库内容，验证患者病历中各字段描述的合理性。

【患者中医疾病】: {diseases_str}

【患者病历信息】:
{chr(10).join(patient_info_str)}

【标准知识库内容】:
{knowledge_content}

【验证要求】:
1. 请分别验证以下字段的合理性：主诉、现病史、病机、治则/处理
2. 判断标准：结合中医理论和临床实际，不要过于严格
3. 考虑疾病的多样性表现和个体差异
4. 只要描述在医学上说得通就算合理

【输出格式】:
请严格按照以下JSON格式返回，不要添加任何其他内容：
{{
  "主诉验证": {{"结果": "合理/不合理", "原因": "具体原因"}},
  "现病史验证": {{"结果": "合理/不合理", "原因": "具体原因"}},
  "病机验证": {{"结果": "合理/不合理", "原因": "具体原因"}},
  "治则/处理验证": {{"结果": "合理/不合理", "原因": "具体原因"}}
}}
"""

        # 调用LLM
        response = self.call_azure_api(prompt)

        if response:
            try:
                # 清理响应内容
                cleaned_response = response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()

                llm_results = json.loads(cleaned_response)

                # 合并结果（LLM结果优先，空字段结果补充）
                for field_name in self.validation_fields:
                    field_key = f"{field_name}验证"
                    if field_key in llm_results:
                        validation_results[field_key] = llm_results[field_key]
                    elif field_key not in validation_results:
                        # LLM没有返回结果，且之前没有标记为空
                        validation_results[field_key] = {
                            "结果": "验证失败",
                            "原因": "LLM验证失败"
                        }

                logger.debug(f"LLM验证成功，疾病: {diseases_str}")
                return validation_results

            except json.JSONDecodeError as e:
                logger.error(f"LLM响应JSON解析失败: {str(e)}, 响应内容: {response[:200]}...")

        # API调用失败或解析失败的处理
        for field_name in self.validation_fields:
            field_key = f"{field_name}验证"
            if field_key not in validation_results:
                validation_results[field_key] = {
                    "结果": "验证失败",
                    "原因": "LLM调用失败"
                }

        return validation_results

    def validate_single_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证单条病历记录

        Args:
            record: 包含step1结果的病历记录

        Returns:
            添加了step2验证结果的记录
        """
        original_data = record.get('原始数据', {})
        diagnosis_classification = record.get('诊断分类', {})

        # Step2验证结果初始化
        step2_result = {
            "PE检查是否合格": True,
            "PE检查不合格原因": "",
            "中医疾病验证结果": {},
            "映射失败记录": []
        }

        # 1. 验证PE/检查字段
        pe_content = original_data.get('PE/检查', '') or original_data.get('PE/检查 （体现望闻问切）', '')
        pe_valid, pe_reason = self.validate_pe_examination(pe_content)
        step2_result["PE检查是否合格"] = pe_valid
        step2_result["PE检查不合格原因"] = pe_reason

        # 2. 验证中医疾病相关字段
        tcm_diseases = diagnosis_classification.get('中医疾病', [])

        if tcm_diseases:
            # 映射疾病名
            mapped_diseases, failed_mappings = self.map_disease_to_knowledge(tcm_diseases)
            step2_result["映射失败记录"] = failed_mappings

            # 对成功映射的疾病进行LLM验证
            if mapped_diseases:
                try:
                    validation_results = self.validate_tcm_diseases_with_llm(original_data, mapped_diseases)

                    # 构建结果格式
                    tcm_validation = {}
                    for original_disease in tcm_diseases:
                        mapped_disease = self.disease_mapping.get(original_disease, original_disease)
                        if mapped_disease in mapped_diseases:
                            mapping_status = f"成功映射为: {mapped_disease}" if original_disease != mapped_disease else "直接匹配"
                            tcm_validation[original_disease] = {
                                "映射状态": mapping_status,
                                **validation_results
                            }

                    step2_result["中医疾病验证结果"] = tcm_validation

                except Exception as e:
                    logger.error(f"中医疾病验证过程出错: {str(e)}")
                    # 为每个原始疾病添加验证失败记录
                    tcm_validation = {}
                    for original_disease in tcm_diseases:
                        tcm_validation[original_disease] = {
                            "映射状态": "验证过程出错",
                            "主诉验证": {"结果": "验证失败", "原因": "系统错误"},
                            "现病史验证": {"结果": "验证失败", "原因": "系统错误"},
                            "病机验证": {"结果": "验证失败", "原因": "系统错误"},
                            "治则/处理验证": {"结果": "验证失败", "原因": "系统错误"}
                        }
                    step2_result["中医疾病验证结果"] = tcm_validation

        # 添加step2结果到原记录
        record['step2验证结果'] = step2_result
        return record

    def validate_step1_results(self, step1_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        验证step1结果中的所有记录

        Args:
            step1_results: step1的验证结果列表

        Returns:
            添加了step2验证结果的记录列表
        """
        logger.info(f"开始Step2 LLM验证，共 {len(step1_results)} 条记录")

        results = []

        for i, record in enumerate(step1_results, 1):
            try:
                logger.info(f"验证进度: {i}/{len(step1_results)} - 记录编号: {record.get('记录编号', i)}")

                validated_record = self.validate_single_record(record)
                results.append(validated_record)

                # 避免API调用过快
                if i % 5 == 0:
                    time.sleep(1)

            except Exception as e:
                logger.error(f"验证记录 {i} 时出错: {str(e)}")
                # 添加验证失败的记录
                record['step2验证结果'] = {
                    "PE检查是否合格": False,
                    "PE检查不合格原因": "验证过程出错",
                    "中医疾病验证结果": {},
                    "映射失败记录": [],
                    "系统错误": str(e)
                }
                results.append(record)

        # 统计结果
        pe_pass_count = len([r for r in results if r.get('step2验证结果', {}).get('PE检查是否合格', False)])
        tcm_validated_count = len([r for r in results if r.get('step2验证结果', {}).get('中医疾病验证结果', {})])

        logger.info("=" * 60)
        logger.info("Step2 LLM验证统计:")
        logger.info(f"  总记录数: {len(results)}")
        logger.info(f"  PE检查合格: {pe_pass_count}")
        logger.info(f"  PE检查不合格: {len(results) - pe_pass_count}")
        logger.info(f"  包含中医疾病验证: {tcm_validated_count}")
        logger.info(f"  API调用总次数: {self.api_calls}")
        logger.info(f"  API调用失败次数: {self.failed_calls}")
        logger.info("=" * 60)

        return results


def load_step1_results(output_dir: str) -> List[Dict[str, Any]]:
    """
    加载step1的最新验证结果

    Args:
        output_dir: 输出目录

    Returns:
        step1验证结果列表
    """
    latest_file = os.path.join(output_dir, "step1_results_latest.json")

    if not os.path.exists(latest_file):
        raise FileNotFoundError(f"未找到step1结果文件: {latest_file}")

    logger.info(f"加载step1结果文件: {latest_file}")

    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"成功加载step1结果: {len(data)} 条记录")
        return data

    except Exception as e:
        logger.error(f"加载step1结果失败: {str(e)}")
        raise


def run_step2_validation(config: Dict, step1_results: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    运行Step2 LLM验证的主函数

    Args:
        config: 配置字典
        step1_results: step1验证结果（流水线传递）

    Returns:
        step2验证结果列表
    """
    logger.info("=" * 80)
    logger.info("开始执行 Step2: LLM医疗描述合理性验证")
    logger.info("=" * 80)

    # 获取step1结果
    if step1_results is not None:
        logger.info("使用流水线传递的step1结果")
        data = step1_results
    else:
        logger.info("从文件加载step1结果")
        data = load_step1_results(config.get('output_dir', '../../data/result/'))

    # 创建验证器并执行验证
    validator = Step2LLMValidator(config)
    results = validator.validate_step1_results(data)

    logger.info("Step2 LLM验证执行完成")
    return results


if __name__ == "__main__":
    # 独立运行时的测试代码
    config_file = "config.json"

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        results = run_step2_validation(config)

        logger.info(f"Step2验证完成，共处理 {len(results)} 条记录")

        # 保存测试结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_output = f"step2_test_results_{timestamp}.json"

        with open(test_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"测试结果已保存: {test_output}")

    except Exception as e:
        logger.error(f"Step2验证执行失败: {str(e)}")
