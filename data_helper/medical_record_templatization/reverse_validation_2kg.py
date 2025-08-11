import pandas as pd
import json
import logging
import asyncio
from datetime import datetime
from collections import defaultdict
from openai import AzureOpenAI
import time
import os
import hashlib
from tqdm import tqdm
from difflib import SequenceMatcher
import re
import threading
"""
根据《诊疗规范》再跑一版反正验证的结果，这一版没有置信度，要给出不通过的理由。

主要结果文件: ../data/case_data\反向验证2kg_20250810_230721.xlsx
过程记录文件: ../data/case_data\验证过程记录_20250810_230721.xlsx
输出格式说明:
主要结果: Disease | 字段 | 描述内容 | 是否通过 | 不通过理由
过程记录: 包含疾病匹配、知识库使用、完整prompt等详细信息


结果：
处理疾病数: 176
验证描述总数: 3197
通过: 1415 (44.3%)
不通过: 1782 (55.7%)
"""
# 配置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('reverse_validation_2kg.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class KnowledgeBasedValidator:
    """
    基于知识库的医疗描述审核系统
    支持详细日志记录、中间结果追溯
    """

    def __init__(self, azure_api_key, azure_endpoint, deployment_name="o3",
                 cache_dir="cache", knowledge_base_path="../../data/other/book_structured_enhanced.json"):
        """
        初始化审核系统
        """
        logger.info("🚀 初始化基于知识库的医疗描述审核系统")

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
        logger.info(f"📁 缓存目录: {cache_dir}")

        # 定义医疗字段
        self.key_fields = ['主诉', '现病史', '既往史', '辅助检查', 'PE/检查 （体现望闻问切）', '病机', '治则/处理',
                           '医嘱']

        # 统计信息
        self.api_calls = 0
        self.cache_hits = 0
        self.processing_records = []  # 中间结果记录

        # 加载知识库
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)

    def load_knowledge_base(self, file_path):
        """
        加载并预处理知识库
        """
        logger.info(f"📚 开始加载知识库: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_kb = json.load(f)

            # 预处理知识库，建立疾病索引
            processed_kb = {}
            disease_count = 0

            if 'book' in raw_kb and 'disciplines' in raw_kb['book']:
                for discipline in raw_kb['book']['disciplines']:
                    discipline_name = discipline.get('name', '未知科室')
                    logger.info(f"  处理科室: {discipline_name}")

                    for item in discipline.get('items', []):
                        disease_name = item.get('name', '')
                        if disease_name:
                            processed_kb[disease_name] = {
                                'discipline': discipline_name,
                                'sections': item.get('sections', {}),
                                'number': item.get('number', '')
                            }
                            disease_count += 1

            logger.info(f"✅ 知识库加载完成: {disease_count} 个疾病")

            # 输出疾病列表
            diseases = list(processed_kb.keys())
            logger.info(f"📋 知识库疾病列表(前10个): {diseases[:10]}")

            return processed_kb

        except Exception as e:
            logger.error(f"❌ 知识库加载失败: {str(e)}")
            return {}

    def normalize_disease_name(self, disease_name):
        """
        标准化疾病名称
        """
        if not disease_name:
            return ""

        # 移除常见后缀和特殊字符
        normalized = disease_name.strip()
        normalized = re.sub(r'[（）\(\)\[\]【】\s]+', '', normalized)

        # 移除常见前缀（如：混合型、慢性、急性等）
        prefixes = ['混合型', '慢性', '急性', '复发性', '原发性', '继发性', '顽固性']
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break

        # 移除常见后缀
        suffixes = ['病', '症', '证', '综合征', '综合症']
        for suffix in suffixes:
            if normalized.endswith(suffix) and len(normalized) > len(suffix):
                normalized = normalized[:-len(suffix)]
                break

        return normalized.lower()

    def match_disease_to_knowledge(self, excel_disease):
        """
        将Excel中的疾病名匹配到知识库
        """
        logger.debug(f"🔍 开始匹配疾病: {excel_disease}")

        kb_diseases = list(self.knowledge_base.keys())

        # 1. 精确匹配
        if excel_disease in kb_diseases:
            logger.debug(f"✅ 精确匹配成功: {excel_disease}")
            return excel_disease, "精确匹配", 1.0

        # 2. 标准化后匹配
        normalized_excel = self.normalize_disease_name(excel_disease)
        for kb_disease in kb_diseases:
            if normalized_excel == self.normalize_disease_name(kb_disease):
                logger.debug(f"✅ 标准化匹配成功: {excel_disease} -> {kb_disease}")
                return kb_disease, "标准化匹配", 0.9

        # 3. 包含关系匹配
        best_match = None
        best_score = 0
        for kb_disease in kb_diseases:
            # 检查包含关系
            if normalized_excel in self.normalize_disease_name(kb_disease):
                score = len(normalized_excel) / len(self.normalize_disease_name(kb_disease))
                if score > best_score:
                    best_match = kb_disease
                    best_score = score
            elif self.normalize_disease_name(kb_disease) in normalized_excel:
                score = len(self.normalize_disease_name(kb_disease)) / len(normalized_excel)
                if score > best_score:
                    best_match = kb_disease
                    best_score = score

        if best_match and best_score > 0.5:
            logger.debug(f"✅ 包含匹配成功: {excel_disease} -> {best_match} (得分: {best_score:.2f})")
            return best_match, "包含匹配", best_score

        # 4. 相似度匹配
        best_match = None
        best_score = 0
        for kb_disease in kb_diseases:
            similarity = SequenceMatcher(None, normalized_excel,
                                         self.normalize_disease_name(kb_disease)).ratio()
            if similarity > best_score:
                best_match = kb_disease
                best_score = similarity

        if best_match and best_score > 0.6:
            logger.debug(f"✅ 相似度匹配成功: {excel_disease} -> {best_match} (得分: {best_score:.2f})")
            return best_match, "相似度匹配", best_score

        # 匹配失败
        logger.warning(f"❌ 疾病匹配失败: {excel_disease}")
        return None, "匹配失败", 0.0

    def extract_relevant_knowledge(self, disease_name, field_type):
        """
        根据字段类型提取相关知识
        """
        if disease_name not in self.knowledge_base:
            return "", []

        disease_info = self.knowledge_base[disease_name]
        sections = disease_info.get('sections', {})

        # 字段与知识库章节的映射关系
        field_mapping = {
            "主诉": ["正文", "诊断依据"],
            "现病史": ["正文", "诊断依据", "证候分类"],
            "既往史": ["诊断依据", "并发症处理"],
            "辅助检查": ["诊断依据"],
            "PE/检查 （体现望闻问切）": ["诊断依据", "证候分类"],
            "病机": ["证候分类"],
            "治则/处理": ["治疗方案", "其他疗法"],
            "医嘱": ["其他疗法", "并发症处理"]
        }

        relevant_sections = field_mapping.get(field_type, ["正文"])
        knowledge_parts = []
        used_sections = []

        for section_name in relevant_sections:
            if section_name in sections:
                section_content = sections[section_name]
                if isinstance(section_content, list):
                    content = '\n'.join(section_content)
                else:
                    content = str(section_content)

                if content.strip():
                    knowledge_parts.append(f"【{section_name}】\n{content}")
                    used_sections.append(section_name)

        knowledge_text = '\n\n'.join(knowledge_parts)
        logger.debug(f"📖 提取知识 - 疾病: {disease_name}, 字段: {field_type}, 使用章节: {used_sections}")

        return knowledge_text, used_sections

    def clean_json_response(self, response_text):
        """
        清理和修复API返回的JSON响应
        """
        if not response_text:
            return response_text

        # 移除可能的markdown代码块标记
        response_text = re.sub(r'```json\s*\n?', '', response_text)
        response_text = re.sub(r'```\s*$', '', response_text)

        # 处理多行字符串中的换行符
        def escape_newlines_in_strings(match):
            content = match.group(1)
            # 转义字符串内的换行符和引号
            content = content.replace('\n', '\\n').replace('\r', '\\r').replace('"', '\\"')
            return f'"{content}"'

        # 修复字符串值中的换行符
        response_text = re.sub(r'"([^"]*(?:\n[^"]*)*)"', escape_newlines_in_strings, response_text)

        # 移除注释（如果有）
        response_text = re.sub(r'//.*$', '', response_text, flags=re.MULTILINE)

        return response_text.strip()

    def parse_response_backup(self, response, descriptions):
        """
        备用响应解析方法
        """
        try:
            results = []

            # 查找通过/不通过的模式
            for i, description in enumerate(descriptions, 1):
                # 查找编号对应的结果
                pattern = rf'"{i}".*?"result".*?"(通过|不通过)".*?"reason".*?"([^"]*)"'
                match = re.search(pattern, response, re.DOTALL)

                if match:
                    result = match.group(1)
                    reason = match.group(2)
                    results.append({
                        'result': result,
                        'reason': reason
                    })
                else:
                    # 找不到匹配，默认通过
                    results.append({
                        'result': '通过',
                        'reason': '备用解析方法：未找到明确结果，默认通过'
                    })

            return results if len(results) == len(descriptions) else None

        except Exception as e:
            logger.error(f"备用解析方法也失败: {str(e)}")
            return None

    def validate_single_disease_batch(self, disease_data):
        """
        验证单个疾病的所有描述
        """
        disease = disease_data['disease']
        items = disease_data['items']

        logger.info(f"🔬 开始验证疾病: {disease} ({len(items)} 条描述)")

        # 疾病名匹配
        matched_disease, match_method, match_score = self.match_disease_to_knowledge(disease)

        if not matched_disease:
            logger.warning(f"⚠️ 疾病 {disease} 未找到匹配的知识库条目")
            # 返回所有描述为不通过
            results = []
            for item in items:
                results.append({
                    'Disease': disease,
                    '字段': item['field'],
                    '描述内容': item['description'],
                    '是否通过': '不通过',
                    '不通过理由': f'知识库中未找到疾病 "{disease}" 的相关信息，建议检查疾病名称或补充知识库'
                })
            return results

        logger.info(f"✅ 疾病匹配成功: {disease} -> {matched_disease} ({match_method}, 得分: {match_score:.2f})")

        # 按字段分组描述
        field_groups = defaultdict(list)
        for item in items:
            field_groups[item['field']].append(item['description'])

        # 为每个字段构建批量验证prompt
        all_results = []
        for field, descriptions in field_groups.items():
            logger.info(f"  🔍 验证字段: {field} ({len(descriptions)} 条描述)")

            # 提取相关知识
            knowledge_text, used_sections = self.extract_relevant_knowledge(matched_disease, field)

            # 构建验证项目列表
            items_text = []
            for i, desc in enumerate(descriptions, 1):
                items_text.append(f"{i}. {desc}")

            # 优化后的prompt，更加宽松合理
            prompt = f"""
你是权威的中医专家，请基于以下标准医学知识库审核医疗描述的合理性。

【审核疾病】: {matched_disease}
【审核字段】: {field}
【标准知识库内容】:
{knowledge_text if knowledge_text else "该字段暂无对应的标准知识库内容，请基于中医理论和临床常识判断"}

【待审核描述列表】:
{chr(10).join(items_text)}

【审核原则】:
1. 如果知识库有明确内容，优先参考知识库
2. 如果知识库内容有限，基于中医理论和临床常识判断
3. 采用相对宽松的标准，只要不是明显错误就通过
4. 考虑中医的整体观念和辨证论治特色

【判断标准】:
- 通过：描述符合该疾病的可能表现，或与中医理论相符
- 不通过：描述明显错误，与该疾病完全无关，或存在严重医学错误

【输出要求】:
- 必须返回标准JSON格式
- 每个编号对应一个结果
- 理由要具体明确

返回格式（严格JSON，不要额外说明）:
{{
  "1": {{"result": "通过", "reason": "符合该疾病的临床表现"}},
  "2": {{"result": "不通过", "reason": "具体不通过的原因"}}
}}
"""

            # 记录处理过程
            processing_record = {
                'timestamp': datetime.now().isoformat(),
                'excel_disease': disease,
                'matched_disease': matched_disease,
                'match_method': match_method,
                'match_score': match_score,
                'field': field,
                'descriptions_count': len(descriptions),
                'used_knowledge_sections': used_sections,
                'full_prompt': prompt,
                'knowledge_content': knowledge_text
            }
            self.processing_records.append(processing_record)

            # 调用API
            response = self.call_azure_api(prompt)

            # 解析结果（增强容错性）
            if response:
                try:
                    # 清理JSON响应
                    cleaned_response = self.clean_json_response(response)
                    validation_data = json.loads(cleaned_response)

                    for i, description in enumerate(descriptions, 1):
                        key = str(i)
                        if key in validation_data and isinstance(validation_data[key], dict):
                            result_data = validation_data[key]
                            is_passed = result_data.get('result', '不通过')
                            reason = result_data.get('reason', '无明确理由')
                        else:
                            is_passed = '不通过'
                            reason = f'API响应格式错误，缺少编号{key}的验证结果'

                        all_results.append({
                            'Disease': disease,
                            '字段': field,
                            '描述内容': description,
                            '是否通过': is_passed,
                            '不通过理由': reason if is_passed == '不通过' else ''
                        })

                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON解析失败 - 疾病: {disease}, 字段: {field}, 错误: {str(e)}")
                    logger.error(f"原始响应: {response[:500]}...")  # 记录前500字符用于调试

                    # 尝试备用解析方法
                    backup_results = self.parse_response_backup(response, descriptions)
                    if backup_results:
                        for i, (description, result_data) in enumerate(zip(descriptions, backup_results)):
                            all_results.append({
                                'Disease': disease,
                                '字段': field,
                                '描述内容': description,
                                '是否通过': result_data['result'],
                                '不通过理由': result_data['reason'] if result_data['result'] == '不通过' else ''
                            })
                    else:
                        # 解析失败，采用保守策略：全部通过但标记需要人工审核
                        for description in descriptions:
                            all_results.append({
                                'Disease': disease,
                                '字段': field,
                                '描述内容': description,
                                '是否通过': '通过',
                                '不通过理由': f'API响应解析失败，需要人工审核: {str(e)}'
                            })
            else:
                # API调用失败
                logger.error(f"❌ API调用失败 - 疾病: {disease}, 字段: {field}")
                for description in descriptions:
                    all_results.append({
                        'Disease': disease,
                        '字段': field,
                        '描述内容': description,
                        '是否通过': '通过',
                        '不通过理由': 'API调用失败，需要人工审核'
                    })

            # 避免API调用过快
            time.sleep(0.5)

        logger.info(f"✅ 疾病 {disease} 验证完成，共 {len(all_results)} 条结果")
        return all_results

    def call_azure_api(self, prompt, max_retries=3):
        """
        调用Azure OpenAI API（带缓存）
        """
        # 生成缓存键
        cache_key = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        cache_file = os.path.join(self.cache_dir, "validation_api_responses.json")

        # 检查缓存
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                if cache_key in cache_data:
                    self.cache_hits += 1
                    return cache_data[cache_key]
            except:
                cache_data = {}
        else:
            cache_data = {}

        # API调用
        for attempt in range(max_retries):
            try:
                self.api_calls += 1
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system",
                         "content": "你是一位权威的中医专家，精通中医理论和临床实践，擅长基于标准医学知识库审核医疗描述的合理性。"},
                        {"role": "user", "content": prompt}
                    ],
                )

                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    result = response.choices[0].message.content.strip()

                    # 保存到缓存
                    cache_data[cache_key] = result
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, ensure_ascii=False, indent=2)

                    return result
                else:
                    logger.warning(f"⚠️ API返回空响应 (尝试 {attempt + 1}/{max_retries})")

            except Exception as e:
                logger.warning(f"⚠️ API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        logger.error("❌ API调用最终失败")
        return None

    def parse_medical_data(self, df):
        """
        解析纵向展开的医疗数据，按疾病分组
        """
        logger.info("📊 开始解析医疗数据...")

        disease_groups = defaultdict(list)
        current_disease = None

        for index, row in df.iterrows():
            # 更新当前疾病
            if pd.notna(row.get('Disease')):
                current_disease = row['Disease']

            if current_disease is None:
                continue

            # 提取每个字段的描述
            for field in self.key_fields:
                content = row.get(field)
                if pd.notna(content) and str(content).strip():
                    description = str(content).strip()
                    disease_groups[current_disease].append({
                        'field': field,
                        'description': description,
                        'row_index': index
                    })

        # 转换为列表格式
        disease_data_list = []
        for disease, items in disease_groups.items():
            disease_data_list.append({
                'disease': disease,
                'items': items
            })

        total_diseases = len(disease_data_list)
        total_items = sum(len(data['items']) for data in disease_data_list)

        logger.info(f"📊 数据解析完成: {total_diseases} 个疾病, {total_items} 条描述")

        # 显示疾病分布
        for data in disease_data_list[:5]:
            logger.info(f"  📋 {data['disease']}: {len(data['items'])} 条描述")
        if total_diseases > 5:
            logger.info(f"  📋 ... 还有 {total_diseases - 5} 个疾病")

        return disease_data_list

    def run_validation(self, disease_data_list):
        """
        运行验证（带进度条）
        """
        logger.info(f"🚀 开始验证 {len(disease_data_list)} 个疾病")

        all_results = []
        for disease_data in tqdm(disease_data_list, desc="验证进度", unit="疾病"):
            result = self.validate_single_disease_batch(disease_data)
            all_results.extend(result)

        logger.info(f"✅ 验证完成，共处理 {len(all_results)} 条描述")
        return all_results

    def save_results(self, results, processing_records, output_dir="../data/case_data"):
        """
        保存验证结果和中间处理记录
        """
        logger.info("💾 开始保存结果...")

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存主要结果
        results_file = os.path.join(output_dir, f"反向验证2kg_{timestamp}.xlsx")
        df_results = pd.DataFrame(results)
        df_results.to_excel(results_file, index=False)
        logger.info(f"📄 主要结果已保存: {results_file}")

        # 保存中间处理记录
        records_file = os.path.join(output_dir, f"验证过程记录_{timestamp}.xlsx")
        df_records = pd.DataFrame(processing_records)
        df_records.to_excel(records_file, index=False)
        logger.info(f"📄 处理记录已保存: {records_file}")

        return results_file, records_file

    def generate_statistics(self, results):
        """
        生成统计信息
        """
        stats = {
            'total': len(results),
            'passed': len([r for r in results if r['是否通过'] == '通过']),
            'failed': len([r for r in results if r['是否通过'] == '不通过']),
            'diseases_count': len(set(r['Disease'] for r in results)),
        }

        if stats['total'] > 0:
            stats['pass_rate'] = (stats['passed'] / stats['total']) * 100
            stats['fail_rate'] = (stats['failed'] / stats['total']) * 100
        else:
            stats['pass_rate'] = stats['fail_rate'] = 0

        return stats

    def run(self, input_file, max_diseases=None):
        """
        运行完整的医疗描述验证流程
        """
        start_time = time.time()

        logger.info("=" * 100)
        logger.info("🚀 开始基于知识库的医疗描述反向验证任务")
        if max_diseases:
            logger.info(f"⚠️  测试模式：仅处理前 {max_diseases} 个疾病")
        logger.info("=" * 100)

        try:
            # 步骤1：读取数据
            logger.info(f"📋 步骤1: 读取待审核数据文件: {input_file}")
            df = pd.read_excel(input_file)
            logger.info(f"✅ 成功读取 {len(df)} 行数据")

            # 步骤2：解析数据
            logger.info("📋 步骤2: 解析医疗数据")
            disease_data_list = self.parse_medical_data(df)

            if not disease_data_list:
                logger.error("❌ 未能解析到任何医疗数据，程序终止")
                return

            # 限制处理数量
            if max_diseases:
                disease_data_list = disease_data_list[:max_diseases]
                logger.info(f"🔢 限制处理：{len(disease_data_list)} 个疾病")

            # 步骤3：验证描述
            logger.info("📋 步骤3: 验证描述")
            results = self.run_validation(disease_data_list)

            # 步骤4：保存结果
            logger.info("📋 步骤4: 保存验证结果")
            results_file, records_file = self.save_results(results, self.processing_records)

            # 步骤5：生成统计
            stats = self.generate_statistics(results)

            # 输出统计信息
            end_time = time.time()
            processing_time = end_time - start_time

            logger.info("=" * 100)
            logger.info("📊 验证完成统计")
            logger.info("=" * 100)
            logger.info(f"⏱️  总处理时间: {processing_time:.2f} 秒")
            logger.info(f"🔄 API调用次数: {self.api_calls}")
            logger.info(f"💾 缓存命中次数: {self.cache_hits}")
            if self.api_calls + self.cache_hits > 0:
                cache_rate = self.cache_hits / (self.api_calls + self.cache_hits) * 100
                logger.info(f"📈 缓存命中率: {cache_rate:.1f}%")

            logger.info(f"🏥 处理疾病数: {stats['diseases_count']}")
            logger.info(f"📝 验证描述总数: {stats['total']}")
            logger.info(f"✅ 通过: {stats['passed']} ({stats['pass_rate']:.1f}%)")
            logger.info(f"❌ 不通过: {stats['failed']} ({stats['fail_rate']:.1f}%)")

            logger.info(f"📄 主要结果文件: {results_file}")
            logger.info(f"📄 过程记录文件: {records_file}")
            logger.info("")
            logger.info("📋 输出格式说明:")
            logger.info("  - 主要结果: Disease | 字段 | 描述内容 | 是否通过 | 不通过理由")
            logger.info("  - 过程记录: 包含疾病匹配、知识库使用、完整prompt等详细信息")
            logger.info("")
            logger.info("🔍 重点关注:")
            logger.info("  - 标记为'不通过'的描述需要医生重新审核")
            logger.info("  - 查看过程记录文件了解具体的知识库匹配和使用情况")
            logger.info("=" * 100)

        except Exception as e:
            logger.error(f"❌ 程序执行出错: {str(e)}")
            raise


def main():
    """
    主函数，配置参数并运行基于知识库的医疗描述验证
    """
    # Azure OpenAI配置
    AZURE_API_KEY = ""  # 请填入您的API密钥
    AZURE_ENDPOINT = ""  # 请填入您的端点地址
    DEPLOYMENT_NAME = "o3"  # 您的部署名称

    # 文件路径配置
    input_file = "../../data/case_data/病历表_待审核_20250810.xlsx"
    knowledge_base_path = "../../data/other/book_structured_enhanced.json"

    # 测试配置（设置为None处理全部疾病，设置数字只处理前N个疾病）
    MAX_DISEASES = None  # 建议先测试5个疾病，效果满意后改为None处理全部疾病

    # 检查配置
    if not AZURE_API_KEY or AZURE_API_KEY == "":
        logger.error("❌ 请先配置Azure OpenAI API密钥！")
        logger.info("请修改main()函数中的AZURE_API_KEY参数")
        return

    if not AZURE_ENDPOINT or AZURE_ENDPOINT == "":
        logger.error("❌ 请先配置Azure OpenAI端点地址！")
        logger.info("请修改main()函数中的AZURE_ENDPOINT参数")
        return

    # 检查文件是否存在
    if not os.path.exists(input_file):
        logger.error(f"❌ 输入文件不存在: {input_file}")
        return

    if not os.path.exists(knowledge_base_path):
        logger.error(f"❌ 知识库文件不存在: {knowledge_base_path}")
        return

    # 创建验证器
    validator = KnowledgeBasedValidator(
        azure_api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        deployment_name=DEPLOYMENT_NAME,
        knowledge_base_path=knowledge_base_path
    )

    # 运行验证流程
    try:
        validator.run(input_file, max_diseases=MAX_DISEASES)
    except KeyboardInterrupt:
        logger.info("⚠️ 用户中断程序执行")
    except Exception as e:
        logger.error(f"❌ 程序执行异常: {str(e)}")


if __name__ == "__main__":
    main()
