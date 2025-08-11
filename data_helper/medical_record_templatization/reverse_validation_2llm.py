import pandas as pd
import json
import logging
from datetime import datetime
from collections import defaultdict
from openai import AzureOpenAI
import time
import os
import hashlib
"""
医生病历审核结构反向验证，基于模型知识出了一版结果
● 输出格式：每行一个"疾病+字段+描述+审核结果+置信度"
● ✅ 审核结果：通过/不通过/待定（AI不确定时写待定）
● ✅ 输出到：../data/case_data/反向验证0810_all.xlsx
● ✅ 新文件，不修改原表

结果：
处理疾病数: 176
验证描述总数: 3062
通过：2639（86.2%）
不通过：169（5.5%）
待定：254（8.3%）
"""
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MedicalDescriptionValidator:
    """
    医疗描述合理性审核工具
    验证疾病与症状描述的医学合理性
    """

    def __init__(self, azure_api_key, azure_endpoint, deployment_name="o3", cache_dir="cache"):
        """
        初始化审核工具

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
        self.key_fields = ['主诉', '现病史', '既往史', '辅助检查', 'PE/检查 （体现望闻问切）', '病机', '治则/处理',
                           '医嘱']

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
        cache_file = "validation_api_responses.json"
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
                         "content": "你是一位资深的中西医结合专家，精通中医理论和现代医学，擅长判断疾病与症状描述的医学合理性。"},
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

    def parse_medical_data(self, df):
        """
        解析纵向展开的医疗数据

        Args:
            df (pd.DataFrame): 原始数据

        Returns:
            list: [(disease, field, description), ...] 所有需要验证的条目
        """
        logger.info("解析纵向展开的医疗数据...")

        validation_items = []
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
                    validation_items.append((current_disease, field, description))

        logger.info(f"解析完成: {len(validation_items)} 条描述需要验证")

        # 统计信息
        disease_count = len(set(item[0] for item in validation_items))
        logger.info(f"涉及 {disease_count} 个疾病")

        return validation_items

    def batch_validate_descriptions(self, validation_items):
        """
        批量验证描述

        Args:
            validation_items (list): [(disease, field, description), ...] 待验证条目

        Returns:
            list: [{'Disease': ..., '字段': ..., '描述内容': ..., '审核结果': ..., '置信度': ...}, ...]
        """
        logger.info("开始批量验证描述...")

        results = []
        batch_size = 10  # 每批处理10条
        total_batches = (len(validation_items) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(validation_items), batch_size):
            batch_items = validation_items[batch_idx:batch_idx + batch_size]
            current_batch = batch_idx // batch_size + 1

            logger.info(f"处理批次 {current_batch}/{total_batches} ({len(batch_items)} 条)")

            # 构建批量验证prompt
            items_text = []
            for i, (disease, field, description) in enumerate(batch_items, 1):
                items_text.append(f"{i}. 疾病：{disease} | 字段：{field} | 描述：{description}")

            prompt = f"""
请作为中西医结合专家，批量验证以下{len(batch_items)}条医疗描述的合理性：

验证列表：
{chr(10).join(items_text)}

验证标准（采用临床实际标准，不要过于严格）：
1. **临床合理性**：该描述是否可能出现在该疾病患者身上？（包括直接症状、并发症状、代偿症状、伴随症状等）
2. **医学关联性**：该描述与疾病是否有直接或间接的医学关联？
3. **字段适配性**：该描述是否适合在该字段中记录？

重要原则：
- **宽松判断**：只要描述与疾病有合理的医学关联就应该通过
- **临床视角**：从实际临床工作角度考虑，患者可能的复杂表现
- **中医特色**：中医强调整体观念，症状可能涉及多系统
- **代偿机制**：考虑疾病引起的代偿性改变和继发症状

具体判断要点：
- 颈椎病：可能引起颈肩腰背疼痛、头痛头晕、上肢麻木等，因为脊柱是一个整体
- 腰肌劳损：可能伴发颈肩问题，因为姿势代偿
- 病机：中医病机可以多样化，不要求单一标准答案
- 治法：只要不是明显错误的治法都可以通过
- 症状：考虑疾病的多样性表现和个体差异

审核结果判断：
- **通过**：描述在临床上是合理的、可能出现的
- **不通过**：描述明显医学错误，完全不可能与该疾病相关
- **待定**：描述存在一定争议，需要更多临床信息判断

置信度判断：
- **高**：判断非常确定，有充分医学依据
- **中**：判断较确定，基于临床经验和医学常识
- **低**：判断不够确定，可能存在例外情况

请返回JSON格式，键为条目编号(1,2,3...)，值为验证结果：
{{
  "1": {{
    "result": "通过/不通过/待定",
    "confidence": "高/中/低"
  }},
  "2": {{
    "result": "通过/不通过/待定", 
    "confidence": "高/中/低"
  }}
}}

示例：
{{"1": {{"result": "通过", "confidence": "高"}}, "2": {{"result": "不通过", "confidence": "高"}}, "3": {{"result": "待定", "confidence": "中"}}}}
"""

            response = self.call_azure_api(prompt)

            if response:
                try:
                    validation_data = json.loads(response)

                    for i, (disease, field, description) in enumerate(batch_items, 1):
                        key = str(i)
                        if key in validation_data and isinstance(validation_data[key], dict):
                            result_data = validation_data[key]
                            audit_result = result_data.get('result', '待定')
                            confidence = result_data.get('confidence', '低')
                        else:
                            audit_result = '待定'
                            confidence = '低'

                        results.append({
                            'Disease': disease,
                            '字段': field,
                            '描述内容': description,
                            '审核结果': audit_result,
                            '置信度': confidence
                        })

                except json.JSONDecodeError:
                    logger.warning(f"批次 {current_batch} JSON解析失败，标记为待定")
                    # 解析失败，全部标记为待定
                    for disease, field, description in batch_items:
                        results.append({
                            'Disease': disease,
                            '字段': field,
                            '描述内容': description,
                            '审核结果': '待定',
                            '置信度': '低'
                        })
            else:
                # API调用失败，全部标记为待定
                logger.warning(f"批次 {current_batch} API调用失败，标记为待定")
                for disease, field, description in batch_items:
                    results.append({
                        'Disease': disease,
                        '字段': field,
                        '描述内容': description,
                        '审核结果': '待定',
                        '置信度': '低'
                    })

            # 避免API调用过快
            time.sleep(0.5)

        return results

    def generate_statistics(self, results):
        """
        生成统计信息

        Args:
            results (list): 验证结果列表

        Returns:
            dict: 统计信息
        """
        stats = {
            'total': len(results),
            'passed': len([r for r in results if r['审核结果'] == '通过']),
            'failed': len([r for r in results if r['审核结果'] == '不通过']),
            'pending': len([r for r in results if r['审核结果'] == '待定']),
            'high_confidence': len([r for r in results if r['置信度'] == '高']),
            'medium_confidence': len([r for r in results if r['置信度'] == '中']),
            'low_confidence': len([r for r in results if r['置信度'] == '低']),
            'diseases_count': len(set(r['Disease'] for r in results)),
        }

        if stats['total'] > 0:
            stats['pass_rate'] = (stats['passed'] / stats['total']) * 100
            stats['fail_rate'] = (stats['failed'] / stats['total']) * 100
            stats['pending_rate'] = (stats['pending'] / stats['total']) * 100
        else:
            stats['pass_rate'] = stats['fail_rate'] = stats['pending_rate'] = 0

        return stats

    def run(self, input_file, max_diseases=None):
        """
        运行完整的医疗描述验证流程

        Args:
            input_file (str): 输入Excel文件路径
            max_diseases (int): 最大处理疾病数量，None表示处理全部
        """
        start_time = time.time()

        logger.info("=" * 80)
        logger.info("开始医疗描述反向验证任务")
        if max_diseases:
            logger.info(f"⚠️  测试模式：仅处理前 {max_diseases} 个疾病")
        logger.info("=" * 80)

        try:
            # 步骤1：读取数据
            logger.info(f"📋 步骤1: 读取待审核数据文件: {input_file}")
            df = pd.read_excel(input_file)
            logger.info(f"成功读取 {len(df)} 行数据")

            # 步骤2：解析纵向数据
            logger.info("📋 步骤2: 解析医疗数据")
            validation_items = self.parse_medical_data(df)

            if not validation_items:
                logger.error("未能解析到任何医疗数据，程序终止")
                return

            # 限制处理数量（如果设置了max_diseases）
            if max_diseases:
                # 按疾病分组
                disease_items = defaultdict(list)
                for item in validation_items:
                    disease_items[item[0]].append(item)

                # 只取前N个疾病
                limited_diseases = list(disease_items.keys())[:max_diseases]
                validation_items = []
                for disease in limited_diseases:
                    validation_items.extend(disease_items[disease])

                logger.info(f"限制处理：{len(limited_diseases)} 个疾病，{len(validation_items)} 条描述")

            # 步骤3：批量验证描述
            logger.info("📋 步骤3: 批量验证描述合理性")
            results = self.batch_validate_descriptions(validation_items)

            # 步骤4：保存结果
            logger.info("📋 步骤4: 保存验证结果")

            # 确保输出目录存在
            output_dir = "../../data/case_data"
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, "反向验证0810_all.xlsx")

            # 保存到Excel
            df_results = pd.DataFrame(results)
            df_results.to_excel(output_file, index=False)

            # 步骤5：生成统计信息
            stats = self.generate_statistics(results)

            # 输出统计信息
            end_time = time.time()
            processing_time = end_time - start_time

            logger.info("=" * 80)
            logger.info("📊 反向验证完成统计")
            logger.info("=" * 80)
            logger.info(f"处理时间: {processing_time:.2f} 秒")
            logger.info(f"API调用次数: {self.api_calls}")
            logger.info(f"缓存命中次数: {self.cache_hits}")
            if self.api_calls + self.cache_hits > 0:
                cache_rate = self.cache_hits / (self.api_calls + self.cache_hits) * 100
                logger.info(f"缓存命中率: {cache_rate:.1f}%")

            logger.info(f"处理疾病数: {stats['diseases_count']}")
            logger.info(f"验证描述总数: {stats['total']}")
            logger.info(f"通过: {stats['passed']} ({stats['pass_rate']:.1f}%)")
            logger.info(f"不通过: {stats['failed']} ({stats['fail_rate']:.1f}%)")
            logger.info(f"待定: {stats['pending']} ({stats['pending_rate']:.1f}%)")
            logger.info("")
            logger.info(f"高置信度: {stats['high_confidence']}")
            logger.info(f"中置信度: {stats['medium_confidence']}")
            logger.info(f"低置信度: {stats['low_confidence']}")
            logger.info(f"输出文件: {output_file}")
            logger.info("")
            logger.info("📋 输出格式: Disease | 字段 | 描述内容 | 审核结果 | 置信度")
            logger.info("🔍 重点关注: 标记为'不通过'和'待定'的描述")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"程序执行出错: {str(e)}")
            raise


def main():
    """
    主函数，配置参数并运行医疗描述验证
    """
    # Azure OpenAI配置
    AZURE_API_KEY = ""
    AZURE_ENDPOINT = ""
    DEPLOYMENT_NAME = "o3"  # 您的部署名称

    # 输入文件路径
    input_file = "../../data/case_data/病历表_待审核_20250808_.xlsx"

    # 测试配置（设置为None处理全部疾病，设置数字只处理前N个疾病）
    MAX_DISEASES = None  # 建议先测试5个疾病，效果满意后改为None处理全部176个疾病

    # 检查配置
    if not AZURE_API_KEY or AZURE_API_KEY == "":
        logger.error("请先配置Azure OpenAI API密钥！")
        logger.info("请修改main()函数中的AZURE_API_KEY参数")
        return

    if not AZURE_ENDPOINT or AZURE_ENDPOINT == "":
        logger.error("请先配置Azure OpenAI端点地址！")
        logger.info("请修改main()函数中的AZURE_ENDPOINT参数")
        return

    # 创建验证器并运行
    validator = MedicalDescriptionValidator(
        azure_api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        deployment_name=DEPLOYMENT_NAME
    )

    # 运行验证流程
    validator.run(input_file, max_diseases=MAX_DISEASES)


if __name__ == "__main__":
    main()
