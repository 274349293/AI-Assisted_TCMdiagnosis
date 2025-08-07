import pandas as pd
import json
import logging
from datetime import datetime
from collections import defaultdict
from openai import AzureOpenAI
import time
import re

"""
测试版本，该版本的结果为：Disease模板_测试10个_20250805_150827.xlsx

评测过测试结果后，需求有批量更新，代码暂时弃用
"""
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MedicalTemplateGenerator:
    """
    医疗记录模板生成器
    基于Azure OpenAI GPT-4o生成Disease模板，供医生审核使用
    """

    def __init__(self, azure_api_key, azure_endpoint, deployment_name="o3"):
        """
        初始化模板生成器

        Args:
            azure_api_key (str): Azure OpenAI API密钥
            azure_endpoint (str): Azure OpenAI端点
            deployment_name (str): 部署名称，默认o3
        """
        self.client = AzureOpenAI(
            api_key=azure_api_key,
            api_version="2025-01-01-preview",  # 使用o3对应的API版本
            azure_endpoint=azure_endpoint
        )
        self.deployment_name = deployment_name

        # 定义关键医疗字段
        self.key_fields = ['主诉', '现病史', '既往史', '辅助检查', 'PE/检查', '病机', '治则/处理', '医嘱']

    def call_azure_api(self, prompt, max_retries=3):
        """
        调用Azure OpenAI API

        Args:
            prompt (str): 提示词
            max_retries (int): 最大重试次数

        Returns:
            str: API响应内容
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "你是一个专业的医疗信息处理专家，精通中医和西医理论。"},
                        {"role": "user", "content": prompt}
                    ],

                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    logger.error(f"API调用最终失败: {str(e)}")
                    return None

    def extract_diseases_from_record(self, diagnosis_text):
        """
        从诊断文本中提取Disease列表

        Args:
            diagnosis_text (str): 诊断文本

        Returns:
            list: Disease列表
        """
        if not diagnosis_text or pd.isna(diagnosis_text):
            return []

        prompt = f"""
请分析以下中医诊断内容，提取出独立的Disease：

诊断：{diagnosis_text}

提取规则：
1. 中医Disease = 病名 + 证型（例如：颈椎病+气滞血瘀证）
2. 西医Disease = 单独疾病名（例如：偏头痛）
3. 去除所有标注信息，如"(可选词：...)"、"[...]"等
4. 病名和证型用"+"连接
5. 如果只有病名没有证型，或只有证型没有病名，则单独作为一个Disease

请直接返回JSON格式的Disease列表，不要其他说明：
["Disease1", "Disease2", ...]

示例：
输入：颈椎病(可选词：项痹),气滞血瘀证,虚劳病,气血不足证
输出：["颈椎病+气滞血瘀证", "虚劳病+气血不足证"]
"""

        response = self.call_azure_api(prompt)
        if not response:
            return []

        try:
            # 尝试解析JSON
            diseases = json.loads(response)
            if isinstance(diseases, list):
                return [disease.strip() for disease in diseases if disease.strip()]
        except json.JSONDecodeError:
            logger.warning(f"JSON解析失败，原始响应: {response}")
            # 备用解析：尝试从响应中提取疾病名称
            diseases = re.findall(r'"([^"]+)"', response)
            if diseases:
                return [disease.strip() for disease in diseases if disease.strip()]

        return []

    def group_records_by_disease(self, df):
        """
        按Disease分组所有记录

        Args:
            df (pd.DataFrame): 医疗记录数据框

        Returns:
            dict: {disease: [records]}
        """
        logger.info("开始按Disease分组记录...")

        disease_records = defaultdict(list)
        total_records = len(df)

        for index, row in df.iterrows():
            if index % 50 == 0:
                logger.info(f"处理进度: {index + 1}/{total_records}")

            diagnosis = row.get('诊断', '')
            diseases = self.extract_diseases_from_record(diagnosis)

            if not diseases:
                logger.warning(f"记录 {index + 1} 无法提取Disease，诊断内容: {diagnosis}")
                continue

            # 为每个Disease添加这条记录
            for disease in diseases:
                record_data = {
                    '患者ID': f"P{index + 1:03d}",
                    '原始诊断': diagnosis
                }

                # 添加8个关键字段
                for field in self.key_fields:
                    record_data[field] = row.get(field, '') or ''

                disease_records[disease].append(record_data)

        logger.info(f"分组完成，共识别出 {len(disease_records)} 个唯一Disease")
        for disease, records in disease_records.items():
            logger.info(f"  {disease}: {len(records)} 条记录")

        return dict(disease_records)

    def generate_template_for_disease(self, disease, records):
        """
        为单个Disease生成模板

        Args:
            disease (str): Disease名称
            records (list): 相关记录列表

        Returns:
            dict: 模板内容
        """
        logger.info(f"正在为 '{disease}' 生成模板（{len(records)} 条记录）...")

        # 格式化记录内容
        formatted_records = []
        for i, record in enumerate(records[:10], 1):  # 最多使用10条记录，避免prompt过长
            record_text = f"记录{i}:\n"
            for field in self.key_fields:
                content = record[field] if record[field] else "无"
                record_text += f"  {field}: {content}\n"
            formatted_records.append(record_text)

        records_text = "\n".join(formatted_records)

        prompt = f"""
请基于以下病历记录，为Disease "{disease}" 生成临床使用模板。

样本记录数: {len(records)}
使用记录数: {min(len(records), 10)}

病历记录：
{records_text}

生成要求：
1. 为8个字段（主诉、现病史、既往史、辅助检查、PE/检查、病机、治则/处理、医嘱）每个都生成模板
2. 模板化处理：
   - 时间泛化：1月→N月，2天→N天，数年→N年
   - 个人信息泛化：具体姓名→[患者]
   - 数值泛化：65kg→Nkg，3次→N次
   - 程度泛化：轻度/中度/重度保留，具体数值泛化
3. 融合多个记录的共同特征和表述方式
4. 保持中医和西医术语的专业性和准确性
5. 如果某字段在所有记录中都为空或无意义，填写"待完善"
6. 生成的模板要便于医生快速填写和使用

请直接返回JSON格式，不要其他说明：
{{
  "主诉": "模板内容",
  "现病史": "模板内容",
  "既往史": "模板内容", 
  "辅助检查": "模板内容",
  "PE/检查": "模板内容",
  "病机": "模板内容",
  "治则/处理": "模板内容",
  "医嘱": "模板内容"
}}
"""

        response = self.call_azure_api(prompt)
        if not response:
            logger.error(f"为 '{disease}' 生成模板失败")
            return {field: "模板生成失败" for field in self.key_fields}

        try:
            template = json.loads(response)
            if isinstance(template, dict):
                # 确保所有字段都存在
                complete_template = {}
                for field in self.key_fields:
                    complete_template[field] = template.get(field, "待完善")
                return complete_template
        except json.JSONDecodeError:
            logger.warning(f"JSON解析失败，Disease: {disease}，原始响应: {response}")

        # 备用方案：返回空模板
        return {field: "待完善" for field in self.key_fields}

    def process_all_data(self, input_file, max_diseases=None):
        """
        处理所有数据的主流程

        Args:
            input_file (str): 输入Excel文件路径
            max_diseases (int): 最大处理Disease数量，None表示处理全部

        Returns:
            tuple: (templates, original_records)
        """
        logger.info("=" * 80)
        logger.info("开始医疗记录模板化处理")
        if max_diseases:
            logger.info(f"⚠️  测试模式：仅处理前 {max_diseases} 个Disease")
        logger.info("=" * 80)

        # 读取数据
        logger.info(f"读取数据文件: {input_file}")
        try:
            df = pd.read_excel(input_file)
            logger.info(f"成功读取 {len(df)} 条记录")
        except Exception as e:
            logger.error(f"读取文件失败: {str(e)}")
            return None, None

        # 按Disease分组
        disease_records = self.group_records_by_disease(df)

        if not disease_records:
            logger.error("未能提取到任何Disease，处理终止")
            return None, None

        # 如果设置了最大数量限制，只取前N个Disease
        if max_diseases:
            disease_items = list(disease_records.items())[:max_diseases]
            disease_records = dict(disease_items)
            logger.info(f"🔢 限制处理数量，实际处理 {len(disease_records)} 个Disease")

        # 生成模板
        logger.info("开始生成Disease模板...")
        templates = []
        all_original_records = []

        total_diseases = len(disease_records)
        for i, (disease, records) in enumerate(disease_records.items(), 1):
            logger.info(f"处理进度: {i}/{total_diseases} - {disease}")

            # 生成模板
            template = self.generate_template_for_disease(disease, records)

            # 添加到模板列表
            template_row = {
                'Disease': disease,
                '样本数': len(records),
                '审核状态': '待审核',
                '备注': ''
            }
            template_row.update(template)
            templates.append(template_row)

            # 添加原始记录
            for record in records:
                record_row = {'Disease': disease}
                record_row.update(record)
                all_original_records.append(record_row)

            # 避免API调用过快
            time.sleep(0.5)

        logger.info(f"模板生成完成，共生成 {len(templates)} 个Disease模板")
        return templates, all_original_records

    def save_results(self, templates, original_records, output_file):
        """
        保存结果到Excel文件

        Args:
            templates (list): 模板列表
            original_records (list): 原始记录列表
            output_file (str): 输出文件路径
        """
        logger.info(f"保存结果到: {output_file}")

        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # 工作表1：Disease模板表
                templates_df = pd.DataFrame(templates)
                templates_df.to_excel(writer, sheet_name='Disease模板表', index=False)

                # 工作表2：原始记录对照表
                original_df = pd.DataFrame(original_records)
                original_df.to_excel(writer, sheet_name='原始记录对照表', index=False)

            logger.info(f"✅ 结果已保存到: {output_file}")
            logger.info(f"   - Disease模板表: {len(templates)} 行")
            logger.info(f"   - 原始记录对照表: {len(original_records)} 行")

        except Exception as e:
            logger.error(f"保存文件失败: {str(e)}")

    def run(self, input_file, max_diseases=None):
        """
        运行完整的模板生成流程

        Args:
            input_file (str): 输入Excel文件路径
            max_diseases (int): 最大处理Disease数量，None表示处理全部
        """
        start_time = time.time()

        try:
            # 处理数据
            templates, original_records = self.process_all_data(input_file, max_diseases)

            if templates is None:
                logger.error("处理失败，程序退出")
                return

            # 生成输出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if max_diseases:
                output_file = f"Disease模板_测试{max_diseases}个_{timestamp}.xlsx"
            else:
                output_file = f"Disease模板_待审核_{timestamp}.xlsx"

            # 保存结果
            self.save_results(templates, original_records, output_file)

            # 输出统计信息
            end_time = time.time()
            processing_time = end_time - start_time

            logger.info("=" * 80)
            logger.info("📊 处理完成统计")
            logger.info("=" * 80)
            logger.info(f"处理时间: {processing_time:.2f} 秒")
            logger.info(f"生成Disease数量: {len(templates)}")
            logger.info(f"处理记录总数: {len(original_records)}")
            logger.info(f"输出文件: {output_file}")
            if max_diseases:
                logger.info(f"⚠️  这是测试版本，仅处理了前 {max_diseases} 个Disease")
            logger.info("")
            logger.info("📋 下一步操作：")
            logger.info("1. 打开生成的Excel文件")
            logger.info("2. 在'Disease模板表'中审核和编辑模板内容")
            logger.info("3. 更新'审核状态'列（通过/修改/删除）")
            logger.info("4. 在'备注'列添加修改说明")
            logger.info("5. 使用'原始记录对照表'作为参考")
            if max_diseases:
                logger.info("6. 如果测试效果满意，可以去掉max_diseases限制处理全部数据")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"程序执行出错: {str(e)}")
            raise


def main():
    """
    主函数，配置参数并运行
    """
    # Azure OpenAI配置 - 根据你的示例更新
    AZURE_API_KEY = ""  # 替换为你的API密钥
    AZURE_ENDPOINT = ""  # 你的端点
    DEPLOYMENT_NAME = "o3"  # 你的部署名称

    # 输入文件路径
    input_file = "../data/case_data/病历数据_可使用_20250804_172720.xlsx"

    # 🔧 测试配置：设置为10只处理前10个Disease，设置为None处理全部
    MAX_DISEASES = 10  # 改为None可处理全部Disease

    # 检查配置
    if AZURE_API_KEY == "<your-api-key>":
        logger.error("请先配置Azure OpenAI API密钥！")
        logger.info("请修改main()函数中的AZURE_API_KEY参数")
        return

    # 创建生成器并运行
    generator = MedicalTemplateGenerator(
        azure_api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        deployment_name=DEPLOYMENT_NAME
    )

    # 运行处理流程
    generator.run(input_file, max_diseases=MAX_DISEASES)


if __name__ == "__main__":
    main()
