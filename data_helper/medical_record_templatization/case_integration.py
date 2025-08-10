import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
合并友杨和喜氏的门诊病历，对门诊病历进行筛选

合并数据源：
"../data/case_data/友杨0725+.xlsx",
"../data/case_data/喜氏0725+.xlsx"

输出2个文件：
1. 至少一个字段为空的记录
2. 至少一个字段有数据的记录
"""


def load_and_merge_medical_records():
    """
    加载两个病历Excel文件并合并数据

    Returns:
        pd.DataFrame: 合并后的病历数据
    """
    logger.info("开始加载病历数据文件...")

    # 定义文件路径
    file_paths = [
        "../data/case_data/友杨0725+.xlsx",
        "../data/case_data/喜氏0725+.xlsx"
    ]

    # 存储所有数据的列表
    all_records = []

    for file_path in file_paths:
        try:
            logger.info(f"正在处理文件: {file_path}")

            # 读取Excel文件
            df = pd.read_excel(file_path, engine='openpyxl')
            logger.info(f"成功读取文件 {file_path}，包含 {len(df)} 条记录")

            # 添加数据来源标识
            df['数据来源'] = file_path
            all_records.append(df)

        except FileNotFoundError:
            logger.error(f"文件不存在: {file_path}")
        except Exception as e:
            logger.error(f"读取文件 {file_path} 时出错: {str(e)}")

    # 合并所有数据
    if all_records:
        combined_df = pd.concat(all_records, ignore_index=True)
        logger.info(f"数据合并完成，总计 {len(combined_df)} 条病历记录")
        return combined_df
    else:
        logger.error("没有成功加载任何数据文件")
        return pd.DataFrame()


def analyze_data_completeness(df):
    """
    分析数据完整性并分类

    Args:
        df (pd.DataFrame): 病历数据

    Returns:
        tuple: (至少一个字段为空的记录, 至少一个字段有数据的记录, 分析结果)
    """
    logger.info("开始分析数据完整性...")

    # 定义关键医疗字段
    key_fields = [
        '主诉',  # 患者主要症状描述
        '现病史',  # 现在疾病的病史
        '既往史',  # 过去的疾病史
        '辅助检查',  # 各种检查结果
        'PE/检查',  # 体格检查结果
        '病机',  # 中医病机分析
        '治则/处理',  # 治疗原则
        '医嘱'  # 医生的处方和建议
    ]

    def is_empty_value(value):
        """判断值是否为空"""
        if pd.isna(value):
            return True
        if isinstance(value, str) and value.strip() == '':
            return True
        return False

    # 计算每条记录的缺失字段数
    missing_counts = []
    for _, row in df.iterrows():
        missing_count = sum(1 for field in key_fields
                            if field in df.columns and is_empty_value(row[field]))
        missing_counts.append(missing_count)

    df['缺失字段数量'] = missing_counts
    df['总关键字段数'] = len([f for f in key_fields if f in df.columns])
    df['数据完整性百分比'] = ((df['总关键字段数'] - df['缺失字段数量']) / df['总关键字段数'] * 100).round(2)

    # 分类数据
    # 第一类：至少一个字段为空的记录
    incomplete_records = df[df['缺失字段数量'] > 0].copy()

    # 第二类：至少一个字段有数据的记录
    complete_records = df[df['缺失字段数量'] < len([f for f in key_fields if f in df.columns])].copy()

    # 统计分析结果
    analysis_results = {
        '总记录数': len(df),
        '至少一个字段为空的记录数': len(incomplete_records),
        '至少一个字段有数据的记录数': len(complete_records),
        '字段填充率': {}
    }

    # 计算各字段填充率
    for field in key_fields:
        if field in df.columns:
            empty_count = df[field].apply(is_empty_value).sum()
            fill_rate = ((len(df) - empty_count) / len(df)) * 100
            analysis_results['字段填充率'][field] = round(fill_rate, 2)

    # 按数据来源统计
    if '数据来源' in df.columns:
        analysis_results['数据来源统计'] = {}
        for source in df['数据来源'].unique():
            source_df = df[df['数据来源'] == source]
            at_least_one_empty = len(source_df[source_df['缺失字段数量'] > 0])
            at_least_one_data = len(
                source_df[source_df['缺失字段数量'] < len([f for f in key_fields if f in df.columns])])
            analysis_results['数据来源统计'][source] = {
                '总记录数': len(source_df),
                '至少一个字段为空': at_least_one_empty,
                '至少一个字段有数据': at_least_one_data
            }

    return incomplete_records, complete_records, analysis_results


def print_analysis_results(analysis_results):
    """
    打印数据分析结果

    Args:
        analysis_results (dict): 分析结果
    """
    logger.info("=" * 80)
    logger.info("病历数据质量分析报告")
    logger.info("=" * 80)

    # 基本统计
    logger.info("📊 基本统计:")
    logger.info(f"   总记录数: {analysis_results['总记录数']} 条")
    logger.info(f"   至少一个字段为空的记录数: {analysis_results['至少一个字段为空的记录数']} 条")
    logger.info(f"   至少一个字段有数据的记录数: {analysis_results['至少一个字段有数据的记录数']} 条")

    # 字段填充率
    logger.info("")
    logger.info("📋 各字段填充率:")
    field_stats = analysis_results['字段填充率']
    sorted_fields = sorted(field_stats.items(), key=lambda x: x[1], reverse=True)
    for i, (field, rate) in enumerate(sorted_fields, 1):
        logger.info(f"   {i}. {field}: {rate}%")

    # 数据来源对比
    if '数据来源统计' in analysis_results:
        logger.info("")
        logger.info("📊 数据来源质量对比:")
        for source, stats in analysis_results['数据来源统计'].items():
            logger.info(f"   {source}:")
            logger.info(f"     总记录数: {stats['总记录数']} 条")
            logger.info(f"     至少一个字段为空: {stats['至少一个字段为空']} 条")
            logger.info(f"     至少一个字段有数据: {stats['至少一个字段有数据']} 条")

    logger.info("=" * 80)


def save_results(incomplete_records, complete_records):
    """
    保存分类结果到Excel文件

    Args:
        incomplete_records (pd.DataFrame): 至少一个字段为空的记录
        complete_records (pd.DataFrame): 至少一个字段有数据的记录
    """
    logger.info("开始保存分类结果...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存至少一个字段为空的记录
    if len(incomplete_records) > 0:
        incomplete_file = f"../data/case_data/病历数据_待补充_{timestamp}.xlsx"
        incomplete_records.to_excel(incomplete_file, index=False)
        logger.info(f"✅ 至少一个字段为空的记录已保存: {incomplete_file}")

    # 保存至少一个字段有数据的记录
    if len(complete_records) > 0:
        complete_file = f"../data/case_data/病历数据_可使用_{timestamp}.xlsx"
        complete_records.to_excel(complete_file, index=False)
        logger.info(f"✅ 至少一个字段有数据的记录已保存: {complete_file}")


def main():
    """
    主函数：执行完整的数据清洗流程
    """
    logger.info("=" * 80)
    logger.info("开始执行病历数据清洗任务")
    logger.info("=" * 80)

    try:
        # 步骤1：加载和合并数据
        logger.info("🔄 步骤 1/4: 加载和合并病历数据")
        combined_data = load_and_merge_medical_records()

        if combined_data.empty:
            logger.error("❌ 没有加载到任何数据，程序终止")
            return

        # 步骤2：分析数据完整性并分类
        logger.info("🔄 步骤 2/4: 分析数据完整性并分类")
        incomplete_records, complete_records, analysis_results = analyze_data_completeness(combined_data)

        # 步骤3：打印分析结果
        logger.info("🔄 步骤 3/4: 生成分析报告")
        print_analysis_results(analysis_results)

        # 步骤4：保存分类结果
        logger.info("🔄 步骤 4/4: 保存分类结果")
        save_results(incomplete_records, complete_records)

        logger.info("=" * 80)
        logger.info("✅ 病历数据清洗任务完成!")
        logger.info("=" * 80)
        logger.info("📁 输出文件:")
        logger.info(f"   ├── 病历数据_至少一个字段为空_[时间戳].xlsx ({len(incomplete_records)}条记录)")
        logger.info(f"   └── 病历数据_至少一个字段有数据_[时间戳].xlsx ({len(complete_records)}条记录)")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ 程序执行过程中出现错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
