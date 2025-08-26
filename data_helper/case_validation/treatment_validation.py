import pandas as pd
import re
from datetime import datetime, timedelta
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_excel_file(file_path):
    """读取Excel文件"""
    try:
        df = pd.read_excel(file_path)
        logger.info(f"成功读取Excel文件，共{len(df)}条记录")
        return df
    except Exception as e:
        logger.error(f"读取Excel文件失败: {e}")
        raise


def should_skip_treatment_item(item):
    """判断治疗项是否应该跳过"""
    skip_keywords = ['中医辨证论治', '红外线治疗(TDP)', '煎药机煎药(门诊)']

    # 检查是否包含跳过的关键词
    for keyword in skip_keywords:
        if keyword in item:
            return True

    # 检查最后两个字是否是"穴位"
    if len(item.strip()) >= 2 and item.strip().endswith('穴位'):
        return True

    return False


def has_non_skip_treatment_items(treatment_text):
    """
    检查治疗文本是否包含非跳过的治疗项

    Args:
        treatment_text: 治疗列内容

    Returns:
        True如果包含有效的治疗项，False如果全部为跳过项或为空
    """
    if pd.isna(treatment_text) or not str(treatment_text).strip():
        return False

    # 按照 (1), (2), (3) 等分割治疗项
    items = re.split(r'\(\d+\)', str(treatment_text))
    items = [item.strip() for item in items if item.strip()]

    # 检查是否有任何非跳过项
    for item in items:
        if not should_skip_treatment_item(item):
            # 进一步检查该项是否包含有效数字
            try:
                number = extract_numbers_from_treatment_item(item)
                if number > 0:
                    return True
            except:
                # 如果提取数字失败，仍然认为这是一个需要考虑的治疗项
                return True

    return False


def extract_numbers_from_treatment_item(item):
    """从治疗项中提取阿拉伯数字，只取整数部分"""
    # 查找所有数字（包括小数）
    numbers = re.findall(r'\d+\.?\d*', item)

    if len(numbers) == 0:
        raise ValueError(f"治疗项中未找到数字: {item}")
    elif len(numbers) > 1:
        raise ValueError(f"治疗项中存在多个数字: {item}, 找到的数字: {numbers}")

    # 只取整数部分
    number = float(numbers[0])
    return int(number)


def parse_treatment_column(treatment_text):
    """解析治疗列，返回有效治疗项中的最大数字"""
    if pd.isna(treatment_text) or treatment_text.strip() == '':
        return None  # 治疗列为空，跳过

    # 按照 (1), (2), (3) 等分割治疗项
    items = re.split(r'\(\d+\)', treatment_text)
    items = [item.strip() for item in items if item.strip()]

    valid_numbers = []
    exceptions = []

    for item in items:
        try:
            if should_skip_treatment_item(item):
                continue  # 跳过该项

            number = extract_numbers_from_treatment_item(item)
            valid_numbers.append(number)

        except ValueError as e:
            exceptions.append(str(e))

    if exceptions:
        raise ValueError(f"治疗项解析异常: {'; '.join(exceptions)}")

    if not valid_numbers:
        return 0  # 所有项都被跳过，返回0

    return max(valid_numbers)


def find_problematic_records(df):
    """查找有问题的病历记录"""
    problematic_records = []
    processing_errors = []

    # 按患者分组
    for patient_name in df['姓名'].unique():
        if pd.isna(patient_name):
            continue

        patient_records = df[df['姓名'] == patient_name].copy()
        patient_records = patient_records.sort_values('就诊日期')

        for idx, record in patient_records.iterrows():
            try:
                # 解析治疗列
                max_number = parse_treatment_column(record['治疗'])

                if max_number is None or max_number == 0:
                    continue  # 跳过空治疗或所有项都被跳过的记录

                # 解析就诊日期
                visit_date = pd.to_datetime(record['就诊日期']).date()  # 只取日期部分

                # 检查该患者在包含当天的max_number天内是否有就诊记录
                check_start_date = visit_date  # 从当天开始检查
                check_end_date = visit_date + timedelta(days=max_number - 1)  # 包含当天的max_number天

                # 查找在检查时间范围内的就诊记录（排除当前记录本身）
                patient_records_dates = pd.to_datetime(patient_records['就诊日期']).dt.date  # 只取日期部分
                potential_conflicts = patient_records[
                    (patient_records_dates >= check_start_date) &
                    (patient_records_dates <= check_end_date) &
                    (patient_records.index != idx)  # 排除当前记录本身
                    ]

                # 新增逻辑：过滤掉治疗项全部为跳过项的冲突记录
                actual_conflicts = []
                for _, conflict_record in potential_conflicts.iterrows():
                    conflict_treatment = conflict_record['治疗']
                    if has_non_skip_treatment_items(conflict_treatment):
                        # 只有当冲突记录包含有效治疗项时才算真正冲突
                        actual_conflicts.append(conflict_record)
                    else:
                        logger.debug(f"跳过冲突检查 - 冲突记录的治疗项全部为跳过项: "
                                     f"患者 {patient_name}, "
                                     f"冲突日期 {pd.to_datetime(conflict_record['就诊日期']).date()}")

                if len(actual_conflicts) > 0:
                    # 记录当前问题记录和真正的冲突记录
                    problem_info = {
                        'current_record': record,
                        'max_number': max_number,
                        'check_period': f"{check_start_date.strftime('%Y-%m-%d')} 到 {check_end_date.strftime('%Y-%m-%d')}",
                        'conflicting_dates': [pd.to_datetime(cr['就诊日期']).strftime('%Y-%m-%d') for cr in
                                              actual_conflicts]
                    }
                    problematic_records.append(problem_info)

                    logger.info(f"发现问题记录 - 患者: {patient_name}, "
                                f"就诊日期: {visit_date.strftime('%Y-%m-%d')}, "
                                f"最大数字: {max_number}, "
                                f"冲突就诊日期: {problem_info['conflicting_dates']}")

            except Exception as e:
                error_info = {
                    'patient_name': patient_name,
                    'visit_date': record['就诊日期'],
                    'treatment': record['治疗'],
                    'error': str(e)
                }
                processing_errors.append(error_info)
                logger.error(f"处理记录时发生异常 - 患者: {patient_name}, "
                             f"就诊日期: {record['就诊日期']}, 错误: {e}")

    return problematic_records, processing_errors


def save_problematic_records(problematic_records, original_df, output_file):
    """保存有问题的病历到Excel文件"""
    if not problematic_records:
        logger.info("没有发现问题记录")
        # 即使没有问题记录，也要保存原始数据
        original_df_copy = original_df.copy()
        original_df_copy['问题分析'] = '无冲突'
        original_df_copy.to_excel(output_file, index=False)
        logger.info(f"已保存{len(original_df_copy)}条记录到 {output_file}（无冲突记录）")
        return

    # 创建原始数据的副本，所有记录都保留
    result_df = original_df.copy()

    # 收集所有问题记录的索引
    problem_indices = set()
    for problem in problematic_records:
        # 找到原始记录在DataFrame中的索引
        current_record = problem['current_record']
        matching_rows = original_df[
            (original_df['姓名'] == current_record['姓名']) &
            (original_df['就诊日期'] == current_record['就诊日期']) &
            (original_df['治疗'] == current_record['治疗'])
            ]
        problem_indices.update(matching_rows.index.tolist())

    # 为所有记录添加分析信息列
    analysis_info = []
    for idx in result_df.index:
        if idx in problem_indices:
            # 找到对应的问题信息
            record = original_df.loc[idx]
            matching_problem = None
            for problem in problematic_records:
                current_record = problem['current_record']
                if (record['姓名'] == current_record['姓名'] and
                        str(record['就诊日期']) == str(current_record['就诊日期']) and
                        str(record['治疗']) == str(current_record['治疗'])):
                    matching_problem = problem
                    break

            if matching_problem:
                info = f"有冲突 - 最大数字: {matching_problem['max_number']}, " \
                       f"检查期间: {matching_problem['check_period']}, " \
                       f"冲突日期: {matching_problem['conflicting_dates']}"
            else:
                info = "有冲突 - 分析信息未找到"
        else:
            info = "无冲突"

        analysis_info.append(info)

    result_df['问题分析'] = analysis_info

    # 保存到Excel
    result_df.to_excel(output_file, index=False)
    logger.info(f"已保存{len(result_df)}条记录到 {output_file}（其中{len(problem_indices)}条有冲突）")


def save_error_log(processing_errors, error_log_file):
    """保存处理异常日志到Excel"""
    if not processing_errors:
        return

    error_df = pd.DataFrame(processing_errors)
    error_df.to_excel(error_log_file, index=False)
    logger.info(f"已保存{len(processing_errors)}条异常记录到 {error_log_file}")


def main():
    """主函数"""
    input_file = '../../data/case_data/0823-0825治疗-友杨.xlsx'  # 输入文件路径
    output_file = '../../data/result/treatment_validation_results_0823-0825治疗-友杨.xlsx'  # 问题记录输出文件
    error_log_file = '../../data/result/treatment_validation_results_0823-0825治疗-友杨.log'  # 异常日志输出文件

    try:
        # 读取数据
        df = read_excel_file(input_file)

        # 查找问题记录
        logger.info("开始分析治疗记录...")
        problematic_records, processing_errors = find_problematic_records(df)

        # 保存结果
        save_problematic_records(problematic_records, df, output_file)
        save_error_log(processing_errors, error_log_file)

        # 输出统计信息
        logger.info(f"分析完成 - 总记录数: {len(df)}, "
                    f"问题记录数: {len(problematic_records)}, "
                    f"处理异常数: {len(processing_errors)}")

    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise


if __name__ == "__main__":
    main()