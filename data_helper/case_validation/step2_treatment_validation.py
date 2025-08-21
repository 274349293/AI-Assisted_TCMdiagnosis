import json
import logging
import re
import os
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import pandas as pd

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Step2TreatmentValidator:
    """
    Step2: 治疗项冲突验证器
    验证患者治疗项中的天数与就诊记录时间间隔是否合理
    """

    def __init__(self):
        """初始化验证器"""
        self.skip_keywords = ['中医辨证论治', '红外线治疗(TDP)', '煎药机煎药(门诊)']

    def should_skip_treatment_item(self, item: str) -> bool:
        """
        判断治疗项是否应该跳过

        Args:
            item: 治疗项内容

        Returns:
            是否跳过该治疗项
        """
        if not item or not item.strip():
            return True

        # 检查是否包含跳过的关键词
        for keyword in self.skip_keywords:
            if keyword in item:
                return True

        # 检查最后两个字是否是"穴位"
        item_stripped = item.strip()
        if len(item_stripped) >= 2 and item_stripped.endswith('穴位'):
            return True

        return False

    def extract_numbers_from_treatment_item(self, item: str) -> Tuple[int, List[str]]:
        """
        从治疗项中提取阿拉伯数字，只取整数部分

        Args:
            item: 治疗项内容

        Returns:
            (提取的数字, 异常信息列表)
        """
        exceptions = []

        # 查找所有数字（包括小数）
        numbers = re.findall(r'\d+\.?\d*', item)

        if len(numbers) == 0:
            exceptions.append(f"治疗项缺少数字: {item}")
            return 0, exceptions
        elif len(numbers) > 1:
            exceptions.append(f"治疗项包含多个数字: {item}, 找到的数字: {numbers}")
            return 0, exceptions

        # 只取整数部分
        try:
            number = float(numbers[0])
            return int(number), exceptions
        except ValueError:
            exceptions.append(f"治疗项数字格式错误: {item}, 数字: {numbers[0]}")
            return 0, exceptions

    def parse_treatment_column(self, treatment_text: str) -> Tuple[int, List[str]]:
        """
        解析治疗列，返回有效治疗项中的最大数字

        Args:
            treatment_text: 治疗列内容

        Returns:
            (最大数字, 异常信息列表)
        """
        if pd.isna(treatment_text) or not str(treatment_text).strip():
            return 0, ["治疗列为空"]

        # 按照 (1), (2), (3) 等分割治疗项
        items = re.split(r'\(\d+\)', str(treatment_text))
        items = [item.strip() for item in items if item.strip()]

        valid_numbers = []
        all_exceptions = []

        for item in items:
            # 跳过不需要处理的项
            if self.should_skip_treatment_item(item):
                logger.debug(f"跳过治疗项: {item}")
                continue

            # 提取数字
            number, exceptions = self.extract_numbers_from_treatment_item(item)

            if exceptions:
                all_exceptions.extend(exceptions)
            else:
                valid_numbers.append(number)

        if not valid_numbers and not all_exceptions:
            # 所有项都被跳过，这是正常情况
            return 0, []

        if not valid_numbers and all_exceptions:
            # 有非跳过项但都有问题
            return 0, all_exceptions

        return max(valid_numbers), all_exceptions

    def parse_visit_date(self, date_str: str) -> Optional[datetime]:
        """
        解析就诊日期

        Args:
            date_str: 日期字符串

        Returns:
            解析后的datetime对象，解析失败返回None
        """
        if pd.isna(date_str):
            return None

        try:
            # 尝试多种日期格式
            date_formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
                '%Y/%m/%d %H:%M:%S',
                '%Y/%m/%d',
                '%m/%d/%Y',
                '%d/%m/%Y'
            ]

            date_str = str(date_str).strip()

            for fmt in date_formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue

            # 如果都失败了，尝试pandas的解析
            return pd.to_datetime(date_str)

        except Exception as e:
            logger.warning(f"日期解析失败: {date_str}, 错误: {str(e)}")
            return None

    def check_patient_visit_conflicts(self, patient_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        检查单个患者的就诊记录冲突

        Args:
            patient_records: 同一患者的所有就诊记录

        Returns:
            更新后的记录列表（包含step2验证结果）
        """
        # 按就诊日期排序
        valid_records = []

        for record in patient_records:
            visit_date_str = record.get('原始数据', {}).get('就诊日期', '')
            visit_date = self.parse_visit_date(visit_date_str)

            if visit_date is None:
                # 日期解析失败，添加异常信息
                record['step2验证结果'] = {
                    "是否有治疗冲突": False,
                    "最大治疗天数": 0,
                    "冲突的就诊记录": [],
                    "异常信息": [f"就诊日期解析失败: {visit_date_str}"]
                }
                valid_records.append(record)
                continue

            record['解析后就诊日期'] = visit_date
            valid_records.append(record)

        # 按日期排序
        valid_records.sort(key=lambda x: x.get('解析后就诊日期', datetime.min))

        # 检查每条记录的治疗冲突
        for i, record in enumerate(valid_records):
            if '解析后就诊日期' not in record:
                continue

            visit_date = record['解析后就诊日期']
            treatment_text = record.get('原始数据', {}).get('治疗', '')

            # 解析治疗项
            max_days, exceptions = self.parse_treatment_column(treatment_text)

            # 如果最大天数为0（所有项都被跳过或没有有效治疗项），跳过检查
            if max_days == 0:
                record['step2验证结果'] = {
                    "是否有治疗冲突": False,
                    "最大治疗天数": 0,
                    "冲突的就诊记录": [],
                    "异常信息": exceptions
                }
                continue

            # 计算检查时间范围（包含当天的max_days天）
            check_start_date = visit_date.date()
            check_end_date = check_start_date + timedelta(days=max_days - 1)

            # 查找在检查时间范围内的其他就诊记录
            conflicting_visits = []
            for j, other_record in enumerate(valid_records):
                if i == j or '解析后就诊日期' not in other_record:
                    continue

                other_visit_date = other_record['解析后就诊日期'].date()

                # 检查是否在时间范围内
                if check_start_date <= other_visit_date <= check_end_date:
                    conflicting_visits.append(other_visit_date.strftime('%Y-%m-%d'))

            # 构建验证结果
            has_conflict = len(conflicting_visits) > 0
            record['step2验证结果'] = {
                "是否有治疗冲突": has_conflict,
                "最大治疗天数": max_days,
                "冲突的就诊记录": conflicting_visits,
                "异常信息": exceptions
            }

            if has_conflict:
                logger.info(f"发现治疗冲突 - 患者: {record.get('原始数据', {}).get('姓名', 'Unknown')}, "
                            f"就诊日期: {visit_date.strftime('%Y-%m-%d')}, "
                            f"最大治疗天数: {max_days}, "
                            f"冲突日期: {conflicting_visits}")

        return valid_records

    def validate_step1_results(self, step1_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        验证step1结果中的治疗项冲突

        Args:
            step1_results: step1的验证结果列表

        Returns:
            添加了step2验证结果的记录列表
        """
        logger.info(f"开始Step2治疗项冲突验证，共 {len(step1_results)} 条记录")

        # 按患者姓名分组
        patient_groups = defaultdict(list)

        for record in step1_results:
            patient_name = record.get('原始数据', {}).get('姓名', '')
            if patient_name:
                patient_groups[patient_name].append(record)
            else:
                # 姓名为空的记录单独处理
                record['step2验证结果'] = {
                    "是否有治疗冲突": False,
                    "最大治疗天数": 0,
                    "冲突的就诊记录": [],
                    "异常信息": ["患者姓名为空"]
                }

        logger.info(f"按患者分组完成，共 {len(patient_groups)} 个患者")

        # 验证每个患者的记录
        all_validated_records = []

        for patient_name, patient_records in patient_groups.items():
            logger.debug(f"验证患者: {patient_name}, 记录数: {len(patient_records)}")

            validated_records = self.check_patient_visit_conflicts(patient_records)
            all_validated_records.extend(validated_records)

        # 添加姓名为空的记录
        for record in step1_results:
            if not record.get('原始数据', {}).get('姓名', ''):
                all_validated_records.append(record)

        # 统计结果
        total_records = len(all_validated_records)
        conflict_records = len([r for r in all_validated_records
                                if r.get('step2验证结果', {}).get('是否有治疗冲突', False)])

        logger.info(f"Step2验证完成:")
        logger.info(f"  总记录数: {total_records}")
        logger.info(f"  有治疗冲突: {conflict_records}")
        logger.info(f"  无治疗冲突: {total_records - conflict_records}")

        # 清理临时字段（防止JSON序列化错误）
        for record in all_validated_records:
            if '解析后就诊日期' in record:
                del record['解析后就诊日期']

        return all_validated_records


def load_step1_results(file_path: str = None, output_dir: str = None) -> List[Dict[str, Any]]:
    """
    加载step1的验证结果

    Args:
        file_path: 指定的文件路径
        output_dir: 输出目录，用于查找最新的step1结果文件

    Returns:
        step1验证结果列表
    """
    if file_path and os.path.exists(file_path):
        target_file = file_path
    elif output_dir:
        # 查找最新的step1结果文件
        latest_file = os.path.join(output_dir, "step1_results_latest.json")
        if os.path.exists(latest_file):
            target_file = latest_file
        else:
            # 查找带时间戳的最新文件
            step1_files = glob.glob(os.path.join(output_dir, "step1_results_*.json"))
            if not step1_files:
                raise FileNotFoundError(f"在目录 {output_dir} 中未找到step1结果文件")
            target_file = max(step1_files, key=os.path.getctime)
    else:
        raise ValueError("必须提供file_path或output_dir参数")

    logger.info(f"加载step1结果文件: {target_file}")

    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"成功加载step1结果: {len(data)} 条记录")
        return data

    except Exception as e:
        logger.error(f"加载step1结果失败: {str(e)}")
        raise


def run_step2_validation(step1_results: List[Dict[str, Any]] = None,
                         step1_file_path: str = None,
                         output_dir: str = None) -> List[Dict[str, Any]]:
    """
    运行Step2验证的主函数

    Args:
        step1_results: step1验证结果（流水线传递）
        step1_file_path: step1结果文件路径（独立运行）
        output_dir: 输出目录（查找最新step1文件）

    Returns:
        step2验证结果列表
    """
    logger.info("=" * 80)
    logger.info("开始执行 Step2: 治疗项冲突验证")
    logger.info("=" * 80)

    # 获取step1结果
    if step1_results is not None:
        logger.info("使用流水线传递的step1结果")
        data = step1_results
    else:
        logger.info("从文件加载step1结果")
        data = load_step1_results(step1_file_path, output_dir)

    # 创建验证器并执行验证
    validator = Step2TreatmentValidator()
    results = validator.validate_step1_results(data)

    logger.info("Step2验证执行完成")
    return results


if __name__ == "__main__":
    # 独立运行时的测试代码
    try:
        # 可以指定具体文件路径
        # results = run_step2_validation(step1_file_path="../../data/result/step1_results_20240101_120000.json")

        # 或者指定输出目录，自动查找最新文件
        results = run_step2_validation(output_dir="../../data/result/")

        logger.info(f"Step2验证完成，共处理 {len(results)} 条记录")

        # 可以在这里添加保存结果的代码
        # with open("step2_test_results.json", 'w', encoding='utf-8') as f:
        #     json.dump(results, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"Step2验证执行失败: {str(e)}")