import pandas as pd
import json
import logging
import re
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TreatmentValidator:
    """
    治疗项冲突验证器
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
            更新后的记录列表（包含治疗验证结果）
        """
        # 按就诊日期排序
        valid_records = []

        for record in patient_records:
            visit_date_str = record.get('就诊日期', '')
            visit_date = self.parse_visit_date(visit_date_str)

            if visit_date is None:
                # 日期解析失败，添加异常信息
                record['治疗验证结果'] = {
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
            treatment_text = record.get('治疗', '')

            # 解析治疗项
            max_days, exceptions = self.parse_treatment_column(treatment_text)

            # 如果最大天数为0（所有项都被跳过或没有有效治疗项），跳过检查
            if max_days == 0:
                record['治疗验证结果'] = {
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
            record['治疗验证结果'] = {
                "是否有治疗冲突": has_conflict,
                "最大治疗天数": max_days,
                "冲突的就诊记录": conflicting_visits,
                "异常信息": exceptions
            }

            if has_conflict:
                logger.info(f"发现治疗冲突 - 患者: {record.get('姓名', 'Unknown')}, "
                            f"就诊日期: {visit_date.strftime('%Y-%m-%d')}, "
                            f"最大治疗天数: {max_days}, "
                            f"冲突日期: {conflicting_visits}")

        # 清理临时字段
        for record in valid_records:
            if '解析后就诊日期' in record:
                del record['解析后就诊日期']

        return valid_records

    def validate_excel_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        验证整个Excel文件

        Args:
            file_path: Excel文件路径

        Returns:
            验证结果列表
        """
        logger.info(f"开始验证Excel文件: {file_path}")

        try:
            # 读取Excel文件
            df = pd.read_excel(file_path)
            logger.info(f"成功读取 {len(df)} 条记录")

            # 按患者姓名分组
            patient_groups = defaultdict(list)

            for index, row in df.iterrows():
                # 将pandas Series转换为字典
                record = row.to_dict()

                # 添加记录信息
                record['记录编号'] = index + 1
                record['源文件'] = os.path.basename(file_path)

                patient_name = record.get('姓名', '')
                if patient_name:
                    patient_groups[patient_name].append(record)
                else:
                    # 姓名为空的记录单独处理
                    record['治疗验证结果'] = {
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
            for index, row in df.iterrows():
                record = row.to_dict()
                if not record.get('姓名', ''):
                    record['记录编号'] = index + 1
                    record['源文件'] = os.path.basename(file_path)
                    record['治疗验证结果'] = {
                        "是否有治疗冲突": False,
                        "最大治疗天数": 0,
                        "冲突的就诊记录": [],
                        "异常信息": ["患者姓名为空"]
                    }
                    all_validated_records.append(record)

            # 统计验证结果
            total_records = len(all_validated_records)
            conflict_records = len([r for r in all_validated_records
                                    if r.get('治疗验证结果', {}).get('是否有治疗冲突', False)])

            logger.info(f"文件 {os.path.basename(file_path)} 验证完成:")
            logger.info(f"  总记录数: {total_records}")
            logger.info(f"  有治疗冲突: {conflict_records}")
            logger.info(f"  无治疗冲突: {total_records - conflict_records}")

            return all_validated_records

        except Exception as e:
            logger.error(f"验证Excel文件失败: {str(e)}")
            raise

    def validate_multiple_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        验证多个Excel文件

        Args:
            file_paths: Excel文件路径列表

        Returns:
            所有文件的验证结果列表
        """
        logger.info(f"开始批量验证 {len(file_paths)} 个Excel文件")

        all_results = []

        for file_path in file_paths:
            try:
                file_results = self.validate_excel_file(file_path)
                all_results.extend(file_results)
            except Exception as e:
                logger.error(f"验证文件 {file_path} 时出错: {str(e)}")
                continue

        # 总体统计
        total_records = len(all_results)
        conflict_records = len([r for r in all_results
                                if r.get('治疗验证结果', {}).get('是否有治疗冲突', False)])

        logger.info("=" * 60)
        logger.info("治疗验证总体统计:")
        logger.info(f"  处理文件数: {len(file_paths)}")
        logger.info(f"  总记录数: {total_records}")
        logger.info(f"  有治疗冲突: {conflict_records} ({conflict_records / total_records * 100:.1f}%)")
        logger.info(
            f"  无治疗冲突: {total_records - conflict_records} ({(total_records - conflict_records) / total_records * 100:.1f}%)")
        logger.info("=" * 60)

        return all_results

    def save_results(self, results: List[Dict[str, Any]], output_dir: str):
        """
        保存验证结果到JSON和Excel文件

        Args:
            results: 验证结果列表
            output_dir: 输出目录
        """
        if not results:
            logger.warning("验证结果为空，跳过保存")
            return

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存JSON文件
        json_file = os.path.join(output_dir, f"treatment_validation_results_{timestamp}.json")

        try:
            # 重新整理结果格式
            formatted_results = []
            for result in results:
                formatted_result = {
                    "记录编号": result.get('记录编号', ''),
                    "源文件": result.get('源文件', ''),
                    "原始数据": {k: v for k, v in result.items()
                                 if k not in ['记录编号', '源文件', '治疗验证结果']},
                    "治疗验证结果": result.get('治疗验证结果', {})
                }
                formatted_results.append(formatted_result)

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_results, f, ensure_ascii=False, indent=2)

            logger.info(f"验证结果已保存到JSON: {json_file}")

            # 保存Excel文件
            self._save_to_excel(formatted_results, output_dir, timestamp)

        except Exception as e:
            logger.error(f"保存验证结果失败: {str(e)}")

    def _save_to_excel(self, results: List[Dict[str, Any]], output_dir: str, timestamp: str):
        """
        将结果保存为Excel格式

        Args:
            results: 验证结果列表
            output_dir: 输出目录
            timestamp: 时间戳
        """
        try:
            # 展平结果数据以便于在Excel中查看
            flattened_data = []

            for result in results:
                row = {
                    '记录编号': result.get('记录编号', ''),
                    '源文件': result.get('源文件', ''),
                    '姓名': result.get('原始数据', {}).get('姓名', ''),
                    '就诊日期': result.get('原始数据', {}).get('就诊日期', ''),
                    '治疗': result.get('原始数据', {}).get('治疗', ''),
                    '是否有治疗冲突': result.get('治疗验证结果', {}).get('是否有治疗冲突', ''),
                    '最大治疗天数': result.get('治疗验证结果', {}).get('最大治疗天数', ''),
                    '冲突的就诊记录': ', '.join(result.get('治疗验证结果', {}).get('冲突的就诊记录', [])),
                    '异常信息': '; '.join(result.get('治疗验证结果', {}).get('异常信息', [])),
                }

                # 添加其他原始数据字段
                for key, value in result.get('原始数据', {}).items():
                    if key not in ['姓名', '就诊日期', '治疗']:
                        row[f'原始_{key}'] = value

                flattened_data.append(row)

            # 保存为Excel
            df = pd.DataFrame(flattened_data)
            excel_file = os.path.join(output_dir, f"treatment_validation_results_{timestamp}.xlsx")
            df.to_excel(excel_file, index=False)

            logger.info(f"验证结果已保存为Excel: {excel_file}")

        except Exception as e:
            logger.error(f"保存Excel格式失败: {str(e)}")


def load_config(config_path: str = "config.json") -> Dict:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        raise


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("开始执行治疗项冲突验证")
    logger.info("=" * 80)

    config_file = "config.json"

    try:
        # 加载配置
        config = load_config(config_file)

        # 检查配置中是否有治疗验证文件配置
        if "treatment_validation_files" not in config:
            logger.error("配置文件中缺少 'treatment_validation_files' 字段")
            logger.info("请在config.json中添加treatment_validation_files字段，指定要验证的Excel文件")
            return

        input_files = config["treatment_validation_files"]
        output_dir = config.get("output_dir", "../../data/result/")

        # 检查输入文件是否存在
        for file_path in input_files:
            if not os.path.exists(file_path):
                logger.error(f"输入文件不存在: {file_path}")
                return

        # 创建验证器并执行验证
        validator = TreatmentValidator()
        results = validator.validate_multiple_files(input_files)

        # 保存结果
        validator.save_results(results, output_dir)

        logger.info("=" * 80)
        logger.info("治疗项冲突验证执行完成")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")


if __name__ == "__main__":
    main()
