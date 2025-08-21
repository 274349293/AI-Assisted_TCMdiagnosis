import pandas as pd
import json
import logging
from typing import Dict, List, Tuple, Any
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Step1DiseaseValidator:
    """
    Step1: 疾病诊断验证器
    验证门诊病历中的诊断内容是否符合要求
    """

    def __init__(self, disease_catalog_path: str, disease_syndrome_path: str):
        """
        初始化验证器

        Args:
            disease_catalog_path: 门诊疾病目录JSON文件路径
            disease_syndrome_path: 中医疾病-证型对应关系JSON文件路径
        """
        self.disease_catalog = self._load_disease_catalog(disease_catalog_path)
        self.disease_syndrome_map = self._load_disease_syndrome_map(disease_syndrome_path)

        # 构建分类索引
        self.tcm_diseases = set()  # 中医疾病
        self.syndromes = set()  # 证型
        self.western_diagnoses = set()  # 西医诊断

        self._build_classification_index()

    def _load_disease_catalog(self, file_path: str) -> Dict:
        """加载门诊疾病目录"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功加载门诊疾病目录: {len(data.get('items', []))} 条记录")
            return data
        except Exception as e:
            logger.error(f"加载门诊疾病目录失败: {str(e)}")
            raise

    def _load_disease_syndrome_map(self, file_path: str) -> Dict:
        """加载中医疾病-证型对应关系"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功加载中医疾病-证型对应关系: {len(data)} 个疾病")
            return data
        except Exception as e:
            logger.error(f"加载中医疾病-证型对应关系失败: {str(e)}")
            raise

    def _build_classification_index(self):
        """构建分类索引"""
        items = self.disease_catalog.get('items', [])

        for item in items:
            name = item.get('name', '')
            item_type = item.get('type', '')

            if item_type == '中医疾病':
                self.tcm_diseases.add(name)
            elif item_type == '证候':
                self.syndromes.add(name)
            elif item_type == '西医诊断':
                self.western_diagnoses.add(name)

        logger.info(f"分类索引构建完成:")
        logger.info(f"  中医疾病: {len(self.tcm_diseases)} 个")
        logger.info(f"  证候: {len(self.syndromes)} 个")
        logger.info(f"  西医诊断: {len(self.western_diagnoses)} 个")

    def classify_diagnosis_items(self, diagnosis_text: str) -> Dict[str, List[str]]:
        """
        分类诊断内容中的各个项目

        Args:
            diagnosis_text: 诊断文本，逗号分隔

        Returns:
            分类结果字典
        """
        if not diagnosis_text or pd.isna(diagnosis_text):
            return {
                "中医疾病": [],
                "中医证型": [],
                "西医诊断": [],
                "未匹配项": []
            }

        # 按逗号分割诊断项
        items = [item.strip() for item in str(diagnosis_text).split(',') if item.strip()]

        classification = {
            "中医疾病": [],
            "中医证型": [],
            "西医诊断": [],
            "未匹配项": []
        }

        for item in items:
            if item in self.tcm_diseases:
                classification["中医疾病"].append(item)
            elif item in self.syndromes:
                classification["中医证型"].append(item)
            elif item in self.western_diagnoses:
                classification["西医诊断"].append(item)
            else:
                classification["未匹配项"].append(item)

        return classification

    def validate_tcm_disease_syndrome_match(self, tcm_diseases: List[str], syndromes: List[str]) -> Tuple[bool, str]:
        """
        验证中医疾病与证型的匹配关系

        Args:
            tcm_diseases: 中医疾病列表
            syndromes: 证型列表

        Returns:
            (是否匹配, 详细说明)
        """
        if not tcm_diseases or not syndromes:
            return True, "无需验证中医疾病-证型匹配关系"

        unmatched_pairs = []

        for tcm_disease in tcm_diseases:
            # 查找该疾病对应的证型
            disease_syndromes = self.disease_syndrome_map.get(tcm_disease, [])

            if not disease_syndromes:
                # 疾病-证型映射中没有该疾病，跳过验证
                continue

            # 检查每个证型是否属于该疾病
            for syndrome in syndromes:
                if syndrome not in disease_syndromes:
                    unmatched_pairs.append(f"{tcm_disease}与{syndrome}")

        if unmatched_pairs:
            return False, f"中医疾病与证型不匹配: {', '.join(unmatched_pairs)}"
        else:
            return True, "中医疾病与证型匹配正确"

    def validate_single_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证单条病历记录

        Args:
            record: 病历记录字典

        Returns:
            验证结果字典
        """
        diagnosis_text = record.get('诊断', '')

        # 步骤1: 分类诊断项
        classification = self.classify_diagnosis_items(diagnosis_text)

        # 步骤2: 验证必要条件
        validation_result = {
            "是否符合要求": True,
            "不符合原因": []
        }

        # 检查是否包含至少一个中医疾病
        if not classification["中医疾病"]:
            validation_result["是否符合要求"] = False
            validation_result["不符合原因"].append("缺少中医疾病")

        # 检查是否包含至少一个证型
        if not classification["中医证型"]:
            validation_result["是否符合要求"] = False
            validation_result["不符合原因"].append("缺少中医证型")

        # 检查是否包含至少一个西医诊断
        if not classification["西医诊断"]:
            validation_result["是否符合要求"] = False
            validation_result["不符合原因"].append("缺少西医诊断")

        # 检查是否有未匹配项
        if classification["未匹配项"]:
            validation_result["是否符合要求"] = False
            validation_result["不符合原因"].append(f"存在未匹配项: {', '.join(classification['未匹配项'])}")

        # 步骤3: 验证中医疾病与证型匹配关系
        if classification["中医疾病"] and classification["中医证型"]:
            is_match, match_detail = self.validate_tcm_disease_syndrome_match(
                classification["中医疾病"],
                classification["中医证型"]
            )
            if not is_match:
                validation_result["是否符合要求"] = False
                validation_result["不符合原因"].append(match_detail)

        # 整理不符合原因
        if validation_result["不符合原因"]:
            validation_result["不符合原因"] = "; ".join(validation_result["不符合原因"])
        else:
            validation_result["不符合原因"] = ""

        # 构建完整结果
        result = {
            "原始数据": record,
            "诊断分类": classification,
            "step1验证结果": validation_result
        }

        return result

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

            results = []

            for index, row in df.iterrows():
                # 将pandas Series转换为字典
                record = row.to_dict()

                # 验证单条记录
                validation_result = self.validate_single_record(record)

                # 添加记录编号
                validation_result["记录编号"] = index + 1
                validation_result["源文件"] = os.path.basename(file_path)

                results.append(validation_result)

                # 日志输出进度
                if (index + 1) % 100 == 0:
                    logger.info(f"已处理 {index + 1}/{len(df)} 条记录")

            # 统计验证结果
            total_records = len(results)
            passed_records = len([r for r in results if r["step1验证结果"]["是否符合要求"]])
            failed_records = total_records - passed_records

            logger.info(f"文件 {os.path.basename(file_path)} 验证完成:")
            logger.info(f"  总记录数: {total_records}")
            logger.info(f"  通过验证: {passed_records} ({passed_records / total_records * 100:.1f}%)")
            logger.info(f"  未通过验证: {failed_records} ({failed_records / total_records * 100:.1f}%)")

            return results

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
        passed_records = len([r for r in all_results if r["step1验证结果"]["是否符合要求"]])
        failed_records = total_records - passed_records

        logger.info("=" * 60)
        logger.info("Step1验证总体统计:")
        logger.info(f"  处理文件数: {len(file_paths)}")
        logger.info(f"  总记录数: {total_records}")
        logger.info(f"  通过验证: {passed_records} ({passed_records / total_records * 100:.1f}%)")
        logger.info(f"  未通过验证: {failed_records} ({failed_records / total_records * 100:.1f}%)")
        logger.info("=" * 60)

        return all_results


def run_step1_validation(config_path: str) -> List[Dict[str, Any]]:
    """
    运行Step1验证的主函数

    Args:
        config_path: 配置文件路径

    Returns:
        验证结果列表
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 创建验证器
    validator = Step1DiseaseValidator(
        disease_catalog_path=config["门诊疾病目录"],
        disease_syndrome_path=config["中医疾病-证型"]
    )

    # 执行验证
    results = validator.validate_multiple_files(config["input_files"])

    return results


if __name__ == "__main__":
    # 测试代码
    config_path = "config.json"

    try:
        results = run_step1_validation(config_path)
        logger.info(f"Step1验证完成，共处理 {len(results)} 条记录")
    except Exception as e:
        logger.error(f"Step1验证执行失败: {str(e)}")
