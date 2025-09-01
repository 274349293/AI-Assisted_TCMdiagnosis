import pandas as pd
import json
import logging
import re
from typing import Dict, List, Tuple, Any, Set
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Step1DiseaseValidator:
    """
    Step1: 疾病诊断验证器
    验证门诊病历中的诊断内容是否符合要求
    """

    def __init__(self, disease_catalog_path: str, disease_syndrome_path: str, syndromes_mapping_path: str = None):
        """
        初始化验证器

        Args:
            disease_catalog_path: 门诊疾病目录JSON文件路径
            disease_syndrome_path: 中医疾病-证型对应关系JSON文件路径
            syndromes_mapping_path: 证型层级映射JSON文件路径（可选）
        """
        self.disease_catalog = self._load_disease_catalog(disease_catalog_path)
        self.disease_syndrome_map = self._load_disease_syndrome_map(disease_syndrome_path)
        self.syndromes_mapping = self._load_syndromes_mapping(syndromes_mapping_path)

        # 构建分类索引
        self.tcm_diseases = set()  # 中医疾病
        self.syndromes = set()  # 证型
        self.western_diagnoses = set()  # 西医诊断
        self.tcm_western_common = set()  # 中西医同名疾病

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

    def _load_syndromes_mapping(self, mapping_path: str = None) -> Dict:
        """加载证型层级映射关系"""
        if not mapping_path:
            logger.info("未配置证型层级映射文件路径，跳过层级匹配功能")
            return {}

        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            mappings = data.get('mappings', [])
            logger.info(f"成功加载证型层级映射: {len(mappings)} 条映射关系")
            return data
        except Exception as e:
            logger.error(f"加载证型层级映射失败: {str(e)}")
            logger.warning("将跳过层级匹配功能")
            return {}

    def _extract_alternative_names(self, disease_name: str) -> List[str]:
        """
        从疾病名中提取可选词

        例如：流痰(可选词：骨痨；穿骨流注) -> ["流痰(可选词：骨痨；穿骨流注)", "流痰", "骨痨", "穿骨流注"]

        Args:
            disease_name: 原始疾病名称

        Returns:
            包含原始名称、基础名称和所有可选词的列表
        """
        names = [disease_name]  # 保留原始名称

        # 匹配括号内的可选词模式
        pattern = r'\(可选词：([^)]+)\)'
        match = re.search(pattern, disease_name)

        if match:
            # 提取去括号的基础名称
            base_name = re.sub(r'\(可选词：[^)]+\)', '', disease_name).strip()
            if base_name and base_name not in names:
                names.append(base_name)

            # 提取括号内的内容
            alternatives_text = match.group(1)

            # 按分号或中文分号分割
            alternatives = re.split('[;；]', alternatives_text)

            # 清理空白字符并添加到列表
            for alt in alternatives:
                alt = alt.strip()
                if alt and alt not in names:
                    names.append(alt)

            logger.debug(f"提取可选词: {disease_name} -> {names}")

        return names

    def _get_syndrome_descendants(self, target_syndrome: str) -> List[str]:
        """
        获取指定证型的所有下级证型（用于大类证型匹配）

        Args:
            target_syndrome: 目标证型名称

        Returns:
            下级证型名称列表
        """
        descendants = []
        mappings = self.syndromes_mapping.get('mappings', [])

        for mapping in mappings:
            ancestors = mapping.get('ancestors', [])
            source_name = mapping.get('source', {}).get('name', '')

            if not source_name:
                continue

            # 检查target_syndrome是否在ancestors中（需要处理可选词）
            for ancestor in ancestors:
                ancestor_name = ancestor.get('name', '')
                if not ancestor_name:
                    continue

                # 提取ancestor的所有可选词进行匹配
                ancestor_names = self._extract_alternative_names(ancestor_name)
                target_names = self._extract_alternative_names(target_syndrome)

                # 如果有任何匹配，则source是target的下级证型
                if any(target_name in ancestor_names for target_name in target_names):
                    # 提取source的所有可选词
                    source_names = self._extract_alternative_names(source_name)
                    descendants.extend(source_names)
                    break

        # 去重并返回
        return list(set(descendants))

    def _get_syndrome_ancestors(self, source_syndrome: str) -> List[str]:
        """
        获取指定证型的所有上级证型（用于小类证型匹配）

        Args:
            source_syndrome: 源证型名称

        Returns:
            上级证型名称列表
        """
        ancestors = []
        mappings = self.syndromes_mapping.get('mappings', [])

        for mapping in mappings:
            source_name = mapping.get('source', {}).get('name', '')
            if not source_name:
                continue

            # 提取source的所有可选词进行匹配
            source_names = self._extract_alternative_names(source_name)
            target_names = self._extract_alternative_names(source_syndrome)

            # 如果source匹配target，获取其ancestors
            if any(target_name in source_names for target_name in target_names):
                mapping_ancestors = mapping.get('ancestors', [])
                for ancestor in mapping_ancestors:
                    ancestor_name = ancestor.get('name', '')
                    if ancestor_name:
                        # 提取ancestor的所有可选词
                        ancestor_names = self._extract_alternative_names(ancestor_name)
                        ancestors.extend(ancestor_names)
                break

        # 去重并返回
        return list(set(ancestors))

    def _build_classification_index(self):
        """构建分类索引"""
        items = self.disease_catalog.get('items', [])

        for item in items:
            name = item.get('name', '')
            item_type = item.get('type', '')

            if not name:
                continue

            if item_type == '中医疾病':
                self.tcm_diseases.add(name)

            elif item_type == '证候':
                self.syndromes.add(name)

            elif item_type == '西医诊断':
                self.western_diagnoses.add(name)

            elif item_type == '中西医同名疾病':
                # 中西医同名疾病同时加入中医疾病和西医诊断
                self.tcm_diseases.add(name)
                self.western_diagnoses.add(name)
                self.tcm_western_common.add(name)
                logger.debug(f"中西医同名疾病: {name}")

        logger.info(f"分类索引构建完成:")
        logger.info(f"  中医疾病: {len(self.tcm_diseases)} 个")
        logger.info(f"  证候: {len(self.syndromes)} 个")
        logger.info(f"  西医诊断: {len(self.western_diagnoses)} 个")
        logger.info(f"  中西医同名疾病: {len(self.tcm_western_common)} 个")

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
            matched = False

            # 检查是否为中医疾病（包括中西医同名疾病）
            if item in self.tcm_diseases:
                classification["中医疾病"].append(item)
                matched = True

            # 检查是否为证型
            if item in self.syndromes:
                classification["中医证型"].append(item)
                matched = True

            # 检查是否为西医诊断（包括中西医同名疾病）
            if item in self.western_diagnoses:
                classification["西医诊断"].append(item)
                matched = True

            # 如果都没有匹配到，则为未匹配项
            if not matched:
                classification["未匹配项"].append(item)

        # 记录中西医同名疾病的日志
        for item in items:
            if item in self.tcm_western_common:
                logger.debug(f"识别到中西医同名疾病: {item}")

        # 新增：后处理拆分可选词
        for category in ["中医疾病", "中医证型"]:
            expanded_items = []
            for item in classification[category]:
                expanded_items.append(item)  # 保留原始名称
                # 提取并添加可选词
                alternative_names = self._extract_alternative_names(item)
                for alt_name in alternative_names[1:]:  # 跳过第一个（原始名称）
                    if alt_name not in expanded_items:
                        expanded_items.append(alt_name)
                        logger.debug(f"从 '{item}' 中提取可选词: '{alt_name}'")
            classification[category] = expanded_items

        return classification

    def validate_tcm_disease_syndrome_match(self, tcm_diseases: List[str], syndromes: List[str]) -> Tuple[bool, str]:
        """
        验证中医疾病与证型的匹配关系
        支持双向层级匹配：大类证型匹配下级证型，小类证型匹配上级证型

        Args:
            tcm_diseases: 中医疾病列表
            syndromes: 证型列表

        Returns:
            (是否匹配, 详细说明)
        """
        if not tcm_diseases or not syndromes:
            return True, "无需验证中医疾病-证型匹配关系"

        unmatched_pairs = []
        match_details = []

        # 构建原始疾病名到匹配用名称的映射
        original_disease_map = {}
        for original_name in self.disease_syndrome_map.keys():
            alternative_names = self._extract_alternative_names(original_name)
            for alt_name in alternative_names:
                original_disease_map[alt_name] = original_name

        for tcm_disease in tcm_diseases:
            # 找到对应的原始疾病名（用于在映射表中查找）
            original_disease_name = original_disease_map.get(tcm_disease, tcm_disease)

            # 查找该疾病对应的证型
            disease_syndromes = self.disease_syndrome_map.get(original_disease_name, [])

            if not disease_syndromes:
                # 疾病-证型映射中没有该疾病，跳过验证
                logger.debug(f"疾病-证型映射中未找到: {tcm_disease} (原始名: {original_disease_name})")
                continue

            # 展开疾病支持的证型（处理可选词）
            expanded_disease_syndromes = set()
            for syndrome in disease_syndromes:
                syndrome_alternatives = self._extract_alternative_names(syndrome)
                expanded_disease_syndromes.update(syndrome_alternatives)

            # 检查每个证型是否匹配
            for syndrome in syndromes:
                syndrome_matched = False
                match_type = ""

                # 展开病历证型（处理可选词）
                syndrome_alternatives = self._extract_alternative_names(syndrome)

                # 第一层：直接匹配
                for syndrome_alt in syndrome_alternatives:
                    if syndrome_alt in expanded_disease_syndromes:
                        syndrome_matched = True
                        match_type = "直接匹配"
                        match_details.append(f"{tcm_disease} ↔ {syndrome} (直接匹配: {syndrome_alt})")
                        logger.debug(f"直接匹配成功: {tcm_disease} ↔ {syndrome} ({syndrome_alt})")
                        break

                # 第二层：层级匹配（如果直接匹配失败）
                if not syndrome_matched and self.syndromes_mapping:
                    # 情况1：病历证型是大类，查找其下级证型
                    descendants = self._get_syndrome_descendants(syndrome)
                    if descendants:
                        for descendant in descendants:
                            if descendant in expanded_disease_syndromes:
                                syndrome_matched = True
                                match_type = "大类证型匹配"
                                match_details.append(
                                    f"{tcm_disease} ↔ {syndrome} (大类匹配: 通过下级证型'{descendant}')")
                                logger.info(
                                    f"大类证型匹配成功: {tcm_disease} ↔ {syndrome} (通过下级证型'{descendant}')")
                                break

                    # 情况2：病历证型是小类，查找其上级证型
                    if not syndrome_matched:
                        ancestors = self._get_syndrome_ancestors(syndrome)
                        if ancestors:
                            for ancestor in ancestors:
                                if ancestor in expanded_disease_syndromes:
                                    syndrome_matched = True
                                    match_type = "小类证型匹配"
                                    match_details.append(
                                        f"{tcm_disease} ↔ {syndrome} (小类匹配: 通过上级证型'{ancestor}')")
                                    logger.info(
                                        f"小类证型匹配成功: {tcm_disease} ↔ {syndrome} (通过上级证型'{ancestor}')")
                                    break

                # 如果所有匹配都失败
                if not syndrome_matched:
                    unmatched_pairs.append(f"{tcm_disease}与{syndrome}")

        if unmatched_pairs:
            return False, f"中医疾病与证型不匹配: {', '.join(unmatched_pairs)}"
        else:
            details_str = "; ".join(match_details) if match_details else "匹配正确"
            return True, f"中医疾病与证型匹配正确 - {details_str}"

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
        if total_records > 0:
            passed_records = len([r for r in all_results if r["step1验证结果"]["是否符合要求"]])
            failed_records = total_records - passed_records

            logger.info("=" * 60)
            logger.info("Step1验证总体统计:")
            logger.info(f"  处理文件数: {len(file_paths)}")
            logger.info(f"  总记录数: {total_records}")
            logger.info(f"  通过验证: {passed_records} ({passed_records / total_records * 100:.1f}%)")
            logger.info(f"  未通过验证: {failed_records} ({failed_records / total_records * 100:.1f}%)")
            logger.info("=" * 60)
        else:
            logger.warning("没有成功处理任何记录")

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
        disease_syndrome_path=config["中医疾病-证型"],
        syndromes_mapping_path=config.get("证型层级映射", None)  # 可选配置
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
