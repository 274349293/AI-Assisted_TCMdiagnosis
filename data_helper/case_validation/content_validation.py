# 打印总结
for step, stats in summary["步骤结果"].items():
    if step == "step1":
        logger.info(f"{step.upper()} 统计: 总数={stats['总记录数']}, "
                    f"通过={stats['通过数']}, 失败={stats['失败数']}, "
                    f"通过率={stats['通过率']}")
    # elif step == "step2":
    #     logger.info(f"{step.upper()} 统计: 总数={stats['总记录数']}, "
    #                f"有冲突={stats['有冲突数']}, 无冲突={stats['无冲突数']}, "
    #                fimport json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any

# 导入各个步骤的验证模块
from step1_disease_validation import run_step1_validation

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContentValidationPipeline:
    """
    病历内容验证流水线
    按步骤执行各种验证任务
    """

    def __init__(self, config_path: str):
        """
        初始化验证流水线

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.results = {}

    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"成功加载配置文件: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            raise

    def _validate_config(self) -> bool:
        """验证配置文件的有效性"""
        required_keys = ["input_files", "门诊疾病目录", "中医疾病-证型", "output_dir"]

        for key in required_keys:
            if key not in self.config:
                logger.error(f"配置文件缺少必要字段: {key}")
                return False

        # 检查输入文件是否存在
        for file_path in self.config["input_files"]:
            if not os.path.exists(file_path):
                logger.error(f"输入文件不存在: {file_path}")
                return False

        # 检查依赖文件是否存在
        for file_key in ["门诊疾病目录", "中医疾病-证型"]:
            file_path = self.config[file_key]
            if not os.path.exists(file_path):
                logger.error(f"依赖文件不存在 ({file_key}): {file_path}")
                return False

        # 创建输出目录
        output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"输出目录: {output_dir}")

        return True

    def run_step1(self) -> List[Dict[str, Any]]:
        """
        执行Step1: 疾病诊断验证

        Returns:
            Step1验证结果
        """
        logger.info("=" * 80)
        logger.info("开始执行 Step1: 疾病诊断验证")
        logger.info("=" * 80)

        try:
            step1_results = run_step1_validation(self.config_path)
            self.results["step1"] = step1_results

            logger.info("Step1执行完成")
            return step1_results

        except Exception as e:
            logger.error(f"Step1执行失败: {str(e)}")
            raise

    def run_step2(self) -> List[Dict[str, Any]]:
        """
        执行Step2: 待实现
        """
        logger.info("=" * 80)
        logger.info("Step2: 待实现")
        logger.info("=" * 80)

        # TODO: 实现新的Step2逻辑
        pass

    def run_step3(self) -> List[Dict[str, Any]]:
        """
        执行Step3: 待实现
        """
        logger.info("=" * 80)
        logger.info("Step3: 待实现")
        logger.info("=" * 80)

        # TODO: 实现Step3逻辑
        pass

    def save_results(self, step_name: str, results: List[Dict[str, Any]]):
        """
        保存验证结果到JSON文件

        Args:
            step_name: 步骤名称
            results: 验证结果
        """
        if not results:
            logger.warning(f"{step_name} 结果为空，跳过保存")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存带时间戳的文件
        timestamped_file = os.path.join(
            self.config["output_dir"],
            f"{step_name}_results_{timestamp}.json"
        )

        # 保存固定文件名的文件（最新版本）
        latest_file = os.path.join(
            self.config["output_dir"],
            f"{step_name}_results_latest.json"
        )

        try:
            # 保存带时间戳的版本
            with open(timestamped_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # 保存最新版本
            with open(latest_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            logger.info(f"{step_name} 结果已保存到: {timestamped_file}")
            logger.info(f"{step_name} 最新版本已保存到: {latest_file}")

            # 同时保存一份可读性更好的Excel文件
            self._save_to_excel(step_name, results, timestamp)

        except Exception as e:
            logger.error(f"保存{step_name}结果失败: {str(e)}")

    def _save_to_excel(self, step_name: str, results: List[Dict[str, Any]], timestamp: str):
        """
        将结果保存为Excel格式（便于查看）

        Args:
            step_name: 步骤名称
            results: 验证结果
            timestamp: 时间戳
        """
        try:
            import pandas as pd

            # 展平结果数据以便于在Excel中查看
            flattened_data = []

            for result in results:
                row = {
                    '记录编号': result.get('记录编号', ''),
                    '源文件': result.get('源文件', ''),
                    '姓名': result.get('原始数据', {}).get('姓名', ''),
                    '就诊日期': result.get('原始数据', {}).get('就诊日期', ''),
                    '诊断': result.get('原始数据', {}).get('诊断', ''),
                    '治疗': result.get('原始数据', {}).get('治疗', ''),
                }

                # Step1结果
                if 'step1验证结果' in result:
                    step1_result = result['step1验证结果']
                    row.update({
                        'Step1_是否符合要求': step1_result.get('是否符合要求', ''),
                        'Step1_不符合原因': step1_result.get('不符合原因', ''),
                        '中医疾病': ', '.join(result.get('诊断分类', {}).get('中医疾病', [])),
                        '中医证型': ', '.join(result.get('诊断分类', {}).get('中医证型', [])),
                        '西医诊断': ', '.join(result.get('诊断分类', {}).get('西医诊断', [])),
                        '未匹配项': ', '.join(result.get('诊断分类', {}).get('未匹配项', [])),
                    })

                # 添加其他原始数据字段
                for key, value in result.get('原始数据', {}).items():
                    if key not in ['姓名', '就诊日期', '诊断', '治疗']:
                        row[f'原始_{key}'] = value

                flattened_data.append(row)

            # 保存为Excel
            df = pd.DataFrame(flattened_data)
            excel_file = os.path.join(
                self.config["output_dir"],
                f"{step_name}_results_{timestamp}.xlsx"
            )
            df.to_excel(excel_file, index=False)

            logger.info(f"{step_name} 结果已保存为Excel: {excel_file}")

        except ImportError:
            logger.warning("pandas未安装，跳过Excel格式保存")
        except Exception as e:
            logger.error(f"保存Excel格式失败: {str(e)}")

    def generate_summary_report(self):
        """生成总结报告"""
        logger.info("=" * 80)
        logger.info("生成验证总结报告")
        logger.info("=" * 80)

        summary = {
            "验证时间": datetime.now().isoformat(),
            "配置文件": self.config_path,
            "输入文件": self.config["input_files"],
            "步骤结果": {}
        }

        # Step1统计
        if "step1" in self.results:
            step1_data = self.results["step1"]
            total = len(step1_data)
            passed = len([r for r in step1_data if r["step1验证结果"]["是否符合要求"]])

            summary["步骤结果"]["step1"] = {
                "总记录数": total,
                "通过数": passed,
                "失败数": total - passed,
                "通过率": f"{passed / total * 100:.1f}%" if total > 0 else "0%"
            }

        # Step2统计 - 待实现
        # if "step2" in self.results:
        #     step2_data = self.results["step2"]
        #     total = len(step2_data)
        #     ...
        #     summary["步骤结果"]["step2"] = {...}

        # 保存总结报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(
            self.config["output_dir"],
            f"validation_summary_{timestamp}.json"
        )

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"总结报告已保存: {summary_file}")

        # 打印总结
        for step, stats in summary["步骤结果"].items():
            if step == "step1":
                logger.info(f"{step.upper()} 统计: 总数={stats['总记录数']}, "
                            f"通过={stats['通过数']}, 失败={stats['失败数']}, "
                            f"通过率={stats['通过率']}")
            # elif step == "step2":
            #     logger.info(f"{step.upper()} 统计: 总数={stats['总记录数']}, "
            #                f"有冲突={stats['有冲突数']}, 无冲突={stats['无冲突数']}, "
            #                f"冲突率={stats['冲突率']}")

    def run_all_steps(self):
        """运行所有验证步骤"""
        logger.info("开始执行病历内容验证流水线")

        # 验证配置
        if not self._validate_config():
            logger.error("配置验证失败，终止执行")
            return

        try:
            # Step1: 疾病诊断验证
            step1_results = self.run_step1()
            self.save_results("step1", step1_results)

            # TODO: 添加新的步骤
            # step2_results = self.run_step2()
            # self.save_results("step2", step2_results)

            # step3_results = self.run_step3()
            # self.save_results("step3", step3_results)

            # 生成总结报告
            self.generate_summary_report()

            logger.info("=" * 80)
            logger.info("病历内容验证流水线执行完成")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"验证流水线执行失败: {str(e)}")
            raise


def create_sample_config():
    """创建示例配置文件"""
    sample_config = {
        "input_files": [
            "../../data/case_data/xishi2025.xlsx",
            "../../data/case_data/youyang2025.xlsx"
        ],
        "门诊疾病目录": "../../data/other/disease.json",
        "中医疾病-证型": "../../data/other/disease_to_syndromes_merged.completed.json",
        "output_dir": "../../data/result/",
        "treatment_validation_files": [
            "../../data/case_data/xishi2025.xlsx",
            "../../data/case_data/youyang2025.xlsx"
        ]
    }

    config_file = "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, ensure_ascii=False, indent=2)

    logger.info(f"示例配置文件已创建: {config_file}")
    logger.info("请根据实际情况修改配置文件中的路径")


def main():
    """主函数"""
    config_file = "config.json"

    # 如果配置文件不存在，创建示例配置
    if not os.path.exists(config_file):
        logger.info("配置文件不存在，创建示例配置文件")
        create_sample_config()
        logger.info("请修改配置文件后重新运行程序")
        return

    # 创建并运行验证流水线
    try:
        pipeline = ContentValidationPipeline(config_file)
        pipeline.run_all_steps()
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")


if __name__ == "__main__":
    main()
