import pandas as pd
import json
import os
from pathlib import Path
import glob


def extract_diagnosis_from_xlsx_files(xlsx_file_paths):
    """
    从指定的多个xlsx文件中提取诊断列的数据

    Args:
        xlsx_file_paths: xlsx文件路径列表

    Returns:
        set: 去重后的疾病名称集合
    """
    all_diseases = set()

    if not xlsx_file_paths:
        print("没有指定任何xlsx文件")
        return all_diseases

    print(f"将处理 {len(xlsx_file_paths)} 个xlsx文件:")
    for file_path in xlsx_file_paths:
        print(f"  - {file_path}")

    # 检查文件是否存在
    existing_files = []
    for file_path in xlsx_file_paths:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            print(f"  警告: 文件不存在 - {file_path}")

    if not existing_files:
        print("没有找到任何有效的xlsx文件")
        return all_diseases

    # 处理每个xlsx文件
    for xlsx_file in existing_files:
        try:
            print(f"\n正在处理文件: {os.path.basename(xlsx_file)}")

            # 读取Excel文件
            df = pd.read_excel(xlsx_file)

            # 检查是否存在"诊断"列
            if '诊断' not in df.columns:
                print(f"  警告: 文件 {os.path.basename(xlsx_file)} 中没有找到'诊断'列")
                print(f"  可用列名: {list(df.columns)}")
                continue

            # 提取诊断列数据
            diagnosis_series = df['诊断'].dropna()  # 去除空值

            # 处理每一行的诊断数据
            for diagnosis in diagnosis_series:
                if pd.isna(diagnosis) or diagnosis == "":
                    continue

                # 将诊断按逗号分隔，并去除空白字符
                diseases_in_row = [disease.strip() for disease in str(diagnosis).split(',') if disease.strip()]
                all_diseases.update(diseases_in_row)

            print(f"  从文件中提取了 {len(diagnosis_series)} 条诊断记录")

        except Exception as e:
            print(f"  错误: 处理文件 {os.path.basename(xlsx_file)} 时出现异常: {str(e)}")
            continue

    print(f"\n总共提取到 {len(all_diseases)} 个不重复的疾病名称")
    return all_diseases


def load_medical_terms_json(json_file_path):
    """
    加载医疗术语JSON文件，并提取"中医疾病"类型的项目

    Args:
        json_file_path: JSON文件路径

    Returns:
        set: 中医疾病名称集合
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取type为"中医疾病"的项目
        tcm_diseases = set()
        if 'items' in data:
            for item in data['items']:
                if item.get('type') == '中医疾病':
                    tcm_diseases.add(item.get('name', ''))

        print(f"从JSON文件中加载了 {len(tcm_diseases)} 个中医疾病")
        return tcm_diseases

    except FileNotFoundError:
        print(f"错误: 找不到JSON文件 {json_file_path}")
        return set()
    except json.JSONDecodeError:
        print(f"错误: JSON文件 {json_file_path} 格式不正确")
        return set()
    except Exception as e:
        print(f"错误: 加载JSON文件时出现异常: {str(e)}")
        return set()


def validate_diseases(extracted_diseases, valid_tcm_diseases):
    """
    验证提取的疾病是否在有效的中医疾病列表中

    Args:
        extracted_diseases: 从xlsx文件中提取的疾病集合
        valid_tcm_diseases: 有效的中医疾病集合

    Returns:
        tuple: (验证通过的疾病集合, 未通过验证的疾病集合)
    """
    validated_diseases = extracted_diseases.intersection(valid_tcm_diseases)
    invalid_diseases = extracted_diseases - valid_tcm_diseases

    print(f"\n验证结果:")
    print(f"  验证通过的疾病数量: {len(validated_diseases)}")
    print(f"  未通过验证的疾病数量: {len(invalid_diseases)}")

    if invalid_diseases:
        print(f"  未通过验证的疾病前10个示例:")
        for i, disease in enumerate(list(invalid_diseases)[:10]):
            print(f"    - {disease}")
        if len(invalid_diseases) > 10:
            print(f"    ... 还有 {len(invalid_diseases) - 10} 个")

    return validated_diseases, invalid_diseases


def save_diseases_to_txt(diseases, output_file_path):
    """
    将疾病集合保存到txt文件中，每行一个疾病名称

    Args:
        diseases: 疾病名称集合
        output_file_path: 输出文件路径
    """
    try:
        # 对疾病名称进行排序以便查看
        sorted_diseases = sorted(diseases)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            for disease in sorted_diseases:
                f.write(f"{disease}\n")

        print(f"\n已将 {len(diseases)} 个验证通过的中医疾病保存到文件: {output_file_path}")

    except Exception as e:
        print(f"错误: 保存文件时出现异常: {str(e)}")


def main():
    """
    主函数，执行完整的处理流程
    """
    print("=== 医疗诊断数据处理程序 ===\n")

    # 配置文件路径 - 在这里指定你要处理的xlsx文件
    xlsx_file_paths = [
        "../../data/case_data/xishi2025.xlsx",
        "../../data/case_data/youyang2025.xlsx"
        # "其他文件1.xlsx",  # 示例文件2，取消注释并修改为实际路径
        # "其他文件2.xlsx",  # 示例文件3，取消注释并修改为实际路径
        # 可以继续添加更多文件...
    ]

    json_file_path = "../../data/other/merged_medical_terms.json"  # JSON文件路径
    output_file_path = "../../data/other/dvalidated_tcm_diseases.txt"  # 输出文件路径

    # 步骤1: 从xlsx文件中提取诊断数据
    print("步骤1: 从xlsx文件中提取诊断数据")
    extracted_diseases = extract_diagnosis_from_xlsx_files(xlsx_file_paths)

    if not extracted_diseases:
        print("没有提取到任何疾病数据，程序结束")
        return

    # 步骤2: 加载医疗术语JSON文件
    print("\n步骤2: 加载医疗术语JSON文件")
    valid_tcm_diseases = load_medical_terms_json(json_file_path)

    if not valid_tcm_diseases:
        print("没有加载到有效的中医疾病数据，程序结束")
        return

    # 步骤3: 验证疾病
    print("\n步骤3: 验证疾病数据")
    validated_diseases, invalid_diseases = validate_diseases(extracted_diseases, valid_tcm_diseases)

    # 步骤4: 保存结果
    print("\n步骤4: 保存验证结果")
    save_diseases_to_txt(validated_diseases, output_file_path)

    print(f"\n=== 处理完成 ===")
    print(f"原始提取的疾病数量: {len(extracted_diseases)}")
    print(f"验证通过的疾病数量: {len(validated_diseases)}")
    print(f"结果已保存到: {output_file_path}")


if __name__ == "__main__":
    main()
