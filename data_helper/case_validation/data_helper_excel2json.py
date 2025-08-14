# -*- coding: utf-8 -*-
"""
excel2json_build.py
功能：读取三个 Excel 指定字段，清洗（去空白、全角转半角、保序去重），合并为一个 JSON。

读取目录（相对本文件）：
  ../../data/other/
    - 中医症型.xlsx              （取列：证候分类名称）
    - 中医疾病目录最新.xlsx      （取列：疾病分类名称）
    - 西医诊断.xlsx              （取列：诊断名称）

输出文件：
  ../../data/other/merged_medical_terms.json
"""

import os
import json
import re

try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("需要 pandas，请先安装：pip install pandas") from e


# =============== 路径与字段配置（按需可改） ===============
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
READ_DIR = os.path.normpath(os.path.join(ROOT_DIR, "../../data/other"))
OUT_DIR  = os.path.normpath(os.path.join(ROOT_DIR, "../../data/other"))
OUT_JSON = os.path.join(OUT_DIR, "merged_medical_terms.json")

SOURCES = [
    # (文件名, 列名, 标注类型)
    ("中医症型.xlsx",         "证候分类名称", "证候"),
    ("中医疾病目录最新.xlsx", "疾病分类名称", "中医疾病"),
    ("西医诊断.xlsx",         "诊断名称",     "西医诊断"),
]
VERSION = "2025-08-14"
# =======================================================


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def to_halfwidth(s: str) -> str:
    """全角转半角（含空格）"""
    res = []
    for ch in s:
        code = ord(ch)
        if code == 12288:  # 全角空格
            code = 32
        elif 65281 <= code <= 65374:  # 全角字符
            code -= 65248
        res.append(chr(code))
    return "".join(res)


def normalize(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    s = s.replace("\u3000", " ")        # 中文空格
    s = to_halfwidth(s)                 # 全角转半角
    s = re.sub(r"\s+", " ", s)          # 压缩多空格
    return s


def dedupe_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def read_one(file_name: str, col_name: str, typ: str):
    fpath = os.path.join(READ_DIR, file_name)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"未找到文件：{fpath}")
    df = pd.read_excel(fpath, dtype=str)
    if col_name not in df.columns:
        raise ValueError(f"{file_name} 未找到列：{col_name}，实际列：{list(df.columns)}")

    vals = []
    for v in df[col_name].tolist():
        if isinstance(v, str):
            nv = normalize(v)
            if nv:
                vals.append(nv)

    vals = dedupe_preserve_order(vals)  # 单表内保序去重
    return [{"name": v, "type": typ} for v in vals]


def main():
    ensure_dir(OUT_DIR)

    all_items = []
    meta_sources = {}

    for file_name, col_name, typ in SOURCES:
        print(f"读取：{file_name}（列：{col_name}，类型：{typ}）")
        part = read_one(file_name, col_name, typ)
        all_items.extend(part)
        # 记录元数据（方便追溯）
        if typ == "证候":
            meta_sources["证候表"] = file_name
        elif typ == "中医疾病":
            meta_sources["中医疾病目录"] = file_name
        elif typ == "西医诊断":
            meta_sources["西医诊断"] = file_name

    # 跨表再做一次按 (name, type) 保序去重
    seen = set()
    merged = []
    for it in all_items:
        key = (it["name"], it["type"])
        if key not in seen:
            seen.add(key)
            merged.append(it)

    payload = {
        "version": VERSION,
        "source_files": meta_sources,
        "items": merged
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ 完成：{OUT_JSON}")
    print(f"统计：证候/中医疾病/西医诊断 共 {len(merged)} 条。")


if __name__ == "__main__":
    main()
