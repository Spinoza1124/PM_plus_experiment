#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据emotion_labels.xlsx文件，从emotion_annotation文件夹复制对应的数据
"""

import pandas as pd
import os
import shutil
from pathlib import Path
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('copy_emotion_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def read_emotion_labels(excel_file_path):
    """
    读取emotion_labels.xlsx文件
    
    Args:
        excel_file_path (str): Excel文件路径
        
    Returns:
        pandas.DataFrame: 包含情感标签数据的DataFrame
    """
    try:
        logger.info(f"正在读取Excel文件: {excel_file_path}")
        df = pd.read_excel(excel_file_path)
        logger.info(f"成功读取Excel文件，共{len(df)}行，{len(df.columns)}列")
        logger.info(f"列名: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"读取Excel文件失败: {e}")
        return None

def find_matching_folders(df, source_dir):
    """
    根据DataFrame中的数据找到匹配的文件夹
    
    Args:
        df (pandas.DataFrame): 情感标签数据
        source_dir (str): 源数据目录
        
    Returns:
        list: 匹配的文件夹路径列表
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        logger.error(f"源目录不存在: {source_dir}")
        return []
    
    # 获取所有可用的文件夹
    available_folders = [f.name for f in source_path.iterdir() if f.is_dir()]
    logger.info(f"源目录中共有{len(available_folders)}个文件夹")
    
    matching_folders = []
    
    # 假设Excel文件中有一列包含文件名或ID信息
    # 需要根据实际的Excel文件结构调整这部分逻辑
    for column in df.columns:
        logger.info(f"检查列: {column}")
        if df[column].dtype == 'object':  # 字符串类型的列
            for value in df[column].dropna().unique():
                value_str = str(value)
                # 尝试匹配文件夹名
                for folder in available_folders:
                    if value_str in folder or folder in value_str:
                        folder_path = source_path / folder
                        if folder_path not in [Path(f) for f in matching_folders]:
                            matching_folders.append(str(folder_path))
                            logger.info(f"找到匹配文件夹: {folder} (匹配值: {value_str})")
    
    # 如果没有找到匹配的文件夹，尝试直接匹配spk开头的文件夹
    if not matching_folders:
        logger.info("尝试匹配所有spk开头的文件夹...")
        for folder in available_folders:
            if folder.startswith('spk'):
                folder_path = source_path / folder
                matching_folders.append(str(folder_path))
                logger.info(f"添加spk文件夹: {folder}")
    
    logger.info(f"总共找到{len(matching_folders)}个匹配的文件夹")
    return matching_folders

def copy_data(matching_folders, target_dir):
    """
    复制匹配的文件夹到目标目录
    
    Args:
        matching_folders (list): 要复制的文件夹路径列表
        target_dir (str): 目标目录
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 创建一个子目录来存放复制的数据
    emotion_data_dir = target_path / "emotion_annotation"
    emotion_data_dir.mkdir(exist_ok=True)
    
    copied_count = 0
    failed_count = 0
    
    for folder_path in matching_folders:
        source_folder = Path(folder_path)
        if not source_folder.exists():
            logger.warning(f"源文件夹不存在: {folder_path}")
            failed_count += 1
            continue
            
        target_folder = emotion_data_dir / source_folder.name
        
        try:
            if target_folder.exists():
                logger.info(f"目标文件夹已存在，跳过: {target_folder}")
                continue
                
            shutil.copytree(source_folder, target_folder)
            logger.info(f"成功复制: {source_folder.name} -> {target_folder}")
            copied_count += 1
            
        except Exception as e:
            logger.error(f"复制文件夹失败 {source_folder.name}: {e}")
            failed_count += 1
    
    logger.info(f"复制完成: 成功{copied_count}个，失败{failed_count}个")
    return copied_count, failed_count

def main():
    """
    主函数
    """
    # 文件路径配置
    excel_file = "/mnt/shareEEx/liuyang/code/PM_plus_experiment/data/PM/emotion_labels.xlsx"
    source_dir = "/mnt/shareEEx/liuyang/code/emotion_labeling_refactoring/data/emotion_annotation_copy"
    target_dir = "/mnt/shareEEx/liuyang/code/PM_plus_experiment/data"
    
    logger.info("开始处理情感数据复制任务")
    logger.info(f"Excel文件: {excel_file}")
    logger.info(f"源目录: {source_dir}")
    logger.info(f"目标目录: {target_dir}")
    
    # 检查文件和目录是否存在
    if not os.path.exists(excel_file):
        logger.error(f"Excel文件不存在: {excel_file}")
        return
    
    if not os.path.exists(source_dir):
        logger.error(f"源目录不存在: {source_dir}")
        return
    
    # 读取Excel文件
    df = read_emotion_labels(excel_file)
    if df is None:
        logger.error("无法读取Excel文件，程序退出")
        return
    
    # 显示Excel文件的基本信息
    logger.info("\nExcel文件内容预览:")
    logger.info(f"前5行数据:\n{df.head()}")
    
    # 查找匹配的文件夹
    matching_folders = find_matching_folders(df, source_dir)
    
    if not matching_folders:
        logger.warning("没有找到匹配的文件夹")
        return
    
    # 复制数据
    copied_count, failed_count = copy_data(matching_folders, target_dir)
    
    logger.info(f"\n任务完成!")
    logger.info(f"总共处理: {len(matching_folders)}个文件夹")
    logger.info(f"成功复制: {copied_count}个")
    logger.info(f"复制失败: {failed_count}个")
    
    # 生成报告文件
    report_file = f"{target_dir}/copy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"情感数据复制报告\n")
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"Excel文件: {excel_file}\n")
        f.write(f"源目录: {source_dir}\n")
        f.write(f"目标目录: {target_dir}\n")
        f.write(f"总共处理: {len(matching_folders)}个文件夹\n")
        f.write(f"成功复制: {copied_count}个\n")
        f.write(f"复制失败: {failed_count}个\n")
        f.write(f"\n匹配的文件夹列表:\n")
        for folder in matching_folders:
            f.write(f"  {folder}\n")
    
    logger.info(f"报告文件已生成: {report_file}")

if __name__ == "__main__":
    main()