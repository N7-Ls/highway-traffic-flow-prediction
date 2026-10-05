# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 09:17:46 2025

@author: User
"""

import pandas as pd

# 加載已合併的月數據
monthly_data_path = "merged\\2024_1_merged.csv"  # 假設是2024年1月的數據
monthly_data = pd.read_csv(monthly_data_path)


# 加載漏掉的一天數據
missing_day_data_path = "merged\\missing_day.csv"  # 漏掉的數據文件
missing_day_data = pd.read_csv(missing_day_data_path)
missing_day_data = missing_day_data[monthly_data.columns]

# 合併數據
updated_data = pd.concat([monthly_data, missing_day_data], ignore_index=True)
updated_data = updated_data.drop_duplicates()

# 覆蓋保存原文件
updated_data.to_csv("merged\\2024_1_merged.csv", index=False, encoding="cp950")
