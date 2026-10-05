# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 16:07:40 2024

@author: N7
"""
#%% check 資料夾
import os

os.makedirs("execution_time", exist_ok=True)
os.makedirs("merged", exist_ok=True)
#%% 前置作業
import pandas as pd
import re
import time
import gc
# from concurrent.futures import ThreadPoolExecutor

execution_time = {} #timer

table = pd.read_csv("highway_rain.csv", header=None)
table[0] = table[0].str.strip()
pattern = re.compile('|'.join(table[0].astype(str)))
#%% read_hightway
def read_highway(yyyy, mm, dd, hh, minute):
    try:
        file_path = f"highway\\{yyyy}{mm}{dd}\\M05A\\{yyyy}{mm}{dd}\\{hh}\\TDCS_M05A_{yyyy}{mm}{dd}_{hh}{minute}00.csv"
        og_df = pd.read_csv(file_path, header=None)
    except FileNotFoundError:
        print(yyyy, mm, dd, hh, minute)
        return None
    og_df.columns = ['年月', '路段起始', '路段結束', '車種', '平均車速', '通過數量']
    
    filtered_df = og_df[
        (og_df['路段起始'].str.contains(pattern, na=False)) &
        (og_df['路段結束'].str.contains(pattern, na=False)) &
        (og_df['通過數量'] != 0) &
        (~og_df['車種'].isin([5])) #32?
    ].copy()
    
    if filtered_df.empty:
        return None
    
    loc = filtered_df['路段起始'].str[:-1]
    rain_values = read_rain(yyyy, mm, dd, hh, loc)
    
    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df['降雨量'] = rain_values.values
    
    filtered_df['路段'] = filtered_df['路段起始'] + '_' + filtered_df['路段結束']
    filtered_df['年'] = yyyy
    filtered_df['月'] = mm
    filtered_df['日'] = dd
    filtered_df['時'] = hh
    filtered_df['分'] = minute
    
    filtered_df = filtered_df[['年', '月', '日', '時', '分', '路段', '車種', '平均車速', '通過數量', '降雨量']]
    
    return(filtered_df)
#%% read_rain 
def read_rain(yyyy, mm, dd, hh, loc):
    rain_values = []
    
    for location in loc:
        station = table[table[0] == location].iloc[0, 1] if not table[table[0] == location].empty else None
        if station is None:
            rain_values.append(0)
            continue
        
        file_path = f"rain\\{station}\\{station}-{yyyy}-{mm}-Precipitation-hour.csv"

        try:
            rain_df = pd.read_csv(file_path)
            
            # 移除無關欄位（如總和）
            rain_df = rain_df.drop(columns=['總和'])
            rain_df = rain_df[:-1]
            
            # 重塑數據，將行列轉換為標準格式
            data_melted = rain_df.melt(id_vars='日/時', var_name='小時', value_name='降雨量')
            data_melted.rename(columns={'日/時': '日期'}, inplace=True)
            
            # 處理無效數據
            data_melted['降雨量'] = pd.to_numeric(data_melted['降雨量'], errors='coerce').fillna(0)
            
            # 將日期與小時轉換為數值型態
            data_melted['日期'] = data_melted['日期'].astype(int)
            data_melted['小時'] = data_melted['小時'].astype(int)
            data_melted['降雨量'] = data_melted['降雨量'].astype(float)
            
            # 排序數據
            data_melted.sort_values(by=['日期', '小時'], inplace=True)
            data_melted = data_melted[data_melted['降雨量'] != 0].copy()

            # data_melted.to_csv('test_rain.csv', index=False, encoding="cp950")
            
            # 找到符合時間的降雨量
            rain_value = data_melted[
                (data_melted['日期'] == int(dd)) &
                (data_melted['小時'] == int(hh)+1)
            ]['降雨量']
            
            # 若找到值則取第一個，否則為 0
            rain_values.append(rain_value.iloc[0] if not rain_value.empty else 0)
        except FileNotFoundError:
            rain_values.append(0)
        
    # for i in pd.Series(rain_values):
    #     if i != 0:
    #         print(i)
        
    return pd.Series(rain_values)
#%% merge_highway(all)
def merge_highway(yyyy, mm, interval="30"):
    data = [] 
    for dd in range(1, 32): #1, 32
        start_time = time.time()
        for hh in range(6,22): #24 maybe 6,22(6~21)
            for minute in range(0,60,int(interval)):
                child_df = read_highway(yyyy, mm.zfill(2), str(dd).zfill(2), str(hh).zfill(2), str(minute).zfill(2))
                print(child_df)
                if child_df is not None:
                    data.append(child_df)  
        
        elapsed_time = time.time() - start_time  # 每天的執行時間
        dd_key = f"{yyyy}-{mm}-{dd:02d}"
        execution_time[dd_key] = elapsed_time
        print(f"Date: {dd_key}, Execution Time: {elapsed_time:.2f} seconds")
    
    # with ThreadPoolExecutor() as executor:
    #     tasks = [
    #         executor.submit(
    #             read_highway,
    #             str(year), str(mm).zfill(2), str(dd).zfill(2), str(hh).zfill(2), str(minute).zfill(2)
    #         )
    #         for year in range(int(syyyy), int(eyyyy) + 1)
    #         for mm in range(1, 13)  
    #         for dd in range(1, 32)  
    #         for hh in range(24)
    #         for minute in range(0,60,int(interval))
    #     ]
        
    #     for task in tasks:
    #         result = task.result()
    #         if result is not None:
    #             data.append(result)
    
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()
#%% save timer
def save_execution_time(d_time, yyyy, mm):
    total_time = d_time['執行時間'].sum()
    d_time.loc['總時間'] = total_time
    print("Total Execution Time: ", total_time)
    d_time.to_csv(f"execution_time\\{yyyy}_{mm}_execution_time.csv", encoding="cp950")
#%% main
for yyyy in range(2022,2023):
    for mm in range(12,11,-1): 
        data = merge_highway(str(yyyy),str(mm))
        
        d_time = pd.DataFrame.from_dict(execution_time, orient='index', columns=['執行時間'])
        d_time.index.name = '日期' 
        save_execution_time(d_time, yyyy, mm)
        
        data.to_csv(f"merged\\{yyyy}_{mm}_merged.csv", index = False, encoding = "cp950")
        
        # 清空執行時間紀錄，防止影響下一個月
        execution_time.clear()
        del data
        gc.collect()  # 強制執行垃圾回收

