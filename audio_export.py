#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   audio_export.py
@Time    :   2024-12-07 14:20:32
@Author  :   Ez
@Version :   1.0
@Desc    :   整理分块文件，处理为统一格式，并根据文件尺寸大小限制拼接 

'''


import os 
import io
import json 
import base64
import hashlib
from datetime import datetime

from pydub  import AudioSegment 

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB 
# MAX_FILE_SIZE = 500 * 1024 # test 


# 对于 每个音频 的 每个 block，生成 指定格式 数据  
def gen_block_data(sub_audio_path, sub_stt_path, audio_start_time, start_block_id, blocks_dir, audio_type): 
  
    stt_ret = json.loads(open(sub_stt_path).read())
    current_time = datetime.now().strftime("%Y%m%d") 
    chunks = stt_ret['chunks']
    
    sound = AudioSegment.from_file(sub_audio_path)  
    audio_duration = sound.duration_seconds  * 1000
    print('-- audio_duration 1 : ', audio_duration) 
    sampling_rate = sound.frame_rate                 
     
    for chunk in chunks: 
        stt = chunk['text']  
        start_time, end_time = chunk['timestamp']
        if start_time == None or end_time == None:  # 异常数据 
            print('xx err msg ') 
            continue
        
        # 将时间转换为毫秒
        start_ms = int(float(start_time) * 1000)
        end_ms = int(float(end_time) * 1000)
        
        # 切分音频
        audio_segment = sound[start_ms:end_ms]
        audio_byte_arr = io.BytesIO()
        audio_segment.export(audio_byte_arr, format=audio_type)
        audio_bytes = audio_byte_arr.getvalue() 
        
        print('-- audio_bytes : ', len(audio_bytes) )  
        a_path1 = os.path.join(blocks_dir, f'{start_block_id:05d}.mp3')  
        with open(a_path1, 'wb') as f:f.write(audio_bytes)  
        
        a_path2 = os.path.join(blocks_dir, f'{start_block_id:05d}.txt')  
        audio_str = base64.b64encode(audio_bytes).decode('utf-8')   
        with open(a_path2, 'w') as f:f.write(audio_str)  
         
        # stime  = audio_start_time + float(start_time) * 1000
        extended_fields = {  
            'sampling_rate' : sampling_rate, # 采样率  
            'duration': len(audio_segment),  # 音频段长度（毫秒）
            # 'start_time': stime, 
        }
        
        readable_hash = hashlib.md5(audio_bytes).hexdigest() 

        dict = {
                '实体ID': os.path.basename(sub_audio_path),  # 子音频文件名
                '块ID': start_block_id,  # 0  
                '时间': current_time,  # 数据首次出现时间
                '扩展字段': json.dumps(extended_fields), # ？
                '文本': '', # 音频文件 提取的文本数据，存储在 STT文本 
                '图片': '', # 音频文件没有 图片 数据 
                'OCR文本': '',  
                '音频': audio_str,    # 音频块的内容
                'STT文本': stt, # 子音频 文本
                '其它块': None,   
                '块类型': '音频',  
                'md5': readable_hash,  
                '页ID': ''  
            }
  
        save_path = os.path.join(blocks_dir, f'{start_block_id:05d}.json') 
        with open(save_path, 'a') as fa:fa.write(json.dumps(dict, ensure_ascii=False) + '\n') 
        start_block_id += 1 

    audio_start_time += audio_duration 
    print(audio_start_time, start_block_id)
    return audio_start_time, start_block_id

# 拼接数据 
def concact_block_files(src_dir, save_dir):
    if not os.path.isdir(src_dir):return 
    if not os.path.isdir(save_dir):os.makedirs(save_dir) 
 
    file_id = 0
    files = os.listdir(src_dir)
    files.sort() 

    for file_name in files:
        if not file_name.endswith('.json'):continue
        file_path = os.path.join(src_dir, file_name) 
    
        c_size = os.stat(file_path).st_size 

        save_path = os.path.join(save_dir, f'{file_id:05d}.jsonl') 

        exist_size = 0
        if os.path.isfile(save_path):
            exist_size = os.stat(save_path).st_size 
        
        if c_size + exist_size > MAX_FILE_SIZE:
            print('-- file_id : ', file_id, exist_size) 
            file_id += 1
            save_path = os.path.join(save_dir, f'{file_id:05d}.jsonl') 
        
        content = open(file_path).read().strip()  
        with open(save_path, 'a') as f:
            f.write(content + '\n') 

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as parquet

def json2parquet(jsonl_path, parquet_path):
    
    if not os.path.isfile(jsonl_path):
        print('xx 文件不存在：', jsonl_path) 
        return
    
    doct_arr = []
    for line in open(jsonl_path):
        dict = json.loads(line.strip())
        doct_arr.append(dict)
    
    df = pd.DataFrame(doct_arr)   
    table = pa.Table.from_pandas(df) 
    parquet.write_table(table, parquet_path)

 

if __name__ == '__main__':
    
    # test_base642audio()

    jsonl_path = '/Users/ez/Documents/data/audio/1-1/ret/00001.jsonl' 
    parquet_path = jsonl_path + '.parquet' 
    json2parquet(jsonl_path, parquet_path)