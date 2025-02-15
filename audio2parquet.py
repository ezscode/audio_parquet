#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   parse_audio.py
@Time    :   2024-12-07 14:14:17
@Author  :   Ez
@Version :   1.0
@Desc    :   音频处理
 
'''

import os
import sys 
import time 
import json  
import shutil 

from pydub  import AudioSegment
from pydub.silence import split_on_silence 
 
import jax.numpy as jnp
from whisper_jax import FlaxWhisperPipline

from audio_export import gen_block_data, concact_block_files, json2parquet 

from file_tools import get_file_info

stt_model_name = "openai/whisper-large-v3"
stt_pipeline = FlaxWhisperPipline(stt_model_name, dtype=jnp.bfloat16, batch_size=16)  


# 静音切分后的数据 --> 识别
def stt_chunks(audio_obj, chunks_dir, blocks_dir): 
    
    chunk_files = os.listdir(chunks_dir)  
    chunk_files.sort()   
    print('-- chunk_files : ', len(chunk_files) )


    # 静音切分后的数据，使用 whisper-jax 提取 timestamp、text   
    for sub_file_name in chunk_files:
        if not sub_file_name.endswith(audio_obj.file_extns):continue
        
        sub_audio_file_path = os.path.join(chunks_dir, sub_file_name)   

        # 语音转文本  
        stt_ret = stt_pipeline(sub_audio_file_path, task="transcribe", return_timestamps=True)
        print('-- stt_ret :', stt_ret)   
        
        stt_path = os.path.join(chunks_dir, sub_file_name.replace(f'.{audio_obj.file_extns}', '.json'))
        with open(stt_path, 'w') as f:f.write(json.dumps(stt_ret, ensure_ascii=False )) 
    
    # tts 后，生成统一格式数据
    chunk_files = os.listdir(chunks_dir)
    chunk_files.sort()  

    audio_start_time = 0   # 本段开始时间  
    start_block_id = 0 # 本段开始 id 

    for sub_file_name in chunk_files:
        if not sub_file_name.endswith('.json'):continue
        sub_stt_path = os.path.join(chunks_dir, sub_file_name)
        sub_audio_path = sub_stt_path.replace('.json', f'.{audio_obj.file_extns}') 
        
        audio_start_time, start_block_id = gen_block_data(sub_audio_path, sub_stt_path, audio_start_time, start_block_id, blocks_dir, audio_obj.file_extns)    


    
def split_audio_by_silence(fileObj, chunks_dir='', min_silence_len=1000, silence_thresh=-70):
    """
    min_silence_len: 拆分语句时，静默满0.3秒则拆分
    silence_thresh：小于-70dBFS以下的为静默
    """
    sound = AudioSegment.from_file(fileObj.file_path, format=fileObj.file_extns)
    # 分割 
    print('\n---- start split by silence', fileObj.file_path)
    chunks = split_on_silence(sound, min_silence_len=min_silence_len, silence_thresh=silence_thresh) 
    print('-- chunks : ', len(chunks)) 
    
    if len(chunks_dir) > 0 and len(chunks) > 0: 
        # 保存所有分段 
        for i, chunk in enumerate(chunks):
            save_path = os.path.join(chunks_dir, f'{i:04d}.{fileObj.file_extns}') 
            print(f'-- {i:04d}',  len(chunk), save_path )
            # print('-- ', save_path)
            chunk.export(save_path, format=fileObj.file_extns)

    print('== end split by silence ', len(chunks))


def prcs_audio(file_path):
    audio_obj = get_file_info(file_path) 
        
    # 文件夹 - 存储 根据静音 切分的数据块 
    chunks_dir = os.path.join(audio_obj.work_dir, 'chunks') 
    if not os.path.exists(chunks_dir):os.makedirs(chunks_dir) 
    # 文件夹 - 存储根据 whisper-jax 分块后的数据 
    blocks_dir = os.path.join(audio_obj.work_dir , 'blocks' ) 
    if not os.path.isdir(blocks_dir):os.makedirs(blocks_dir) 

    # 静音分割文件 
    split_audio_by_silence(audio_obj, chunks_dir=chunks_dir, min_silence_len=1000, silence_thresh=-70) 
    # stt 识别音频对应文本
    stt_chunks(audio_obj, chunks_dir, blocks_dir)          
    
    # 存储 block 按照大小限制 拼接的数据 
    ret_dir = os.path.join(audio_obj.work_dir , 'ret' )  
    concact_block_files(blocks_dir, ret_dir)
    
    for file_name in os.listdir(ret_dir):
        if not file_name.endswith('.jsonl'):continue
        
        jsonl_path = os.path.join(ret_dir, file_name)
        parquet_path = jsonl_path.replace('.jsonl', '.parquet') 
        json2parquet(jsonl_path, parquet_path)
        os.remove(jsonl_path) 

    # 删除中间文件 
    shutil.rmtree(chunks_dir)
    shutil.rmtree(blocks_dir)


def handle_paths(paths):
    for path in paths: 
        if os.path.isfile(path):
            print('-- ', path) 
            prcs_audio(path)

        if os.path.isdir(path): 
            for file_name in os.listdir(path):
                file_path = os.path.join(path, file_name) 
                prcs_audio(file_path) 
 

if __name__ == '__main__':
    
    paths = sys.argv[1:] 
    print('-- ', paths) 
    handle_paths(paths) 
    




 