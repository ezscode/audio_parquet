
import base64
import pyarrow.parquet as pq

# 读取 parquet 音频数据 和 对应文本 
def read():
    parquet_file_path = '/Users/ez/Documents/data/audio/1/ret/00000.parquet'
    row_idx = 4

    # 打开Parquet文件
    parquet_file = pq.ParquetFile(parquet_file_path)
    
    # 读取整个文件内容
    table = parquet_file.read()
     
    row = table.slice(row_idx, 1)  # 从索引 2 开始，取 1 行
    
    # row.column_names

    text_column_name = 'STT文本' 
    stt_text = row[text_column_name][0].as_py()
    
    audio_column_name = '音频'
    audio_base64_str = row[audio_column_name][0].as_py()
    
    audio_save_path = parquet_file_path + f'_{row_idx}.mp3' 
    audio_bytes = base64.b64decode(audio_base64_str.encode('utf-8'))
    print('-- audio_bytes : ',len(audio_bytes)) 
    with open(audio_save_path, 'wb') as f:f.write(audio_bytes) 
    
    print('-- ', row_idx, stt_text, audio_save_path) 
    

    
read()

