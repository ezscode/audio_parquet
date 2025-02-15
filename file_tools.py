from enum import StrEnum 
from pathlib import Path  


# 文件扩展名 和 所属分类  
EXT_AUDIO = ['mp3', 'mov', 'wav', 'ogg', 'flac', 'm4a']  
EXT_VIDEO = ['mp4', 'mepg']  

# 文件类型 - 文件内容信息 
class FileType(StrEnum): 
    audio = 'audio'  
    video = 'video'   
    none  = 'none'

class FileObj(object):

    def __init__(self) -> None:
        self.file_type = FileType.none 
        self.file_path = '' 
        self.file_name = ''
        self.file_id = '' 
        self.file_extns = '' # 文件名后缀
        self.work_dir = ''

        self.source = '' 
        self.url = ''
        self.title = ''
        self.content = ''
        self.addinfo = {} # 额外信息 

def get_file_info(file_path): 
    
    fileObj = FileObj() 
    fileObj.file_path = file_path


    p = Path(file_path)
    fileObj.file_extns = p.suffix[1:] 
    fileObj.file_name = p.name

    fileObj.work_dir = file_path.replace(f'.{fileObj.file_extns}', '') 

    if fileObj.file_extns in EXT_AUDIO:
        fileObj.file_type = FileType.audio
 
    return fileObj
