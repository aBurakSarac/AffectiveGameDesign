import os
path = r'C:\Users\alibs\miniconda3\envs\facade\Lib\site-packages\hsemotion_onnx\facial_emotions.py'
print('File size:', os.path.getsize(path))
with open(path, 'rb') as f:
    raw = f.read()
null_count = raw.count(b'\x00')
print('Null byte count:', null_count)
print('First 60 bytes (hex):', raw[:60].hex())
print('Encoding BOM check:', repr(raw[:4]))
