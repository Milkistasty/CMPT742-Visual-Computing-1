import os
import json

# 图片文件夹和JSON文件的路径
image_dir = "C:\\Users\\Alienware\\Desktop\\github\\MPOSE2021\\normalmap"
json_file = "C:\\Users\\Alienware\\Desktop\\github\\MPOSE2021\\prompt.json"

# 读取所有图片文件名
image_files = set(os.listdir(image_dir))

# 读取并解析JSON文件
with open(json_file, 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

# 过滤掉不存在的图片的数据
filtered_data = [item for item in data if os.path.basename(item['source'].replace("source/", "")) in image_files]

# 保存更新后的JSON数据
with open(json_file, 'w', encoding='utf-8') as file:
    for item in filtered_data:
        item['source'] = "source/" + item['source'].replace("source/", "")
        item['target'] = "target/" + item['target'].replace("target/", "")
        file.write(json.dumps(item) + '\n')
