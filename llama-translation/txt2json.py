import json

# 读取两个文本文件的内容
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]

# 主函数
def create_json(x_file_path, y_file_path, output_file_path):
    x_lines = read_file(x_file_path)
    y_lines = read_file(y_file_path)

    # 确保两个文件中的行数相同
    assert len(x_lines) == len(y_lines), "Files have different numbers of lines"

    # 构建JSON结构
    json_data = []
    for x_line, y_line in zip(x_lines, y_lines):
        json_data.append({
            "instruction": "translate the input",
            "input": x_line,
            "output": y_line
        })

    # 写入JSON文件
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=2, ensure_ascii=False)

    print(f"JSON file '{output_file_path}' has been created successfully.")

# 用具体的文件路径调用函数
create_json('./data/x.txt', './data/y.txt', 'train_data.json')
