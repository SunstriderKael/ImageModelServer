import os


def get_txt_list(txt_folder):
    txt_list = list()
    for root, dirs, files in os.walk(txt_folder):
        for file in files:
            ext = os.path.splitext(file)[-1]
            if ext == '.txt':
                txt_list.append(file)
    return txt_list


def list2txt(line_list, txt_path, encoding='utf-8'):
    if os.path.exists(txt_path):
        os.remove(txt_path)

    txt_file = open(txt_path, 'a', encoding=encoding)
    for line in line_list:
        line += '\n'
        txt_file.write(line)
    txt_file.close()


def txt2list(txt_path, encoding='utf-8'):
    txt_file = open(txt_path, 'r', encoding=encoding)
    output_list = list()
    for line in txt_file.readlines():
        line = line.strip()
        line = line.replace('\n', '')
        output_list.append(line)
    return output_list
