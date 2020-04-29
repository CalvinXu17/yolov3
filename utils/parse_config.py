import os

import numpy as np


"""
返回的mdefs -> 所有的网络参数信息
一个list中嵌套了多个dict，每一个dict对应的是网络中的一个子模块——卷积、池化、特征拼接、跨层连接或者yolo输出层
"""
# 处理模型参数文件cfg/yolov3.cfg等文件
def parse_model_cfg(path):
    # Parse the yolo *.cfg file and return module definitions path may be 'cfg/yolov3.cfg', 'yolov3.cfg', or 'yolov3'
    if not path.endswith('.cfg'):  # 若遗漏扩展名则添加.cfg
        path += '.cfg'
    if not os.path.exists(path) and os.path.exists('cfg' + os.sep + path):  # 如果遗漏则在开头添加cfg/
        path = 'cfg' + os.sep + path

    with open(path, 'r') as f:
        lines = f.read().split('\n')  # 读取cfg文件的每一行存入列表
    lines = [x for x in lines if x and not x.startswith('#')]  # 删除掉空行或者以"#"开头的注释行
    lines = [x.rstrip().lstrip() for x in lines]  # 去掉左右两边的空格
    mdefs = []  # 列表存储模块的定义
    for line in lines:
        if line.startswith('['):  # 当遇到 '['时，意味着要新建一个字典了，也就说每个字典对应一个新块 cfg文件以[]代表一个模块
            mdefs.append({})  # 现在最后添加一个空字典，后面直接用 -1 进行索引
            mdefs[-1]['type'] = line[1:-1].rstrip()  # type为[]中当前模块类型，eg: convolutional
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0  # 如果是卷积曾，则添加pre-populate with zeros (may be overwritten later)
        else:
            key, val = line.split("=")  # 以"="分割两边的key和value
            key = key.rstrip()  # 去除两边的空格

            if key == 'anchors':  # 是锚框，返回nparray
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))  # 以,分开锚框数据 -> np anchors
            elif (key in ['from', 'layers', 'mask']) or (key == 'size' and ',' in val):  # return array
                mdefs[-1][key] = [int(x) for x in val.split(',')]
            else:
                val = val.strip()
                if val.isnumeric():  # 返回 int 或 float
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val  # return string

    # Check all fields are supported
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh']

    f = []  # fields
    for x in mdefs[1:]:  # 从1而不是0开始，是因为第一个[net]默认存储网络相关参数，并不是模块层
        [f.append(k) for k in x if k not in f]  # 这个列表[]仅仅是为了python语法正确，去掉了语法错误，无意义
    u = [x for x in f if x not in supported]  # unsupported fields
    # 有未定义的field则报错
    assert not any(u), "Unsupported fields %s in %s. See https://github.com/ultralytics/yolov3/issues/631" % (u, path)

    return mdefs


# 处理*.data文件到options这个dict中
def parse_data_cfg(path):
    # Parses the data configuration file
    if not os.path.exists(path) and os.path.exists('data' + os.sep + path):  # add data/ prefix if omitted
        path = 'data' + os.sep + path

    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options


if __name__ == '__main__':
    print(parse_model_cfg('../cfg/yolov3.cfg'))
