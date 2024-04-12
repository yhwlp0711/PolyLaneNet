import os
import json
import random

import numpy as np
from tabulate import tabulate

from utils.lane import LaneEval
from utils.metric import eval_json

SPLIT_FILES = {
    'train+val': ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
    'train': ['label_data_0313.json', 'label_data_0601.json'],
    'val': ['label_data_0531.json'],
    'test': ['test_label.json'],
}


class TuSimple(object):
    def __init__(self, split='train', max_lanes=None, root=None, metric='default'):
        self.split = split
        self.root = root
        self.metric = metric

        # 检查给定的 split 是否存在于 SPLIT_FILES 字典中
        if split not in SPLIT_FILES.keys():
            raise Exception('Split `{}` does not exist.'.format(split))

        # 所有路径
        self.anno_files = [os.path.join(self.root, path) for path in SPLIT_FILES[split]]

        if root is None:
            raise Exception('Please specify the root directory')

        self.img_w, self.img_h = 1280, 720
        self.max_points = 0
        self.load_annotations()

        # Force max_lanes, used when evaluating testing with models trained on other datasets
        # 强制设定 max_lanes，用于在测试时评估使用在其他数据集上训练的模型
        if max_lanes is not None:
            self.max_lanes = max_lanes

    def get_img_heigth(self, path):
        return 720

    def get_img_width(self, path):
        return 1280

    def get_metrics(self, lanes, idx):
        # 获取评估指标
        label = self.annotations[idx]
        org_anno = label['old_anno']
        # 预测转车道
        pred = self.pred2lanes(org_anno['path'], lanes, org_anno['y_samples'])
        _, _, _, matches, accs, dist = LaneEval.bench(pred, org_anno['org_lanes'], org_anno['y_samples'], 0, True)

        # 匹配情况、准确率、距离
        return matches, accs, dist

    def pred2lanes(self, path, pred, y_samples):
        # 预测的车道线转为具体车道线的坐标点
        # pred是多项式系数？

        # 采样点的y标准化到[0,1]
        ys = np.array(y_samples) / self.img_h
        lanes = []
        for lane in pred:
            if lane[0] == 0:
                # 可能表示无效车道线
                continue
            # lane[3:]为多项式的系数数组，ys为自变量数组
            # 返回对应的因变量数组
            # 乘以img_w 转为水平坐标
            lane_pred = np.polyval(lane[3:], ys) * self.img_w
            # 不在[lane[1],lane[2]]范围内的点的水平坐标设置为-2
            lane_pred[(ys < lane[1]) | (ys > lane[2])] = -2
            lanes.append(list(lane_pred))

        return lanes

    def load_annotations(self):
        # 加载annotations
        self.annotations = []
        max_lanes = 0
        # 遍历路径
        for anno_file in self.anno_files:
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line)
                y_samples = data['h_samples']
                gt_lanes = data['lanes']
                # 它遍历了原始的车道线数据 gt_lanes，其中每个元素都是一个包含 x 坐标的列表，表示车道线上的点。
                # 然后，使用 zip 函数将每个车道线点的 x 坐标和对应的采样点 y_samples 中的 y 坐标一一配对
                # 提取x >= 0的点
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
                # 提取lanes中长度大于0的元素
                lanes = [lane for lane in lanes if len(lane) > 0]
                max_lanes = max(max_lanes, len(lanes))
                # 最多点的车道线的点的个数
                self.max_points = max(self.max_points, max([len(l) for l in gt_lanes]))
                self.annotations.append({
                    'path': os.path.join(self.root, data['raw_file']),
                    'org_path': data['raw_file'],
                    'org_lanes': gt_lanes,
                    'lanes': lanes,
                    'aug': False,
                    'y_samples': y_samples
                })

        if self.split == 'train':
            # 打乱
            random.shuffle(self.annotations)
        print('total annos', len(self.annotations))
        self.max_lanes = max_lanes

    def transform_annotations(self, transform):
        # transform为转换函数
        self.annotations = list(map(transform, self.annotations))

    def pred2tusimpleformat(self, idx, pred, runtime):
        runtime *= 1000.  # s to ms
        img_name = self.annotations[idx]['old_anno']['org_path']
        # 垂直坐标
        h_samples = self.annotations[idx]['old_anno']['y_samples']
        # 预测转车道点
        lanes = self.pred2lanes(img_name, pred, h_samples)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        # 转成json
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, runtimes, filename):
        # 保存预测结果
        lines = []
        for idx in range(len(predictions)):
            # 预测转tusimple格式
            line = self.pred2tusimpleformat(idx, predictions[idx], runtimes[idx])
            lines.append(line)
        # 获取文件所在的目录
        directory = os.path.dirname(filename)
        # 如果目录不存在，则创建它
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def eval(self, exp_dir, predictions, runtimes, label=None, only_metrics=False):
        # 评估
        # label替换{}
        pred_filename = '/tmp/tusimple_predictions_{}.json'.format(label)
        # 保存预测结果
        self.save_tusimple_predictions(predictions, runtimes, pred_filename)
        if self.metric == 'default':
            # 调用bench_one_submit
            # 用于评估单个预测结果和对应的地面真实车道线之间的匹配情况。
            # 可以计算总体准确率、假阳率和假阴率
            # 每个预测车道线的匹配情况、准确率和距离信息（如果指定了 get_matches=True）。
            # 首先会检查预测的车道线格式是否正确，然后根据预测的车道线数量和运行时间进行一些限制，以避免无效的评估。
            # 主要的评估逻辑在遍历真实车道线和预测车道线的过程中完成，
            # 包括计算每条车道线的角度、阈值，以及评估每个预测车道线和真实车道线之间的匹配情况。
            result = json.loads(LaneEval.bench_one_submit(pred_filename, self.anno_files[0]))
        elif self.metric == 'ours':
            # 调用eval_json
            # 用于评估一组预测结果和对应的地面真实车道线集合之间的匹配情况。
            # 首先会加载预测结果和地面真实车道线数据，并对它们进行一些格式转换和预处理。
            # 然后遍历所有的预测结果，针对每个预测结果调用 area_metric 函数计算其与地面真实车道线的匹配情况。
            # 最后根据评估结果计算总体准确率、假阳率、假阴率和平均帧率，并将结果以 JSON 格式返回。
            result = json.loads(eval_json(pred_filename, self.anno_files[0], json_type='tusimple'))
        table = {}
        for metric in result:
            # 使用指标名称作为列标题，并存入指标值
            # eg.
            # metric = {
            #     'name': 'Accuracy',
            #     'value': 0.85,
            #     'order': 'desc'
            # }
            table[metric['name']] = [metric['value']]
        # 将表格数据table转换为具有键头的表格字符串
        table = tabulate(table, headers='keys')

        if not only_metrics: # defaut = false
            # 保存包含完整评估结果的JSON文件
            filename = 'tusimple_{}_eval_result_{}.json'.format(self.split, label)
            with open(os.path.join(exp_dir, filename), 'w') as out_file:
                json.dump(result, out_file)

        return table, result

    def __getitem__(self, idx):
        # object[idx]
        return self.annotations[idx]

    def __len__(self):
        # len(object)
        return len(self.annotations)
