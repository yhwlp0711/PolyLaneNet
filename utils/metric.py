import argparse
from pprint import pprint

import cv2
import numpy as np
import ujson as json
from tqdm import tqdm
from tabulate import tabulate
from scipy.spatial import distance


def show_preds(pred, gt):
    # 可视化真实车道线和预测车道线
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    print(len(gt), 'gts and', len(pred), 'preds')
    for lane in gt:
        for p in lane:
            cv2.circle(img, tuple(map(int, p)), 5, thickness=-1, color=(255, 0, 255))
    for lane in pred:
        for p in lane:
            cv2.circle(img, tuple(map(int, p)), 4, thickness=-1, color=(0, 255, 0))
    cv2.imshow('img', img)
    cv2.waitKey(0)


def area_distance(pred_x, pred_y, gt_x, gt_y, placeholder=np.nan):
    # 两个一维数组堆叠再转置 [(xi,yi),...]
    pred = np.vstack([pred_x, pred_y]).T
    gt = np.vstack([gt_x, gt_y]).T

    # pred = pred[pred[:, 0] > 0][:3, :]
    # gt = gt[gt[:, 0] > 0][:5, :]

    # 两两点之间计算欧式距离
    dist_matrix = distance.cdist(pred, gt, metric='euclidean')

    # axis=0:列 axis=1:行
    # (每列最小值之和+每行最小值之和)/2
    dist = 0.5 * (np.min(dist_matrix, axis=0).sum() + np.min(dist_matrix, axis=1).sum())
    dist /= np.max(gt_y) - np.min(gt_y)
    return dist


def area_metric(pred, gt, debug=None):
    # 根据每条车道线的第一个点与图像中心点的水平距离的绝对值排序 然后后选择距离最近的两条车道线
    pred = sorted(pred, key=lambda ps: abs(ps[0][0] - 720/2.))[:2]
    gt = sorted(gt, key=lambda ps: abs(ps[0][0] - 720/2.))[:2]
    if len(pred) == 0:
        return 0., 0., len(gt)
    line_dists = []
    fp = 0.
    matched = 0.
    gt_matches = [False] * len(gt)
    pred_matches = [False] * len(pred)
    pred_dists = [None] * len(pred)

    # shape = len(gt) * len(pred) init = 1.0
    distances = np.ones((len(gt), len(pred)), dtype=np.float32)
    # 分别提取真实车道线point的横纵坐标
    for i_gt, gt_points in enumerate(gt):
        # 对于每条真实车道线都遍历一遍预测车道线    len(gt) * len(pred)
        x_gts = [x for x, _ in gt_points]
        y_gts = [y for _, y in gt_points]
        # 分别提取预测车道线point的横纵坐标
        for i_pred, pred_points in enumerate(pred):
            x_preds = [x for x, _ in pred_points]
            y_preds = [y for _, y in pred_points]
            # 计算当前真实车道线和预测车道线的距离
            distances[i_gt, i_pred] = area_distance(x_preds, y_preds, x_gts, y_gts)

    # 每行中最小值的列索引 真实车道线与哪个预测车道线最近
    best_preds = np.argmin(distances, axis=1)
    # 每列中最小值的行索引 预测车道线与哪个真实车道线最近
    best_gts = np.argmin(distances, axis=0)
    fp = 0.
    fn = 0.
    dist = 0.
    is_fp = []
    is_fn = []
    for i_pred, best_gt in enumerate(best_gts):
        # 获取距当前真实车道线最近的预测车道线 然后判断该预测车道线最近的真实车道线是否为当前真实车道线
        if best_preds[best_gt] == i_pred:
            dist += distances[best_gt, i_pred]
            is_fp.append(False)
        else:
            # 假阳性+1
            # 无车道线->有车道线
            fp += 1
            is_fp.append(True)
    for i_gt, best_pred in enumerate(best_preds):
        if best_gts[best_pred] != i_gt:
            # 假阴性+1
            # 有->无
            fn += 1
            is_fn.append(True)
        else:
            is_fn.append(False)
    if debug: # default = None
        print('is fp')
        print(is_fp)
        print('is fn')
        print(is_fn)
        print('distances')
        dists = np.min(distances, axis=0)
        dists[np.array(is_fp)] = 0
        print(dists)
        show_preds(pred, gt)

    # 返回匹配的真实车道线和预测车道线之间的距离之和、fp、fn
    return dist, fp, fn


def convert_tusimple_format(json_gt):
    # JSON->TUSimple
    output = []
    for data in json_gt:
        # 提取x>0
        lanes = [[(x, y) for (x, y) in zip(lane, data['h_samples']) if x >= 0] for lane in data['lanes']
                 if any(x > 0 for x in lane)]
        output.append({
            'raw_file': data['raw_file'],
            'run_time': data['run_time'] if 'run_time' in data else None,
            'lanes': lanes
        })
    # 函数返回一个列表，其中每个元素都是一个字典，包含了转换后的TuSimple格式的标注信息
    return output


def eval_json(pred_file, gt_file, json_type=None, debug=False):
    # 封装
    try:
        json_pred = [json.loads(line) for line in open(pred_file).readlines()]
    except BaseException as e:
        raise Exception('Fail to load json file of the prediction.')
    json_gt = [json.loads(line) for line in open(gt_file).readlines()]
    if len(json_gt) != len(json_pred):
        raise Exception('We do not get the predictions of all the test tasks')

    if json_type == 'tusimple':
        for gt, pred in zip(json_gt, json_pred):
            pred['h_samples'] = gt['h_samples']
        json_gt = convert_tusimple_format(json_gt)
        json_pred = convert_tusimple_format(json_pred)
    gts = {l['raw_file']: l for l in json_gt}

    total_distance, total_fp, total_fn, run_time = 0., 0., 0., 0.
    for pred in tqdm(json_pred):
        if 'raw_file' not in pred or 'lanes' not in pred:
            raise Exception('raw_file or lanes not in some predictions.')
        raw_file = pred['raw_file']
        pred_lanes = pred['lanes']
        run_time += pred['run_time'] if 'run_time' in pred else 1.

        if raw_file not in gts:
            raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
        gt = gts[raw_file]
        gt_lanes = gt['lanes']

        distance, fp, fn = area_metric(pred_lanes, gt_lanes, debug=debug)

        total_distance += distance
        total_fp += fp
        total_fn += fn

    num = len(gts)
    return json.dumps([{
        'name': 'Distance',
        'value': total_distance / num, # 平均distance
        'order': 'desc'
    }, {
        'name': 'FP',
        'value': total_fp,
        'order': 'asc'
    }, {
        'name': 'FN',
        'value': total_fn,
        'order': 'asc'
    }, {
        'name': 'FPS',
        'value': 1000. * num / run_time # FPS
    }])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute the metrics")
    parser.add_argument('--preds', required=True, type=str, help=".json with the predictions")
    parser.add_argument('--gt', required=True, type=str, help=".json with the GT")
    parser.add_argument('--gt-type', type=str, help='pass `tusimple` if using the TuSimple file format')
    parser.add_argument('--debug', action='store_true', help='show metrics and preds/gts')
    argv = vars(parser.parse_args())

    result = json.loads(eval_json(argv['preds'], argv['gt'], argv['gt_type'], argv['debug']))

    # pretty-print
    table = {}
    for metric in result:
        if metric['name'] not in table.keys():
            table[metric['name']] = []
        table[metric['name']].append(metric['value'])
    print(tabulate(table, headers='keys'))
