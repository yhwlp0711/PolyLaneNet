import numpy as np
import ujson as json
from sklearn.linear_model import LinearRegression

# 评估结果

class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20     # pixel阈值
    pt_thresh = 0.85      # point阈值

    @staticmethod
    def get_angle(xs, y_samples): # x,y进行线性拟合后计算角度
        # 筛选x坐标大于等于0的点
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        # 如果得到的点个数大于1
        if len(xs) > 1:
            # 进行线性拟合
            LaneEval.lr.fit(ys[:, None], xs)
            # 获取斜率
            k = LaneEval.lr.coef_[0]
            # 计算角度
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh): # 计算线性精度
        # 小于 0 的值替换为 -100，以排除无效点对精度计算的影响
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        # 计算预测结果和真实标签之间的绝对误差 与阈值thresh比较
        # 其中满足条件的位置值为 1.0，不满足条件的位置值为 0.0
        # 通过计算比较结果中小于阈值的元素数量，并除以真实标签的长度，得到线性精度
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def distances(pred, gt):
        # 预测值与真实值相减取绝对值
        return np.abs(pred - gt)

    @staticmethod
    def bench(pred, gt, y_samples, running_time, get_matches=False):
        if any(len(p) != len(y_samples) for p in pred):
            # 遍历预测的车道线 pred 中的每个车道，检查每个车道的长度是否与给定的垂直采样 y_samples 的长度相匹配。
            raise Exception('Format of lanes error.')
        if running_time > 20000 or len(gt) + 2 < len(pred):
            # 如果 running_time 大于 20000 毫秒（即 20 秒），或者预测的车道线数量比地面真实车道线数量多 2 条以上
            # 第一个值为 0. 表示准确度为 0；
            # 第二个值为 0. 表示假阳率为 0；
            # 第三个值为 1. 表示假阴率为 1。
            return 0., 0., 1.
        # 计算每条车道线在图像上的角度
        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        # 根据角度计算每条车道线的阈值
        # 根据车道线的角度调整阈值，使得在斜率较大的车道线上更容易满足阈值要求
        # angle越大 cos(angle)越小 threshs越大
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        # 初始化
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        my_matches = [False] * len(pred)
        my_accs = [0] * len(pred)
        my_dists = [None] * len(pred)
        for x_gts, thresh in zip(gt, threshs):# (真实车道线，阈值)
            # 每条车道线的线性精度
            accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            # 将较大的值保留在 my_accs 中相应的位置上
            my_accs = np.maximum(my_accs, accs)
            # accs的最大值
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            # 预测值和真实值相减取绝对值
            my_dist = [LaneEval.distances(np.array(x_preds), np.array(x_gts)) for x_preds in pred]
            if len(accs) > 0:
                # 找到精度最大的索引
                my_dists[np.argmax(accs)] = {
                    # 获取所有采样点的垂直位置，然后根据真实车道线的有效索引（即大于等于0）筛选出相应的高度，并转换为整数列表
                    # 具有最大准确度的预测车道线与真实车道线之间的距离列表
                    'y_gts': list(np.array(y_samples)[np.array(x_gts) >= 0].astype(int)),
                    'dists': list(my_dist[np.argmax(accs)])
                }

            if max_acc < LaneEval.pt_thresh:
                # 则说明预测的车道线与真实车道线之间的匹配准确度不足，被认为是一个假阴性
                fn += 1
            else:
                # 预测车道线与真实匹配
                my_matches[np.argmax(accs)] = True
                matched += 1
            line_accs.append(max_acc)
        # 假阳性
        fp = len(pred) - matched
        # ？
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
        # ？

        # 如果get_matches参数为真，则返回总体准确率、假阳率、假阴率以及每个预测车道线的匹配情况、准确率和距离信息。
        # 如果为假，则只返回总体准确率、假阳率和假阴率。
        if get_matches: # default = false
            return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(
                min(len(gt), 4.), 1.), my_matches, my_accs, my_dists
        return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(min(len(gt), 4.), 1.)

    @staticmethod
    def bench_one_submit(pred_file, gt_file):
        # 对车道检测模型在测试数据集上的性能评估，并将评估结果以 JSON 格式返回，以便进行排名和分析。
        try:
            json_pred = [json.loads(line) for line in open(pred_file).readlines()]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')
        gts = {l['raw_file']: l for l in json_gt}
        accuracy, fp, fn = 0., 0., 0.
        run_times = []
        for pred in json_pred:
            if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
                raise Exception('raw_file or lanes or run_time not in some predictions.')
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']
            run_time = pred['run_time']
            run_times.append(run_time)
            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]
            gt_lanes = gt['lanes']
            y_samples = gt['h_samples']
            try:
                a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)
            except BaseException as e:
                raise Exception('Format of lanes error.')
            accuracy += a
            fp += p
            fn += n
        num = len(gts)
        # the first return parameter is the default ranking parameter
        return json.dumps([{
            'name': 'Accuracy',
            'value': accuracy / num,
            'order': 'desc'
        }, {
            'name': 'FP',
            'value': fp / num,
            'order': 'asc'
        }, {
            'name': 'FN',
            'value': fn / num,
            'order': 'asc'
        }, {
            'name': 'FPS',
            'value': 1000. / np.mean(run_times)
        }])


if __name__ == '__main__':
    import sys
    try:
        if len(sys.argv) != 3:
            raise Exception('Invalid input arguments')
        print(LaneEval.bench_one_submit(sys.argv[1], sys.argv[2]))
    except Exception as e:
        print(e)
        # sys.exit(e.message)
