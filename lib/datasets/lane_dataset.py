import cv2
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmenters import Resize
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset
from imgaug.augmentables.lines import LineString, LineStringsOnImage

from .elas import ELAS
from .llamas import LLAMAS
from .tusimple import TuSimple
from .nolabel_dataset import NoLabelDataset

GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class LaneDataset(Dataset):
    def __init__(self,
                 dataset='tusimple',
                 augmentations=None,
                 normalize=False,
                 split='train',
                 img_size=(360, 640),
                 aug_chance=1.,
                 **kwargs):
        super(LaneDataset, self).__init__()
        if dataset == 'tusimple':
            # 加载TuSimple数据集
            self.dataset = TuSimple(split=split, **kwargs)
        elif dataset == 'llamas':
            self.dataset = LLAMAS(split=split, **kwargs)
        elif dataset == 'elas':
            self.dataset = ELAS(split=split, **kwargs)
        elif dataset == 'nolabel_dataset':
            self.dataset = NoLabelDataset(**kwargs)
        else:
            raise NotImplementedError()

        # 转格式
        self.transform_annotations()
        self.img_h, self.img_w = img_size

        if augmentations is not None:
            # add augmentations
            # 遍历 augmentations 列表中的每个字典，每个字典包含了一个增强操作的名称和参数
            # 然后，它使用 getattr 函数根据名称获取 imgaug 库中对应的增强函数，
            # 并将其实例化为增强操作对象，传入相应的参数。
            # 最终，将所有实例化的增强操作对象保存在列表 augmentations 中
            augmentations = [getattr(iaa, aug['name'])(**aug['parameters'])
                             for aug in augmentations]  # add augmentation

        self.normalize = normalize # default = false
        # resize
        transformations = iaa.Sequential([Resize({'height': self.img_h, 'width': self.img_w})])
        self.to_tensor = ToTensor()
        # 第一个变换序列 augmentations 是一个列表，包含了一系列的数据增强操作，
        # 每个操作由 iaa.Sometimes 定义，意味着它们只有一定的概率会被应用到样本上
        # 概率由 p=aug_chance 控制
        # 第二个变换序列为 transformations(resize)
        self.transform = iaa.Sequential([iaa.Sometimes(then_list=augmentations, p=aug_chance), transformations])
        self.max_lanes = self.dataset.max_lanes

    def transform_annotation(self, anno, img_wh=None):
        # 返回新的 anno
        if img_wh is None:
            # def get_img_heigth(self, path): return 720
            # def get_img_width(self, path): return 1280
            img_h = self.dataset.get_img_heigth(anno['path'])
            img_w = self.dataset.get_img_width(anno['path'])
        else:
            img_w, img_h = img_wh

        # 提取原始车道线数据，它存储了每条车道线的坐标信息
        old_lanes = anno['lanes']
        # 如果包含了类别信息，则提取；否则将所有车道线都默认为同一类别
        categories = anno['categories'] if 'categories' in anno else [1] * len(old_lanes)
        # 其中每个元素是一个元组，包含 old_lanes 中的一个子列表和 categories 中的一个元素
        old_lanes = zip(old_lanes, categories)
        # 过滤条件：每个子列表的第一个元素的长度大于 0
        old_lanes = filter(lambda x: len(x[0]) > 0, old_lanes)
        # max_lanes行，每行代表一条车道线
        # 列数？
        # 初始化为 -1e5
        lanes = np.ones((self.dataset.max_lanes, 1 + 2 + 2 * self.dataset.max_points), dtype=np.float32) * -1e5
        # 所有行第一列设置为 0
        lanes[:, 0] = 0
        # 根据每条车道线第一个点的 x 坐标排序
        old_lanes = sorted(old_lanes, key=lambda x: x[0][0][0])
        for lane_pos, (lane, category) in enumerate(old_lanes):
            lower, upper = lane[0][1], lane[-1][1]
            # 提取 x, y 然后归一化
            xs = np.array([p[0] for p in lane]) / img_w
            ys = np.array([p[1] for p in lane]) / img_h
            # 前三个元素为类别、归一化下界、归一化上界
            lanes[lane_pos, 0] = category
            lanes[lane_pos, 1] = lower / img_h
            lanes[lane_pos, 2] = upper / img_h
            # 第四个元素开始为归一化的 x 坐标
            lanes[lane_pos, 3:3 + len(xs)] = xs
            # 3 + self.dataset.max_points 开始为归一化的 y 坐标
            lanes[lane_pos, (3 + self.dataset.max_points):(3 + self.dataset.max_points + len(ys))] = ys

        new_anno = {
            'path': anno['path'], # 路径
            'label': lanes, # new_lanes
            'old_anno': anno, # 原始信息
            'categories': [cat for _, cat in old_lanes] # 类别
        }

        return new_anno

    @property
    def annotations(self):
        return self.dataset.annotations

    def transform_annotations(self):
        print('Transforming annotations...')
        # 使用self.transform_annotation作为转换函数 转换self.dataset.annotations
        # ->列表->nparray
        self.dataset.annotations = np.array(list(map(self.transform_annotation, self.dataset.annotations)))
        print('Done.')

    def draw_annotation(self, idx, pred=None, img=None, cls_pred=None):
        # 这个函数用于 annotations 和 predictions 在图像上的可视化
        if img is None: # 未提供图像 读入
            img, label, _ = self.__getitem__(idx, transform=True)
            # Tensor to opencv image
            # Tensor 转为 nparray
            img = img.permute(1, 2, 0).numpy()
            # Unnormalize
            # 如果进行了归一化则再进行反归一化
            if self.normalize:
                # 图像数据乘以标准差，然后加上均值，从而得到原始的未经归一化处理的图像数据
                img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            # float->unsigned int
            # [0,255]
            img = (img * 255).astype(np.uint8)
        else:
            _, label, _ = self.__getitem__(idx)

        # 初始化 shape
        img_h, img_w, _ = img.shape

        # Draw label
        # 绘制真实车道
        for i, lane in enumerate(label):
            # 跳过无效车道线
            if lane[0] == 0:  # Skip invalid lanes
                continue
            # 移除置信度、上限、下限
            lane = lane[3:]  # remove conf, upper and lower positions
            # 分离 x、y 坐标
            # x 和 y 坐标可能被连续地存储在一起，前一半是 x 坐标，后一半是 y 坐标
            xs = lane[:len(lane) // 2]
            ys = lane[len(lane) // 2:]
            # 提取有效坐标
            ys = ys[xs >= 0]
            xs = xs[xs >= 0]

            # draw GT points
            # 根据坐标绘制
            for p in zip(xs, ys):
                # 调整尺寸
                p = (int(p[0] * img_w), int(p[1] * img_h))
                img = cv2.circle(img, p, 5, color=GT_COLOR, thickness=-1)

            # draw GT lane ID
            cv2.putText(img,
                        str(i), (int(xs[0] * img_w), int(ys[0] * img_h)),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1,
                        color=(0, 255, 0))

        if pred is None:
            # 预测为空
            return img

        # Draw predictions
        # 绘制预测车道
        # 过滤无效车道（置信度为0）
        pred = pred[pred[:, 0] != 0]  # filter invalid lanes
        # 获取评估指标 匹配情况、准确率、距离
        matches, accs, _ = self.dataset.get_metrics(pred, idx)
        overlay = img.copy()
        for i, lane in enumerate(pred):
            if matches[i]:
                # 预测与真实匹配 绿色
                color = PRED_HIT_COLOR
            else:
                # 预测与真实不匹配 红色
                color = PRED_MISS_COLOR
            # 移除置信度
            lane = lane[1:]  # remove conf
            # 获取上界、下界
            lower, upper = lane[0], lane[1]
            # 移除上界、下界
            lane = lane[2:]  # remove upper, lower positions

            # generate points from the polynomial
            # 从多项式生成点
            # lower 和 upper 之间生产 100 个均匀数字 作为生成车道线点的垂直位置
            ys = np.linspace(lower, upper, num=100)
            # 初始化
            points = np.zeros((len(ys), 2), dtype=np.int32)
            # 映射到范围内并存在第二列
            points[:, 1] = (ys * img_h).astype(int)
            # 根据多项式系数和 ys 生产 xs 映射后存在第一列
            points[:, 0] = (np.polyval(lane, ys) * img_w).astype(int)
            # 过滤范围外的点
            points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]

            # draw lane with a polyline on the overlay
            # 在覆盖层上绘制车道线的折线
            # overlay = img.copy()
            for current_point, next_point in zip(points[:-1], points[1:]):
                # 相邻点对相连
                overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=color, thickness=2)

            # draw class icon
            # TuSimple 中貌似没用
            if cls_pred is not None and len(points) > 0: # defaul = None
                class_icon = self.dataset.get_class_icon(cls_pred[i])
                class_icon = cv2.resize(class_icon, (32, 32))
                mid = tuple(points[len(points) // 2] - 60)
                x, y = mid

                img[y:y + class_icon.shape[0], x:x + class_icon.shape[1]] = class_icon

            # draw lane ID
            # 点的数量大于 0 则绘制车道线 ID
            if len(points) > 0:
                cv2.putText(img, str(i), tuple(points[0]), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=color)

            # draw lane accuracy
            # 点的数量大于 0 则绘制准确率，保留两位小数
            if len(points) > 0:
                cv2.putText(img,
                            '{:.2f}'.format(accs[i] * 100),
                            tuple(points[len(points) // 2] - 30),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=.75,
                            color=color)
        # Add lanes overlay
        # 使用 0.6 的混合权重对 img 和 overlay 进行混合
        w = 0.6
        img = ((1. - w) * img + w * overlay).astype(np.uint8)

        return img

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            # 每条车道线的点转为 LineString
            # LineString是由一系列连接的线段组成的集合，每个线段都有起点和终点。
            # 每个点都以其 (x, y) 坐标给出。每个段的终点也是下一个段的起点
            lines.append(LineString(lane))

        return lines

    def linestrings_to_lanes(self, lines):
        # LineString().coords = lane
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes

    def __getitem__(self, idx, transform=True):
        item = self.dataset[idx]
        # 读取图像和 label
        img = cv2.imread(item['path'])
        label = item['label']
        if transform: # default = true 指定是否需要转换  img 和 label 进行相应转换
            # 车道转 LineString
            line_strings = self.lane_to_linestrings(item['old_anno']['lanes'])
            # 得到 LinrStringsOnImage 对象
            # 第一个变换序列 augmentations 是一个列表，包含了一系列的数据增强操作，
            # 每个操作由 iaa.Sometimes 定义，意味着它们只有一定的概率会被应用到样本上
            # 概率由 p=aug_chance 控制
            # 第二个变换序列为 transformations(resize)
            line_strings = LineStringsOnImage(line_strings, shape=img.shape)
            img, line_strings = self.transform(image=img, line_strings=line_strings)
            # 剪切图片外
            line_strings.clip_out_of_image_()
            # LineString 转lanes
            new_anno = {'path': item['path'], 'lanes': self.linestrings_to_lanes(line_strings)}
            new_anno['categories'] = item['categories']
            # 得到新的 anno
            label = self.transform_annotation(new_anno, img_wh=(self.img_w, self.img_h))['label']

        # 归一化 [0，1]
        img = img / 255.
        if self.normalize:
            # 进一步处理
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        # 转 tensor
        img = self.to_tensor(img.astype(np.float32))
        # 返回 img、label、idx
        return (img, label, idx)

    def __len__(self):
        return len(self.dataset)


def main():
    import torch
    from lib.config import Config
    # 随机数种子
    np.random.seed(0)
    torch.manual_seed(0)
    # 加载 ymal 文件
    cfg = Config('config.yaml')
    # 获取 train 数据集
    train_dataset = cfg.get_dataset('train')
    for idx in range(len(train_dataset)):
        # 绘制并站hi是
        img = train_dataset.draw_annotation(idx)
        cv2.imshow('sample', img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
