import numpy as np
import torch.distributed as dist
import torch

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

@torch.no_grad()
def cubic_bezier_curve_segment(control_points, sample_points):
    # Cut a batch of cubic bezier curves to its in-image segments (assume at least 2 valid sample points per curve).
    # Based on De Casteljau's algorithm, formula for cubic bezier curve is derived by:
    # https://stackoverflow.com/a/11704152/15449902
    # 批次为 B
    # control_points: B x 4 x 2
    # 样本点
    # sample_points: B x N x 2
    # 返回一个形状为 B x 4 x 2 的张量，代表切割后的贝塞尔曲线的控制点
    if control_points.numel() == 0 or sample_points.numel() == 0:
        return control_points
    B, N = sample_points.shape[:-1]
    # 获取有效的样本点的布尔掩码
    valid_points = get_valid_points(sample_points)  # B x N, bool
    # 生成参数 t，表示每个样本点在曲线上的位置
    t = torch.linspace(0.0, 1.0, steps=N, dtype=sample_points.dtype, device=sample_points.device)

    # First & Last valid index (B)
    # Get unique values for deterministic behaviour on cuda:
    # https://pytorch.org/docs/1.6.0/generated/torch.max.html?highlight=max#torch.max
    # 计算每条曲线的起始和结束参数 t0 和 t1，用于切割曲线
    t0 = t[(valid_points + torch.arange(N, device=valid_points.device).flip([0]) * valid_points).max(dim=-1).indices]
    t1 = t[(valid_points + torch.arange(N, device=valid_points.device) * valid_points).max(dim=-1).indices]

    # Generate transform matrix (old control points -> new control points = linear transform)
    # 将原始的控制点映射到新的控制点。这个变换是线性的，即可以用一个矩阵乘法来表示
    u0 = 1 - t0  # B
    u1 = 1 - t1  # B
    transform_matrix_c = [torch.stack([u0 ** (3 - i) * u1 ** i for i in range(4)], dim=-1),
                          torch.stack([3 * t0 * u0 ** 2,
                                       2 * t0 * u0 * u1 + u0 ** 2 * t1,
                                       t0 * u1 ** 2 + 2 * u0 * u1 * t1,
                                       3 * t1 * u1 ** 2], dim=-1),
                          torch.stack([3 * t0 ** 2 * u0,
                                       t0 ** 2 * u1 + 2 * t0 * t1 * u0,
                                       2 * t0 * t1 * u1 + t1 ** 2 * u0,
                                       3 * t1 ** 2 * u1], dim=-1),
                          torch.stack([t0 ** (3 - i) * t1 ** i for i in range(4)], dim=-1)]
    transform_matrix = torch.stack(transform_matrix_c, dim=-2).transpose(-2, -1)  # B x 4 x 4, f**k this!
    transform_matrix = transform_matrix.unsqueeze(1).expand(B, 2, 4, 4)

    # Matrix multiplication
    # 矩阵乘法
    res = transform_matrix.matmul(control_points.permute(0, 2, 1).unsqueeze(-1))  # B x 2 x 4 x 1

    return res.squeeze(-1).permute(0, 2, 1)

@torch.no_grad()
def get_valid_points(points):
    # ... x 2
    # 首先检查 points 是否为空
    if points.numel() == 0:
        return torch.tensor([1], dtype=torch.bool, device=points.device)
    # 对于非空的点集，使用逐元素的逻辑运算符 * 来检查每个点的坐标是否都在 (0, 1) 范围内
    return (points[..., 0] > 0) * (points[..., 0] < 1) * (points[..., 1] > 0) * (points[..., 1] < 1)

def get_src_permutation_idx(indices):
    # Permute predictions following indices
    # 对下列指数进行置换预测
    # 2-dim indices: (dim0 indices, dim1 indices)
    # 接受一个索引列表 indices，这个列表包含两个维度的索引：dim0 indices 和 dim1 indices
    # batch_idx: 是一个一维张量，包含了每个预测的批次索引。
    # image_idx: 是一个一维张量，包含了每个预测在其批次内的图像索引
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    image_idx = torch.cat([src for (src, _) in indices])

    return batch_idx, image_idx

def upcast(t):
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    # https://github.com/pytorch/vision/pull/3383
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

def lane_pruning(existence, existence_conf, max_lane):
    # Prune lanes based on confidence (a max number constrain for lanes in an image)
    # Maybe too slow (but should be faster than topk/sort),
    # consider batch size >> max number of lanes
    # 基于车道线存在概率进行车道线修剪，确保每个图像中检测到的车道线数量不超过max_lane
    while (existence.sum(dim=1) > max_lane).sum() > 0:
        indices = (existence.sum(dim=1, keepdim=True) > max_lane).expand_as(existence) * \
                  (existence_conf == existence_conf.min(dim=1, keepdim=True).values)
        existence[indices] = 0
        existence_conf[indices] = 1.1  # So we can keep using min

    return existence, existence_conf


def bezier_to_coordinates(control_points, existence, resize_shape, dataset, bezier_curve, ppl=56, gap=10):
    # control_points: L x N x 2
    # control_points: 二维数组，形状为 L x N x 2，其中L是车道线的数量，N是每条车道线的控制点数量，2表示(x, y)坐标。
    # existence: 一个布尔数组，表示每条车道线是否存在。
    # resize_shape: 一个元组，表示调整后的图像形状，形如(H, W)。
    # dataset: 字符串，表示数据集类型，可以是'tusimple'、'culane'或'llamas'。
    # bezier_curve: 贝塞尔曲线对象，用于计算贝塞尔曲线上的点。
    # ppl: 整数，表示在'tusimple'数据集上采样的点数量。
    # gap: 整数，表示在'tusimple'数据集上采样点之间的垂直间隔
    H, W = resize_shape
    cps_of_lanes = []
    # 根据existence数组筛选出存在的车道线的控制点，并转换为列表格式
    for flag, cp in zip(existence, control_points):
        if flag:
            cps_of_lanes.append(cp.tolist())
    coordinates = []
    for cps_of_lane in cps_of_lanes:
        bezier_curve.assign_control_points(cps_of_lane)
        if dataset == 'tusimple':
            # Find x for TuSimple's fixed y eval positions (suboptimal)
            # 存在一组固定的y坐标，代表图像的垂直位置。对于这些固定的y坐标，
            # 代码需要确定对应的x坐标，即贝塞尔曲线在这些y坐标位置上的交点
            # 设置一个阈值，用于判断贝塞尔曲线上的点是否有效
            bezier_threshold = 5.0 / H
            # 生成固定y坐标的数组。其中ppl是采样点的数量，gap是y坐标之间的垂直间隔
            h_samples = np.array([1.0 - (ppl - i) * gap / H for i in range(ppl)], dtype=np.float32)
            # 使用quick_sample_point方法从贝塞尔曲线上快速采样点
            sampled_points = bezier_curve.quick_sample_point(image_size=None)
            temp = []
            # 对于每一个固定的y坐标，计算它与所有采样点y坐标的距离
            dis = np.abs(np.expand_dims(h_samples, -1) - sampled_points[:, 1])
            # 找到距离最小的采样点的索引
            idx = np.argmin(dis, axis=-1)
            # 根据阈值和边界条件，决定对应的x坐标。
            # 如果距离大于阈值或x坐标超出了边界（0到1之间），则将x坐标设置为-2。
            # 否则，将x坐标乘以图像宽度W，得到实际的x坐标
            for i in range(ppl):
                h = H - (ppl - i) * gap
                if dis[i][idx[i]] > bezier_threshold or sampled_points[idx[i]][0] > 1 or sampled_points[idx[i]][0] < 0:
                    temp.append([-2, h])
                else:
                    temp.append([sampled_points[idx[i]][0] * W, h])
            coordinates.append(temp)
        elif dataset in ['culane', 'llamas']:
            temp = bezier_curve.quick_sample_point(image_size=None)
            temp[:, 0] = temp[:, 0] * W
            temp[:, 1] = temp[:, 1] * H
            coordinates.append(temp.tolist())
        else:
            raise ValueError

    return coordinates