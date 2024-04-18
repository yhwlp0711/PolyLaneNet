import numpy as np
import torch
from scipy.interpolate import splprep, splev
from scipy.special import comb as n_over_k
from basecode import upcast

class BezierCurve(object):
    # Define Bezier curves for curve fitting
    def __init__(self, order, num_sample_points=50):
        # self.num_point：贝塞尔曲线的控制点数量，它等于 order + 1。
        # self.control_points：用于存储贝塞尔曲线的控制点，初始为空列表。
        # self.bezier_coeff：贝塞尔曲线的贝塞尔系数，通过 self.get_bezier_coefficient() 方法获取。
        # self.num_sample_points：贝塞尔曲线的采样点数量，默认为50。
        # self.c_matrix：伯恩斯坦矩阵（Bernstein Matrix），通过 self.get_bernstein_matrix() 方法获取
        self.num_point = order + 1
        self.control_points = []
        # self.bezier_coeff = self.get_bezier_coefficient()
        self.num_sample_points = num_sample_points
        # 贝塞尔系数的矩阵形式
        self.c_matrix = self.get_bernstein_matrix()

    def bezier_coeff(self, ts):
        # 用于计算贝塞尔系数的每一项。它接受三个参数：n（曲线阶数），t（参数），k（索引）
        def Mtk(n, t, k):
            return t ** k * (1 - t) ** (n - k) * n_over_k(n, k)

        # 接受一个参数 ts，它是一个参数 t 的列表或数组。这个函数使用 Mtk 函数计算每个 t 对应的贝塞尔系数
        return [[Mtk(self.num_point - 1, t, k) for k in range(self.num_point)] for t in ts]

    def interpolate_lane(self, x, y, n=50):
        # Spline interpolation of a lane. Used on the predictions
        # 对一个车道进行样条插值（Spline Interpolation）。它将输入的 x 和 y 坐标用样条曲线进行插值，从而得到更平滑的曲线
        assert len(x) == len(y)

        # tck, _ = splprep([x, y], s=0, t=n, k=min(3, len(x) - 1))
        tck = splprep([x, y], s=0, t=n, k=min(3, len(x) - 1))

        u = np.linspace(0., 1., n)
        return np.array(splev(u, tck)).T

    def get_control_points(self, x, y, interpolate=False):
        # 是否使用插值方法
        if interpolate:
            # 使用 interpolate_lane 方法对 x 和 y 进行插值，得到平滑的车道坐标
            points = self.interpolate_lane(x, y)
            x = np.array([x for x, _ in points])
            y = np.array([y for _, y in points])

        # 将每两个连续的中间控制点作为一个控制点对添加到 self.control_points 列表中
        middle_points = self.get_middle_control_points(x, y)
        for idx in range(0, len(middle_points) - 1, 2):
            self.control_points.append([middle_points[idx], middle_points[idx + 1]])

    def get_bernstein_matrix(self):
        # 用于计算伯恩斯坦矩阵（Bernstein Matrix），它是贝塞尔曲线上一系列点的贝塞尔系数的矩阵形式
        tokens = np.linspace(0, 1, self.num_sample_points)
        c_matrix = self.bezier_coeff(tokens)
        return np.array(c_matrix)

    def save_control_points(self):
        return self.control_points

    def assign_control_points(self, control_points):
        self.control_points = control_points

    def quick_sample_point(self, image_size=None):
        # 快速采样贝塞尔曲线上的点。
        # 它使用之前计算的伯恩斯坦矩阵 self.c_matrix 和控制点矩阵来计算贝塞尔曲线上的采样点
        control_points_matrix = np.array(self.control_points)
        sample_points = self.c_matrix.dot(control_points_matrix)
        if image_size is not None:
            sample_points[:, 0] = sample_points[:, 0] * image_size[-1]
            sample_points[:, -1] = sample_points[:, -1] * image_size[0]
        return sample_points

    def get_sample_point(self, n=50, image_size=None):
        """
            :param n: the number of sampled points
            :return: a list of sampled points
        """
        # 不同方法进行采样
        # 与 quick_sample_point 方法相比，get_sample_point 方法更为直接地使用了贝塞尔系数和控制点来计算采样点
        t = np.linspace(0, 1, n)
        coeff_matrix = np.array(self.bezier_coeff(t))
        control_points_matrix = np.array(self.control_points)
        sample_points = coeff_matrix.dot(control_points_matrix)
        if image_size is not None:
            sample_points[:, 0] = sample_points[:, 0] * image_size[-1]
            sample_points[:, -1] = sample_points[:, -1] * image_size[0]

        return sample_points

    def get_middle_control_points(self, x, y):
        # 这个 get_middle_control_points 方法用于计算贝塞尔曲线中间的控制点。
        # 它使用了输入的 x 和 y 坐标来计算中间控制点，以便更好地逼近输入的坐标点
        dy = y[1:] - y[:-1]
        dx = x[1:] - x[:-1]
        dt = (dx ** 2 + dy ** 2) ** 0.5
        t = dt / dt.sum()
        t = np.hstack(([0], t))
        t = t.cumsum()
        data = np.column_stack((x, y))
        Pseudoinverse = np.linalg.pinv(self.bezier_coeff(t))  # (9,4) -> (4,9)
        control_points = Pseudoinverse.dot(data)  # (4,9)*(9,2) -> (4,2)
        medi_ctp = control_points[:, :].flatten().tolist()

        return medi_ctp


class BezierSampler(torch.nn.Module):
    # Fast Batch Bezier sampler
    # 用于批量采样贝塞尔曲线上点的PyTorch模块。它使用了贝塞尔系数和伯恩斯坦矩阵来快速计算贝塞尔曲线上的采样点
    def __init__(self, order, num_sample_points, proj_coefficient=0):
        super().__init__()
        # order：贝塞尔曲线的阶数。
        # num_sample_points：要采样的点的数量。
        # proj_coefficient：投影系数，用于修改采样点的分布。
        self.proj_coefficient = proj_coefficient
        self.num_control_points = order + 1
        self.num_sample_points = num_sample_points
        self.control_points = []
        # self.bezier_coeff = self.get_bezier_coefficient()
        self.bernstein_matrix = self.get_bernstein_matrix()

    def Mtk(self, n, t, k):
        return t ** k * (1 - t) ** (n - k) * n_over_k(n, k)

    def bezier_coeff(self, ts):
        return [[self.Mtk(self.num_control_points - 1, t, k) for k in range(self.num_control_points)] for t
                in ts]

    def get_bernstein_matrix(self):
        t = torch.linspace(0, 1, self.num_sample_points)
        if self.proj_coefficient != 0:
            # tokens = tokens + (1 - tokens) * tokens ** self.proj_coefficient
            t[t > 0.5] = t[t > 0.5] + (1 - t[t > 0.5]) * t[t > 0.5] ** self.proj_coefficient
            t[t < 0.5] = 1 - (1 - t[t < 0.5] + t[t < 0.5] * (1 - t[t < 0.5]) ** self.proj_coefficient)
        c_matrix = torch.tensor(self.bezier_coeff(t))
        return c_matrix

    def get_sample_points(self, control_points_matrix):
        if control_points_matrix.numel() == 0:
            return control_points_matrix  # Looks better than a torch.Tensor
        if self.bernstein_matrix.device != control_points_matrix.device:
            self.bernstein_matrix = self.bernstein_matrix.to(control_points_matrix.device)

        return upcast(self.bernstein_matrix).matmul(upcast(control_points_matrix))
