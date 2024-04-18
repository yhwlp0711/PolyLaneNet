import numpy as np
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional

from lib.BezierCurve import BezierCurve
from lib.basecode import lane_pruning, bezier_to_coordinates


class BezierLaneNet(torch.nn.Module):
    # Curve regression network, similar design as simple object detection (e.g. FCOS)
    def __init__(self,
                 backbone_cfg,
                 reducer_cfg,
                 dilated_blocks_cfg,
                 feature_fusion_cfg,
                 head_cfg,
                 aux_seg_head_cfg,
                 image_height=360,
                 num_regression_parameters=8,
                 thresh=0.5,
                 local_maximum_window_size=9):
        super(BezierLaneNet, self).__init__()
        global_stride = 16
        branch_channels = 256

        self.thresh = thresh
        self.local_maximum_window_size = local_maximum_window_size

        self.backbone = MODELS.from_dict(backbone_cfg)
        # self.backbone = EfficientNet.from_pretrained("efficientnet-b0")
        self.dilated_blocks = MODELS.from_dict(dilated_blocks_cfg)
        self.simple_flip_2d = MODELS.from_dict(feature_fusion_cfg)  # Name kept for legacy weights
        self.aggregator = nn.AvgPool2d(kernel_size=((image_height - 1) // global_stride + 1, 1), stride=1, padding=0)
        self.regression_head = MODELS.from_dict(head_cfg)  # Name kept for legacy weights
        self.proj_classification = nn.Conv1d(branch_channels, 1, kernel_size=1, bias=True, padding=0)
        self.proj_regression = nn.Conv1d(branch_channels, num_regression_parameters,
                                         kernel_size=1, bias=True, padding=0)
        self.segmentation_head = MODELS.from_dict(aux_seg_head_cfg)

    @torch.no_grad()
    # 模型推理：根据forward参数决定是否进行前向传播。如果forward为True，则调用self.forward(inputs)进行前向传播。
    # 计算存在概率：使用模型的输出outputs计算存在概率existence_conf，并根据阈值self.thresh确定每条车道线是否存在。
    # 局部最大值检测：如果self.local_maximum_window_size > 0，则进行局部最大值检测，确保每个局部最大值点都是车道线的开始或结束点。
    # 控制点处理：获取模型输出中的控制点control_points，并根据max_lane参数进行车道线数量的修剪。
    # 返回控制点：如果return_cps为True，则计算并返回调整后的控制点cps。
    # 贝塞尔曲线处理：根据dataset选择贝塞尔曲线的采样数量。然后，对每条车道线的控制点和存在标志调用bezier_to_coordinates方法，得到车道线的坐标。
    # 返回结果：返回车道线的坐标
    def inference(self, inputs, input_sizes, gap, ppl, dataset, max_lane=0, forward=True, return_cps=False, n=50):
        # 前向传播
        outputs = self.forward(inputs) if forward else inputs  # Support no forwarding inside this function
        # 计算车道线存在的概率
        existence_conf = outputs['logits'].sigmoid()
        # 确定是否存在 即得到的概率是否大于阈值
        existence = existence_conf > self.thresh

        # Test local maxima
        # 是否检测局部最大值
        if self.local_maximum_window_size > 0:
            # existence_conf.unsqueeze(1)：将existence_conf的形状从[B, N]变为[B, 1, N]。
            # kernel_size=self.local_maximum_window_size：设置池化窗口的大小。
            # stride=1：设置池化的步长。
            # padding=(self.local_maximum_window_size - 1) // 2：设置填充大小，确保池化后的输出与输入大小相同。
            # return_indices=True：返回池化操作后的最大值索引
            _, max_indices = torch.nn.functional.max_pool1d(existence_conf.unsqueeze(1),
                                                            kernel_size=self.local_maximum_window_size, stride=1,
                                                            padding=(self.local_maximum_window_size - 1) // 2,
                                                            return_indices=True)
            # 保存每个池化窗口内的最大值索引
            max_indices = max_indices.squeeze(1)  # B x Q
            # 生成一个索引数组，与max_indices具有相同的形状
            indices = torch.arange(0, existence_conf.shape[1],
                                   dtype=existence_conf.dtype,
                                   device=existence_conf.device).unsqueeze(0).expand_as(max_indices)
            # 将max_indices与indices进行比较，得到一个布尔数组local_maxima，表示哪些点是局部最大值
            local_maxima = max_indices == indices
            # 使用布尔数组local_maxima更新existence，将不是局部最大值的点的存在性置为0
            existence *= local_maxima

        # 控制点
        control_points = outputs['curves']
        # 如果max_lane不为0，代码会调用lane_pruning函数对车道线的存在性和存在概率进行修剪，
        # 以确保模型检测到的车道线数量不超过max_lane指定的最大值
        if max_lane != 0:  # Lane max number prior for testing
            existence, _ = lane_pruning(existence, existence_conf, max_lane=max_lane)

        # 如果return_cps为True，这段代码会计算并返回调整后的控制点cps。
        # 调整包括将控制点从相对于输入尺寸的比例转换为实际的坐标，并根据existence数组筛选出存在的控制点
        if return_cps:
            image_size = torch.tensor([input_sizes[1][1], input_sizes[1][0]],
                                      dtype=torch.float32, device=control_points.device)
            cps = control_points * image_size
            cps = [cps[i][existence[i]].cpu().numpy() for i in range(existence.shape[0])]

        # 进行了数据类型和设备的转换，以及获取输入图像的尺寸
        existence = existence.cpu().numpy()
        control_points = control_points.cpu().numpy()
        H, _ = input_sizes[1]
        # 阶数为3，num_sample_points 是要在曲线上采样的点的数量
        b = BezierCurve(order=3, num_sample_points=H if dataset == 'tusimple' else n)

        lane_coordinates = []
        for j in range(existence.shape[0]):
            lane_coordinates.append(bezier_to_coordinates(control_points=control_points[j], existence=existence[j],
                                                               resize_shape=input_sizes[1], dataset=dataset,
                                                               bezier_curve=b, gap=gap, ppl=ppl))
        if return_cps:
            return cps, lane_coordinates
        else:
            return lane_coordinates

    def forward(self, x):
        # Return shape: B x Q, B x Q x N x 2
        x = self.backbone(x)
        if isinstance(x, dict):
            x = x['out']

        if self.reducer is not None:
            x = self.reducer(x)

        # Segmentation task
        if self.segmentation_head is not None:
            segmentations = self.segmentation_head(x)
        else:
            segmentations = None

        if self.dilated_blocks is not None:
            x = self.dilated_blocks(x)

        with autocast(False):  # TODO: Support fp16 like mmcv
            x = self.simple_flip_2d(x.float())
        x = self.aggregator(x)[:, :, 0, :]

        x = self.regression_head(x)
        logits = self.proj_classification(x).squeeze(1)
        curves = self.proj_regression(x)

        return {'logits': logits,
                'curves': curves.permute(0, 2, 1).reshape(curves.shape[0], -1, curves.shape[-2] // 2, 2).contiguous(),
                'segmentations': segmentations}

    def eval(self, profiling=False):
        super().eval()
        if profiling:
            self.segmentation_head = None