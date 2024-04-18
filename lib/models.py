import torch
import torch.nn as nn
from torchvision.models import resnet34, resnet50, resnet101
from efficientnet_pytorch import EfficientNet



class OutputLayer(nn.Module):
    def __init__(self, fc, num_extra):
        super(OutputLayer, self).__init__()
        # 全连接层
        self.regular_outputs_layer = fc
        self.num_extra = num_extra
        if num_extra > 0:
            # 创建一个额外全连接层
            # 输入 fc.in_features
            # 输出 num_exra
            self.extra_outputs_layer = nn.Linear(fc.in_features, num_extra)

    def forward(self, x):
        # 常规结果
        regular_outputs = self.regular_outputs_layer(x)
        # 额外结果
        if self.num_extra > 0:
            extra_outputs = self.extra_outputs_layer(x)
        else:
            extra_outputs = None

        # 返回 常规结果、额外结果
        return regular_outputs, extra_outputs


class PolyRegression(nn.Module):
    def __init__(self,
                 num_outputs,
                 backbone,
                 pretrained,
                 curriculum_steps=None,
                 extra_outputs=0,
                 share_top_y=True,
                 pred_category=False):
        # num_outputs: 输出的数量。
        # backbone: 使用的主干模型。
        # pretrained: 是否使用预训练的主干模型。
        # curriculum_steps: 训练中使用的课程步骤（如果有）。
        # extra_outputs: 额外输出的数量。
        # share_top_y: 是否共享顶部 Y 坐标。
        # pred_category: 是否进行类别预测。
        super(PolyRegression, self).__init__()
        if 'efficientnet' in backbone:
            if pretrained:
                self.model = EfficientNet.from_pretrained(backbone, num_classes=num_outputs)
            else:
                self.model = EfficientNet.from_name(backbone, override_params={'num_classes': num_outputs})
            # 自定义全连接层
            self.model._fc = OutputLayer(self.model._fc, extra_outputs)
        # 根据不同的 ResNet 模型设置不同的预训练权重
        # 为了适应新的输出要求而修改最后一层全连接层
        # 替换全连接层为一个新的具有相同输入特征维度但输出特征维度为 num_outputs 的线性层，
        # 并通过 OutputLayer 添加额外的输出
        elif backbone == 'resnet34':
            self.model = resnet34(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_outputs)
            self.model.fc = OutputLayer(self.model.fc, extra_outputs)
        elif backbone == 'resnet50':
            self.model = resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_outputs)
            self.model.fc = OutputLayer(self.model.fc, extra_outputs)
        elif backbone == 'resnet101':
            self.model = resnet101(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_outputs)
            self.model.fc = OutputLayer(self.model.fc, extra_outputs)
        else:
            raise NotImplementedError()

        self.curriculum_steps = [0, 0, 0, 0] if curriculum_steps is None else curriculum_steps
        self.share_top_y = share_top_y
        self.extra_outputs = extra_outputs
        self.pred_category = pred_category
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, epoch=None, **kwargs):
        # 调用前向传播
        output, extra_outputs = self.model(x, **kwargs)
        # 让模型在训练的早期阶段集中精力学习较简单的任务或特征
        # 然后逐步引入更复杂的任务或特征
        for i in range(len(self.curriculum_steps)):
            if epoch is not None and epoch < self.curriculum_steps[i]:
                output[:, -len(self.curriculum_steps) + i] = 0
        return output, extra_outputs

    def decode(self, all_outputs, labels, conf_threshold=0.5):
        # 解码函数 解析模型输出得到结果
        outputs, extra_outputs = all_outputs
        if extra_outputs is not None:
            # ？
            extra_outputs = extra_outputs.reshape(labels.shape[0], 5, -1)
            # 沿着第三个维度计算 找到最大值的索引
            # (x,y,z) 返回 (x,y)
            extra_outputs = extra_outputs.argmax(dim=2)
        # 置信度、上界、下界、系数
        outputs = outputs.reshape(len(outputs), -1, 7)  # score + upper + lower + 4 coeffs = 7
        # 对置信度进行 Sigmoid 处理
        # 过滤低置信度的结果
        outputs[:, :, 0] = self.sigmoid(outputs[:, :, 0])
        outputs[outputs[:, :, 0] < conf_threshold] = 0

        if False and self.share_top_y:
            outputs[:, :, 0] = outputs[:, 0, 0].expand(outputs.shape[0], outputs.shape[1])

        return outputs, extra_outputs

    def loss(self,
             outputs, # batchsize * 35 (lanenums=5 * (1 + 2 + 4))
             target, # batchsize * lanenums=5 * 115 每条车道115个点？
             conf_weight=1,
             lower_weight=1,
             upper_weight=1,
             cls_weight=1,
             poly_weight=300,
             threshold=15 / 720.):
        # outputs: 模型的输出，包括车道线的预测结果和额外输出（如果有的话）。
        # target: 真实的标签，包括车道线的真实位置和其他信息。
        # conf_weight: 置信度损失的权重。
        # lower_weight: 下界位置损失的权重。
        # upper_weight: 上界位置损失的权重。
        # cls_weight: 类别损失的权重。
        # poly_weight: 多项式系数损失的权重。
        # threshold: 用于控制车道线预测的阈值，低于这个阈值的预测会被忽略。
        pred, extra_outputs = outputs
        # 二元交叉熵损失
        bce = nn.BCELoss()
        # 均方误差损失
        mse = nn.MSELoss()
        s = nn.Sigmoid()
        # 阈值处理函数
        # 小于置 0 大于不变
        threshold = nn.Threshold(threshold**2, 0.)
        # 获取预测结果
        pred = pred.reshape(-1, target.shape[1], 1 + 2 + 4) # batchsize * lanenums=5 * 7
        # 提取各结果
        # batchsize * lanenums=5 再 reshape (batchsize * lanenums=5) * 1 每条车道线一个数据
        target_categories, pred_confs = target[:, :, 0].reshape((-1, 1)), s(pred[:, :, 0]).reshape((-1, 1))
        target_uppers, pred_uppers = target[:, :, 2].reshape((-1, 1)), pred[:, :, 2].reshape((-1, 1))
        # 前者为 (batchsize * lanenums) * pointnums 后者为 (batchsize * lanenums) * 7
        target_points, pred_polys = target[:, :, 3:].reshape((-1, target.shape[2] - 3)), pred[:, :, 3:].reshape(-1, 4)
        target_lowers, pred_lowers = target[:, :, 1], pred[:, :, 1]

        if self.share_top_y:
            # inexistent lanes have -1e-5 as lower
            # i'm just setting it to a high value here so that the .min below works fine
            # 小于 0 则置为 1
            target_lowers[target_lowers < 0] = 1
            # 每行的最小值复制到每行的所有元素中
            target_lowers[...] = target_lowers.min(dim=1, keepdim=True)[0]
            # pred_lowers[:, 0] 选择了 pred_lowers 的第一列
            # .reshape(-1, 1) 将选定的列重塑为单列向量
            # .expand(pred.shape[0], pred.shape[1]) 扩展该向量，使其具有与 pred 相同的行数和列数
            # pred_lowers[...] 是对整个数组的切片引用
            # 将 pred_lowers 的第一列元素复制到每行的所有元素中
            pred_lowers[...] = pred_lowers[:, 0].reshape(-1, 1).expand(pred.shape[0], pred.shape[1])

        target_lowers = target_lowers.reshape((-1, 1))
        pred_lowers = pred_lowers.reshape((-1, 1))

        # 每个元素等于对应车道的可信度
        target_confs = (target_categories > 0).float()
        # 筛选出具有可信度车道的索引 即有效车道
        valid_lanes_idx = target_confs == 1
        # 二维->一维
        valid_lanes_idx_flat = valid_lanes_idx.reshape(-1)
        # 上界损失、下界损失 标量
        lower_loss = mse(target_lowers[valid_lanes_idx], pred_lowers[valid_lanes_idx])
        upper_loss = mse(target_uppers[valid_lanes_idx], pred_uppers[valid_lanes_idx])

        # classification loss
        if self.pred_category and self.extra_outputs > 0: # pred_category: default = false
            # 交叉熵损失
            ce = nn.CrossEntropyLoss()
            pred_categories = extra_outputs.reshape(target.shape[0] * target.shape[1], -1)
            target_categories = target_categories.reshape(pred_categories.shape[:-1]).long()
            # 筛选
            pred_categories = pred_categories[target_categories > 0]
            target_categories = target_categories[target_categories > 0]
            cls_loss = ce(pred_categories, target_categories - 1)
        else:
            cls_loss = 0

        # poly loss calc
        # 提取 x
        # valid_lanes_idx_flat 为 bool 掩码  有效车道的个数(从 batchsize * lanenums 中筛选) * 有效点的数量
        target_xs = target_points[valid_lanes_idx_flat, :target_points.shape[1] // 2]
        # 提取 y 再转置
        # 多项式 ys
        ys = target_points[valid_lanes_idx_flat, target_points.shape[1] // 2:].t()
        # 贝塞尔 ys
        # ys = target_points[valid_lanes_idx_flat, target_points.shape[1] // 2:]
        # bool 掩码  再计算 x > 0 的掩码
        valid_xs = target_xs >= 0
        # 筛选有效值 pred_polys (batchsize * lanenums) * 4
        pred_polys = pred_polys[valid_lanes_idx_flat]


        # 根据 ys 和多项式系数计算 xs 并转置 计算预测 x
        # 改曲线改这里  pred_x = cal(pred_polys, y)
        pred_xs = pred_polys[:, 0] * ys**3 + pred_polys[:, 1] * ys**2 + pred_polys[:, 2] * ys + pred_polys[:, 3]
        pred_xs.t_()
        # 贝塞尔曲线
        # pred_xs = cal_list(ys, target_xs, valid_xs, pred_polys)

        # 计算权重 有效 xs 求和得到一个标量 X
        # 有效 xs 按行求和得到一个 tensor
        # 使用 X 逐一除以 tensor 得到新的 tensor
        # 开方
        weights = (torch.sum(valid_xs, dtype=torch.float32) / torch.sum(valid_xs, dim=1, dtype=torch.float32))**0.5
        # 加权处理
        # without this, lanes with more points would have more weight on the cost function
        # 如果没有这个步骤，拥有更多点的车道线会在损失函数中拥有更大的权重
        # 确保每条车道线对损失函数的贡献大致相等，而不会被其点的数量所主导
        pred_xs = (pred_xs.t_() * weights).t()
        target_xs = (target_xs.t_() * weights).t()

        # poly_loss = mse(pred_xs[valid_xs], target_xs[valid_xs]) / valid_lanes_idx.sum()

        # 类似均方误差 计算后根据阈值进行处理
        # (valid_lanes_idx.sum() * valid_xs.sum()) 为有效车道数 * 有效车道线对的点的总数量
        poly_loss = threshold(
            (pred_xs[valid_xs] - target_xs[valid_xs])**2).sum() / (valid_lanes_idx.sum() * valid_xs.sum())

        # applying weights to partial losses
        # 加权
        poly_loss = poly_loss * poly_weight
        lower_loss = lower_loss * lower_weight
        upper_loss = upper_loss * upper_weight
        cls_loss = cls_loss * cls_weight
        # 计算置信度损失（二元交叉熵损失）再加权
        conf_loss = bce(pred_confs, target_confs) * conf_weight

        # loss
        loss = conf_loss + lower_loss + upper_loss + poly_loss + cls_loss

        # 返回 loss 和 各部分损失的字典
        return loss, {
            'conf': conf_loss,
            'lower': lower_loss,
            'upper': upper_loss,
            'poly': poly_loss,
            'cls_loss': cls_loss
        }
