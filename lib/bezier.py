# import numpy as np
# import torch
#
#
# def bezier_curve(t, P):
#     """
#     计算贝塞尔曲线在参数t下的点
#     """
#     return (1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3]
#
#
# def bezier_curve_derivative(t, P):
#     """
#     贝塞尔曲线的导数
#     """
#     return -3 * (1 - t) ** 2 * P[0] + 3 * (1 - t) ** 2 * P[1] - 6 * t * (1 - t) * P[1] + 6 * t * (1 - t) * P[
#         2] - 3 * t ** 2 * P[2] + 3 * t ** 2 * P[3]
#
#
# def find_x_from_y(y_target, P, epsilon=1e-6, max_iterations=100):
#     """
#     在给定的y值下找到对应的x值
#     """
#     t = 0.5  # 初始值可以是0.5
#     for _ in range(max_iterations):
#         y = bezier_curve(t, P)[1]
#         if abs(y - y_target) < epsilon:
#             break
#         t -= (y - y_target) / bezier_curve_derivative(t, P)[1]
#
#     # 使用找到的t值计算对应的x值
#     x = bezier_curve(t, P)[0]
#
#     return x
#
# def cal_list(y_target, x_target, valid_xs, pred_polys):
#     x_target.t()
#     lanenums, pointnums = y_target.shape
#     start_idx = -1
#     end_idx = -1
#     res = []
#     device = y_target.device
#     # for i in range(lanenums):
#     #     print(x_target[i])
#     #     print(valid_xs[i])
#     for i in range(lanenums):
#         for j in range(pointnums - 1, -1, -1):
#             if valid_xs[i, j]:
#                 end_idx = j
#                 break
#         for j in range(pointnums):
#             if  valid_xs[i,j]:
#                 start_idx = j
#                 break
#         p0 = torch.cat([x_target[i, start_idx].expand(1), y_target[i, start_idx].expand(1)])
#         p1 = torch.cat([pred_polys[i, 0].expand(1),pred_polys[i, 1].expand(1)])
#         p2 = torch.cat([pred_polys[i, 2].expand(1), pred_polys[i, 3].expand(1)])
#         p3 = torch.cat([x_target[i, end_idx].expand(1),y_target[i,end_idx].expand(1)])
#         # tempx = [find_x_from_y(y,[p0,p1,p2,p3]) for y in y_target[i]]
#         tempx = torch.Tensor([find_x_from_y(y,[p0,p1,p2,p3]) for y in y_target[i]]).to(device)
#         # tempx = tempx.to(device)
#         res.append(tempx)
#     res = torch.stack(res, 0)
#     return res

