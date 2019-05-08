import logging

import torch

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker

logger = logging.getLogger(__name__)
MASK_SCALE = 28


class PlaneClustering(Masker):
    def __init__(self):
        super().__init__()
        assist_info = torch.empty((MASK_SCALE, MASK_SCALE, 3), dtype=torch.float32)
        assist_info[..., 0] = torch.arange(MASK_SCALE)[None, :]
        assist_info[..., 1] = torch.arange(MASK_SCALE)[:, None]
        assist_info[..., 2] = 1
        self.assist_info = assist_info

    def forward_single_image(self, masks, boxes):
        boxes = boxes.convert("xyxy")
        res = [
            self.reg_pyramid_in_image(mask[0], box)
            for mask, box in zip(masks, boxes.bbox)
        ]
        if len(res) > 0:
            res = torch.stack(res, dim=0)[:, None]
        else:
            res = masks.new_empty((0, 1, 4, 2))
        return res

    def reg_pyramid_in_image(self, mask, box):
        boundary = torch.as_tensor([[0, 0], [MASK_SCALE, 0], [MASK_SCALE, MASK_SCALE], [0, MASK_SCALE]],
                                   dtype=torch.float32)
        try:
            if torch.max(mask) <= 0.7:
                raise ValueError('No apex')
            src_points = torch.cat([self.assist_info, mask[..., None]], dim=2)
            pos_points = src_points[(mask > 0.1) & (mask < 0.8)]
            planes = plane_init(pos_points, ideal_value=1)
            planes = plane_clustering(pos_points, planes)
            points = get_intersection_of_plane(planes)
            if not is_clockwise(points):
                raise ValueError("Points is not clockwise")
            points = torch.clamp(points, - 0.5 * MASK_SCALE, 1.5 * MASK_SCALE)
        except ValueError as e:
            # logger.info(f'catch Exception: {e}')
            points = boundary

        w = float(box[2] - box[0] + 1)
        h = float(box[3] - box[1] + 1)
        points *= torch.as_tensor([w / MASK_SCALE, h / MASK_SCALE])
        points += box[:2]
        return points


def plane_init(pos_points, ideal_value):
    center = torch.mean(pos_points, dim=0)
    vector = torch.as_tensor([center[0], center[1], ideal_value]) - torch.as_tensor([
        [center[0], 0, 0], [MASK_SCALE, center[1], 0], [center[0], MASK_SCALE, 0], [0, center[1], 0]
    ])

    planes = torch.empty((3, 4), dtype=torch.float32)
    planes[:2] = -ideal_value * vector[:, :2].t()
    param_C = vector[:, 0] ** 2 + vector[:, 1] ** 2
    if not torch.all(param_C > 0.1):
        raise ValueError('No enough points for some planes')
    planes[:2] /= param_C
    planes[2] = - (ideal_value + torch.matmul(center[:2], planes[:2]))
    return planes


def plane_clustering(pos_points, planes, iter_num=10):
    ones = torch.ones((1, 4), dtype=torch.float32)
    for iter in range(iter_num):
        ans = torch.abs(torch.matmul(pos_points, torch.cat([planes, ones])))
        partition = torch.argmin(ans, dim=1)
        point_groups = [pos_points[partition == i] for i in range(planes.shape[1])]
        for i, group in enumerate(point_groups):
            if len(group) == 0:
                continue
            X = group[:, :3]
            B = -group[:, 3:]
            A = torch.gels(B, X)[0][:3]
            abs_residuals = torch.abs(torch.matmul(X, A) - B)
            abs_residual_scale = torch.median(abs_residuals)
            if abs_residual_scale > 1e-4:
                X_weight = abs_residuals / (6.9460 * abs_residual_scale)
                X_weight[X_weight > 1] = 0
                X_weight[X_weight <= 1] = (1 - X_weight[X_weight <= 1] ** 2) ** 2
                X_weighted = X_weight * X
                X = torch.matmul(X_weighted.t(), X)
                B = torch.matmul(X_weighted.t(), B)
                A = torch.gels(B, X)[0]
            planes[:, i] = A.flatten()
    return planes


def get_intersection_of_plane(normal_vectors):
    """

    :param normal_vectors: [M = 4, {A, B, D}]
    :return:
    """
    points = torch.empty((4, 2), dtype=torch.float32)
    for i in range(4):
        param = normal_vectors[:, [i, (i + 1) % 4]]
        coefficient = param[:2].t()
        ordinate = -param[2]
        points[i] = torch.gels(ordinate, coefficient)[0].squeeze()
    return points


def is_clockwise(corners):
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1] - corners[j][0] * corners[i][1]
    return area > 0
