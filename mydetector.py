from utils.utils import *
from utils.datasets import *


def detector(im0, model, device="cpu", img_size=416, conf_thres=0.8, iou_thres=0.1, classes=None, agnostic_nms=False):

    img = letterbox(im0, new_shape=img_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres,
                                   multi_label=False, classes=classes, agnostic=agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                bbox_xywh1 = []
                bbox_xyxy1 = []
                cls_conf1 = []
                cls_id1 = []
                for *xyxy, conf, cls in det:
                    xyxyxy = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    xyxywh = [(xyxyxy[0] + xyxyxy[2]) / 2, (xyxyxy[1] + xyxyxy[3]) / 2, xyxyxy[2] - xyxyxy[0],
                              xyxyxy[3] - xyxyxy[1]]
                    # print(conf.item())
                    bbox_xyxy1.append(xyxyxy)
                    bbox_xywh1.append(xyxywh)
                    cls_conf1.append(conf.item())
                    cls_id1.append(int(cls))

                bbox_xyxy1 = np.array(bbox_xyxy1)
                bbox_xywh1 = np.array(bbox_xywh1)
                cls_conf1 = np.array(cls_conf1)
                cls_id1 = np.array(cls_id1)

                return bbox_xywh1, cls_conf1, cls_id1, bbox_xyxy1
            else:
                return None, 0, 0, None
