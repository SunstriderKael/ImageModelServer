from ModelHelper.Detection.DetectionUtils.Quadrangle import compute_iou, txt2list
from ModelHelper.Common.CommonUtils.HandleText import get_txt_list
import os


def f1score_single(label_txt, pred_txt, threshold):
    label_list = txt2list(label_txt)
    pred_list = txt2list(pred_txt)

    pred_num = len(pred_list)
    valid_label_num = 0
    correct_num = 0

    for label in label_list:
        if label.area > 10:
            valid_label_num += 1
            max_iou = 0
            pred_idx = -1
            idx = 0
            for pred in pred_list:
                iou = compute_iou(label, pred)
                if iou > max_iou:
                    max_iou = iou
                    pred_idx = idx
                idx += 1

            if max_iou > threshold:
                correct_num += 1
                pred_list.pop(pred_idx)
    if valid_label_num == 0 or correct_num == 0 or pred_num == 0:
        score = 0
    else:
        precision = correct_num / valid_label_num
        recall = correct_num / pred_num
        score = 2 * precision * recall / (precision + recall)
    return score, valid_label_num, pred_num, correct_num


def f1score(label_folder, pred_folder, threshold):
    txt_list = get_txt_list(label_folder)
    total_pred_num = 0
    total_label_num = 0
    total_correct_num = 0
    for txt in txt_list:
        label_txt = os.path.join(label_folder, txt)
        pred_txt = os.path.join(pred_folder, txt)
        assert os.path.exists(pred_txt)
        _, label_num, pred_num, correct_num = f1score_single(label_txt, pred_txt, threshold)
        total_label_num += label_num
        total_pred_num += pred_num
        total_correct_num += correct_num
        while total_pred_num > 10 * total_label_num and total_label_num != 0:
            print('total_pred_num > 10*total_label_num!')
            return 0, 0, 0, 0

    if total_label_num == 0 or total_pred_num == 0 or total_correct_num == 0:
        return 0, total_label_num, total_pred_num, total_correct_num
    else:
        precision = total_correct_num / total_label_num
        recall = total_correct_num / total_pred_num
        score = 2 * precision * recall / (precision + recall)
        return score, total_label_num, total_pred_num, total_correct_num
