import torch.nn.functional
from sklearn.metrics import precision_score, recall_score
from torchmetrics import AveragePrecision, ConfusionMatrix


def batch_pad_collate(batch):
    raw = [item[0] for item in batch]
    label = [item[1] for item in batch]
    max_height = max([item.shape[-2] for item in raw])
    max_width = max([item.shape[-1] for item in raw])
    if max_width > max_height:
        max_dim = max_width
    else:
        max_dim = max_height
    if max_dim % 2 > 0:
        max_dim += 1

    for i, image in enumerate(raw):
        padding_top, padding_right, padding_bottom, padding_left = 0, 0, 0, 0
        if (width_diff := max_dim - image.shape[-1]) > 0:
            padding_left = width_diff // 2
            padding_right = width_diff - padding_left
        if (height_diff := max_dim - image.shape[-2]) > 0:
            padding_top = height_diff // 2
            padding_bottom = height_diff - padding_top
        if any([padding_top, padding_right, padding_bottom, padding_left]):
            raw[i] = torch.nn.functional.pad(image, (padding_left, padding_right, padding_top, padding_bottom))

    for i, image in enumerate(label):
        padding_top, padding_right, padding_bottom, padding_left = 0, 0, 0, 0
        if (width_diff := max_dim - image.shape[-1]) > 0:
            padding_left = width_diff // 2
            padding_right = width_diff - padding_left
        if (height_diff := max_dim - image.shape[-2]) > 0:
            padding_top = height_diff // 2
            padding_bottom = height_diff - padding_top
        if any([padding_top, padding_right, padding_bottom, padding_left]):
            label[i] = torch.nn.functional.pad(image, (padding_left, padding_right, padding_top, padding_bottom))

    return [torch.stack(raw, dim=1), torch.stack(label, dim=1)]


def batch_pad_collate_v2(batch):
    raw = [item[0] for item in batch]
    label = [item[1] for item in batch]
    max_height = max([item[list(item.keys())[0]].shape[-2] for item in raw])
    max_width = max([item[list(item.keys())[0]].shape[-1] for item in raw])
    # if max_width > max_height:
    #     max_dim = max_width
    # else:
    #     max_dim = max_height
    if max_width % 2 > 0:
        max_width += 1
    if max_height % 2 > 0:
        max_height += 1

    raw_labels = []
    out_labels = []

    for i, image_dict in enumerate(raw):
        image = image_dict[list(image_dict.keys())[0]]
        padding_top, padding_right, padding_bottom, padding_left = 0, 0, 0, 0
        if (width_diff := max_width - image.shape[-1]) > 0:
            padding_left = width_diff // 2
            padding_right = width_diff - padding_left
        if (height_diff := max_height - image.shape[-2]) > 0:
            padding_top = height_diff // 2
            padding_bottom = height_diff - padding_top

        input_labels = []
        mean = image_dict['mean']
        std = image_dict['std']
        for key in image_dict:
            if key == 'mean' or key == 'std':
                continue
            if any([padding_top, padding_right, padding_bottom, padding_left]):
                if mean != 0 or std != 0:
                    not_normalized = (image_dict[key] * std) + mean
                else:
                    not_normalized = image_dict[key]
                not_normalized = torch.nn.functional.pad(not_normalized,
                                                         (padding_left, padding_right, padding_top, padding_bottom))
                # if mean != 0 or std != 0:
                #     normalized = (not_normalized - not_normalized.mean()) / not_normalized.std()
                #     input_labels.append(normalized)
                # else:
                input_labels.append(not_normalized)

            else:
                input_labels.append(image_dict[key])
        input_labels = torch.stack(input_labels, dim=0)
        raw_labels.append(input_labels)

        output_labels = []
        for key in label[i]:
            value = 0
            if key == 'background_label':
                value = 1
            if any([padding_top, padding_right, padding_bottom, padding_left]):
                output_labels.append(torch.nn.functional.pad(label[i][key],
                                                             (padding_left, padding_right, padding_top, padding_bottom),
                                                             value=value))
            else:
                output_labels.append(label[i][key])
        output_labels = torch.stack(output_labels, dim=0)
        out_labels.append(output_labels)

    return [torch.stack(raw_labels, dim=1), torch.stack(out_labels, dim=1)]


def simple_collate_fn(batch):
    raw = [item[0] for item in batch]
    label = [item[1] for item in batch]

    return [raw, label]


def calculate_accuracy(output, label, threshold):  # do TP + TN / all_samples
    output = (output > threshold).float() * 1
    per_batch_acc = []
    for batch in range(label.shape[0]):
        per_class_acc = []
        for num_class in range(label.shape[3]):
            one_label = label[batch, :, :, num_class]
            one_output = output[batch, :, :, num_class]
            true_positives = torch.logical_and(one_label, one_output)
            true_negatives = torch.logical_and(torch.logical_not(one_label), torch.logical_not(one_output))
            per_class_acc.append((true_positives.sum() + true_negatives.sum()) / one_output.numel())
        mean_acc = torch.mean(torch.tensor(per_class_acc))
        per_batch_acc.append(mean_acc)
    return torch.mean(torch.tensor(per_batch_acc))


def calculate_accuracy_3d(output, label, threshold):  # do TP + TN / all_samples
    output = (output > threshold).float() * 1
    per_batch_acc = []
    for batch in range(label.shape[0]):
        per_slice_acc = []
        for num_slices in range(label.shape[1]):
            one_label = label[batch, num_slices]
            one_output = output[batch, num_slices]
            true_positives = torch.logical_and(one_label, one_output)
            true_negatives = torch.logical_and(torch.logical_not(one_label), torch.logical_not(one_output))
            per_slice_acc.append((true_positives.sum() + true_negatives.sum()) / one_output.numel())
        mean_acc = torch.mean(torch.tensor(per_slice_acc))
        per_batch_acc.append(mean_acc)
    return torch.mean(torch.tensor(per_batch_acc))


def calculate_iou(output, label, threshold):  # also called jaccard index
    output = (output > threshold).float() * 1
    per_batch_iou = []
    for batch in range(label.shape[0]):
        per_class_iou = []
        for num_class in range(label.shape[3]):
            one_label = label[batch, :, :, num_class]
            one_output = output[batch, :, :, num_class]
            intersection = torch.logical_and(one_label, one_output)
            union = torch.logical_or(one_label, one_output)
            union = torch.sum(union)
            # if union != 0.:
            per_class_iou.append(torch.sum(intersection) / union if union != 0 else 1e-6)
        mean_iou = torch.mean(torch.tensor(per_class_iou))  # .nan_to_num(nan=0.)
        per_batch_iou.append(mean_iou)
    return torch.mean(torch.tensor(per_batch_iou))  # .nan_to_num(nan=0.)


def calculate_iou_3d(output, label, threshold):  # also called jaccard index
    output = (output > threshold).float() * 1
    per_batch_iou = []
    for batch in range(label.shape[0]):
        one_label = torch.flatten(label[batch])
        one_output = torch.flatten(output[batch])
        intersection = torch.logical_and(one_label, one_output)
        union = torch.logical_or(one_label, one_output)
        union = torch.sum(union)
        union = union if union != 0 else 1e-6
        # if union != 0.:
        per_batch_iou.append(torch.sum(intersection) / union)
    return torch.mean(torch.tensor(per_batch_iou))  # .nan_to_num(nan=0.)


def calculate_metric(output, label, metric):
    val = metric(torch.flatten(output, start_dim=1), torch.flatten(label, start_dim=1).byte())
    # val = metric(output, label.long())
    metric.reset()
    return val


def compute_confusion_matrix(output, label, threshold):  # use this in a loop
    label = torch.flatten(label).byte()
    output = (torch.flatten(output) > threshold).byte()
    confusion_matrix = ConfusionMatrix(num_classes=2).to("cuda")
    val = confusion_matrix(output, label)
    confusion_matrix.reset()
    return val


def compute_precision_recall_f1(output, label, threshold):  # this is a micro style computation
    precisions = []
    recalls = []
    f1_scores = []
    for batch in range(label.shape[0]):
        per_class_precision = []
        per_class_recall = []
        per_class_f1_score = []
        for num_class in range(label.shape[3]):
            confusion_matrix = compute_confusion_matrix(output[batch, :, :, num_class],
                                                        label[batch, :, :, num_class],
                                                        threshold)
            tn = confusion_matrix[0, 0]
            tp = confusion_matrix[1, 1]
            fp = confusion_matrix[0, 1]
            fn = confusion_matrix[1, 0]
            # fix this -> only make sure denominator is not 0 (+)
            # if (tp + fp) != 0:
            precision = tp / (tp + fp) if tp > 0 else 0.
            # else:
            #     precision = 0.
            # if (tp + fn) != 0:
            recall = tp / (tp + fn) if tp > 0 else 0.
            # else:
            #     recall = 1.

            per_class_precision.append(precision)
            per_class_recall.append(recall)
            per_class_f1_score.append(calculate_f1_score(precision, recall) if tp > 0 else 0.)

        precisions.append(torch.mean(torch.tensor(per_class_precision)))
        recalls.append(torch.mean(torch.tensor(per_class_recall)))
        f1_scores.append(torch.mean(torch.tensor(per_class_f1_score)))
    return torch.mean(torch.tensor(precisions)), torch.mean(torch.tensor(recalls)), torch.mean(torch.tensor(f1_scores)
                                                                                               .nan_to_num(nan=1.))


def compute_precision_recall_f1_3d(output, label, threshold):  # this is a micro style computation
    precisions = []
    recalls = []
    f1_scores = []
    for batch in range(label.shape[0]):
        confusion_matrix = compute_confusion_matrix(output[batch], label[batch], threshold)
        tn = confusion_matrix[0, 0]
        tp = confusion_matrix[1, 1]
        fp = confusion_matrix[0, 1]
        fn = confusion_matrix[1, 0]

        precision = tp / (tp + fp) if tp > 0 else 0.
        recall = tp / (tp + fn) if tp > 0 else 0.

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(calculate_f1_score(precision, recall))

    return torch.mean(torch.tensor(precisions)), torch.mean(torch.tensor(recalls)), torch.mean(torch.tensor(f1_scores))


def calculate_precision(output, label):
    per_batch_precision = []
    for batch in range(label.shape[0]):
        per_class_precision = []
        for num_class in range(label.shape[3]):
            one_label = label[batch, :, :, num_class].clone().detach().cpu()
            one_output = output[batch, :, :, num_class].clone().detach().cpu()
            per_class_precision.append(precision_score(torch.flatten(one_label).numpy(),
                                                       torch.flatten(one_output).numpy()
                                                       , zero_division=0))
        mean_precision = torch.mean(torch.tensor(per_class_precision))
        per_batch_precision.append(mean_precision)
    return torch.mean(torch.tensor(per_batch_precision))


def calculate_recall(output, label):
    per_batch_precision = []
    for batch in range(label.shape[0]):
        per_class_precision = []
        for num_class in range(label.shape[3]):
            one_label = label[batch, :, :, num_class].clone().detach().cpu()
            one_output = output[batch, :, :, num_class].clone().detach().cpu()
            per_class_precision.append(recall_score(torch.flatten(one_label).numpy(),
                                                    torch.flatten(one_output).numpy()
                                                    , zero_division=0))
        mean_precision = torch.mean(torch.tensor(per_class_precision))
        per_batch_precision.append(mean_precision)
    return torch.mean(torch.tensor(per_batch_precision))


def calculate_f1_score(precision, recall):
    denominator = (precision + recall)
    denominator = denominator if denominator != 0 else 1e-6
    return 2 * (precision * recall) / denominator

#
# def compute_binary_output(output):
#     return torch.where(output[:, 0] > output[:, 1], output[:, 0], torch.tensor(0, dtype=torch.float32).cuda())
