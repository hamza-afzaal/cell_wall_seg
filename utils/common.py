import torch
from torch.utils import data
import numpy as np


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def get_data_loader(training_data, collate_fn=None, batch_size=10, num_workers=4, shuffle=True):
    training_dataloader = data.DataLoader(dataset=training_data,
                                          batch_size=batch_size,
                                          collate_fn=collate_fn,
                                          num_workers=num_workers,
                                          shuffle=shuffle)
    return training_dataloader


def compute_measure_values(measure_values):
    measure_values = np.array(measure_values)
    measure_mask = measure_values > 0
    measure = measure_values[measure_mask]
    if len(measure) > 0:
        measure_mean, measure_std = measure.mean(), measure.std()
    else:
        measure_mean, measure_std = 0., 0.
    old_measure_mean, old_measure_std = measure_values.mean(), measure_values.std()
    return measure_mean, measure_std, old_measure_mean, old_measure_std


# def compute_precision_recall_curve(output, target):
#     list_precision = []
#     list_recall = []
#     list_threshold = []
#     for index in np.arange(0., 1., .01):
#         precision, recall, f1_score = compute_precision_recall_f1_3d(output, target, index)
#         list_precision.append(precision)
#         list_recall.append(recall)
#         list_threshold.append(index)
#     return np.array([list_precision, list_recall, list_threshold])

