import numpy
import torch
from sklearn.metrics import precision_recall_curve
from torchvision import transforms
from tqdm import tqdm

from models.UNet3D import UNet3D
from utils.SegDataLoader3D import SegDataLoader3D
from utils.Utils import calculate_accuracy, calculate_iou_3d, \
    compute_precision_recall_f1_3d
from utils.common import get_data_loader, load_model, compute_measure_values
from utils.transforms3D import Normalize3D, ToTensor3D, GroundTruthToBoundary3D, GaussianBlur3D


def load_dataset(path, x_transforms=None, y_transforms=None, lim_dataset=None, data_augmentation=None,
                 generate_patches=False):
    training_dataset = SegDataLoader3D(files_path=path,
                                       input_transforms=x_transforms,
                                       target_transforms=y_transforms,
                                       lim_dataset=lim_dataset,
                                       data_augmentation=data_augmentation,
                                       generate_patches=generate_patches)
    return training_dataset


def reshape_output(output, labels):
    output = torch.nn.Sigmoid()(output).detach()
    output = output.reshape(output.shape[0], output.shape[1], output.shape[2], output.shape[3])
    labels = labels.reshape(labels.shape[0], labels.shape[1], labels.shape[2], labels.shape[3])
    return output.permute(0, 2, 3, 1), labels.permute(0, 2, 3, 1)


def compute_precision_recall_curve(output, target):
    list_precision = []
    list_recall = []
    list_threshold = []
    for index in numpy.arange(.05, 1., .1):
        precision, recall, f1_score = compute_precision_recall_f1_3d(output, target, index)
        list_precision.append(precision)
        list_recall.append(recall)
        list_threshold.append(index)
    return numpy.array([list_precision, list_recall, list_threshold])


def test_model(model, val_data_loader, threshold):
    num_iterations = len(val_data_loader)
    test_accuracy = []
    test_iou_score = []
    test_precision = []
    test_recall = []
    test_f1_score = []

    precision_recall_score = []
    with torch.no_grad():
        progress_bar = tqdm(val_data_loader)
        for i, (images, labels) in enumerate(progress_bar):
            images = images.cuda()
            labels = labels.cuda()

            output = model(images)

            output_clone = torch.nn.Sigmoid()(output).detach()
            output = output.permute(0, 2, 3, 4, 1)  # batch, height, width, channels
            labels = labels.permute(0, 2, 3, 4, 1)  # batch, height, width, channels

            output_metric, label_metric = reshape_output(output, labels)

            precision, recall, f1_score = compute_precision_recall_f1_3d(output_metric, label_metric, threshold)
            test_precision.append(precision)
            test_recall.append(recall)
            test_f1_score.append(f1_score)

            accuracy = calculate_accuracy(output_metric, label_metric, threshold)
            test_accuracy.append(accuracy)
            iou_score = calculate_iou_3d(output_metric, label_metric, threshold)
            test_iou_score.append(iou_score)

            # if torch.tensor(iou_score).mean() > 0:
            #     precision_recall_score.append(compute_precision_recall_curve(output_metric,
            #                                                                  label_metric))
            # precision_recall_score.append(precision_recall_curve(labels.view(-1).cpu().numpy(),
            #                                                      output_clone.view(-1).cpu().numpy()))
            progress_bar.set_description(f"## Testing## ->Accuracy: {accuracy:.4f}% - "
                                         f"Precision: {precision:.4f} - "
                                         f"Recall: {recall:.4f} - "
                                         f"F1 Score: {f1_score:.4f} - "
                                         f"IOU Score: {iou_score:.4f}")
        precision_recall_score = numpy.array(precision_recall_score)
        test_accuracy = sum(test_accuracy) / num_iterations
        test_iou_score = compute_measure_values(test_iou_score)[0]
        test_precision = compute_measure_values(test_precision)[0]
        test_recall = compute_measure_values(test_recall)[0]
        test_f1_score = compute_measure_values(test_f1_score)[0]
        return test_accuracy, test_iou_score, test_precision, test_recall, test_f1_score


def main():
    with torch.cuda.device(0):
        batch_size = 1
        epochs = 100
        threshold = 0.5
        torch.manual_seed(0)
        torch.cuda.empty_cache()

        test_data_path = './ovules-dataset/test_movie/'
        # save_model_path = './trained_models/UNet-ADAM-FOCAL-DICE-Batch - 3D - new.pth'
        save_model_path = './trained_models/UNet-ADAM-DICE-BCE - 3D.pth'

        model = UNet3D(1, 1).cuda()

        input_transforms_val = transforms.Compose([
            Normalize3D(standardize=True),
            ToTensor3D(torch.float32)
        ])
        output_transforms = transforms.Compose([
            GroundTruthToBoundary3D(),
            GaussianBlur3D(),
            ToTensor3D(torch.float32)
        ])

        test_dataset = load_dataset(test_data_path,
                                    x_transforms=input_transforms_val,
                                    y_transforms=output_transforms)

        test_data_loader = get_data_loader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

        total_training_losses = []
        total_training_accuracy = []
        total_training_iou_score = []

        total_test_accuracy = []
        total_test_iou_score = []

        model = load_model(model, save_model_path)

        test_accuracy, test_iou_score, test_precision, test_recall, test_f1_score = test_model(model,
                                                                                               test_data_loader,
                                                                                               threshold)
        print(f"## Test Statistics: "
              f"Avg Accuracy: {test_accuracy} - "
              f"Avg IOU Score: {test_iou_score}"
              f"Avg Precision: {test_precision}"
              f"Avg Recall: {test_recall}"
              f"Avg F1 Score: {test_f1_score}\n")
        total_test_accuracy.append(test_accuracy)
        total_test_iou_score.append(test_iou_score)


if __name__ == '__main__':
    main()
