import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime

import utils.transformsV2
from models.UNet3D import UNet3D
from utils.SegDataLoader3D import SegDataLoader3D
from utils.Utils import calculate_accuracy, calculate_iou, compute_precision_recall_f1, calculate_iou_3d, \
    compute_precision_recall_f1_3d
from utils.common import get_data_loader, save_model, load_model, compute_measure_values
from utils.loss_functions import DiceBCELoss, DiceFocalLoss
from utils.transforms3D import Normalize3D, ToTensor3D, GroundTruthToBoundary3D, GaussianBlur3D, RandomFlip3D, \
    ElasticDeformation3D, AdditiveGaussianNoise, AdditivePoissonNoise


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


def val_model(model, val_data_loader, epoch, threshold):
    num_iterations = len(val_data_loader)
    validation_accuracy = []
    validation_iou_score = []
    validation_precision = []
    validation_recall = []
    validation_f1_score = []
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
            validation_precision.append(precision)
            validation_recall.append(recall)
            validation_f1_score.append(f1_score)

            accuracy = calculate_accuracy(output_metric, label_metric, threshold)
            validation_accuracy.append(accuracy)
            iou_score = calculate_iou_3d(output_metric, label_metric, threshold)
            validation_iou_score.append(iou_score)
            progress_bar.set_description(f"## Validation epoch {epoch}## ->Accuracy: {accuracy:.4f}% - "
                                         f"Precision: {precision:.4f} - "
                                         f"Recall: {recall:.4f} - "
                                         f"F1 Score: {f1_score:.4f} - "
                                         f"IOU Score: {iou_score:.4f}")
        validation_accuracy = sum(validation_accuracy) / num_iterations
        validation_iou_score = compute_measure_values(validation_iou_score)[0]
        validation_precision = compute_measure_values(validation_precision)[0]
        validation_recall = compute_measure_values(validation_recall)[0]
        validation_f1_score = compute_measure_values(validation_f1_score)[0]
        return validation_accuracy, validation_iou_score, validation_precision, validation_recall, validation_f1_score


def train_model(model, epoch, training_data_loader, criterion, optimizer, threshold):
    num_iterations = len(training_data_loader)

    training_losses = []
    training_accuracy = []
    training_iou_score = []
    start_epoch = datetime.now()
    progress_bar = tqdm(training_data_loader)
    for i, (images, labels) in enumerate(progress_bar):
        images = images.cuda()
        labels = labels.cuda()

        output = model(images)

        output_clone = torch.nn.Sigmoid()(output).detach()
        output = output.permute(0, 2, 3, 4, 1)  # batch, height, width, channels
        labels = labels.permute(0, 2, 3, 4, 1)  # batch, height, width, channels

        output_metric, label_metric = reshape_output(output, labels)

        accuracy = calculate_accuracy(output_metric, label_metric, threshold)
        training_accuracy.append(accuracy)
        iou_score = calculate_iou_3d(output_metric, label_metric, threshold)
        training_iou_score.append(iou_score)
        # overall_iou = torch.tensor(training_iou_score)
        # overall_iou_mask = overall_iou > 0
        # overall_iou = overall_iou[overall_iou_mask]
        # overall_iou_mean = overall_iou.mean()

        output = output.reshape(output.shape[0] * output.shape[1] * output.shape[2], output.shape[3],
                                output.shape[4])  # 2
        labels = labels.reshape(labels.shape[0] * labels.shape[1] * labels.shape[2], labels.shape[3], labels.shape[4])

        loss = criterion(output, labels)

        loss_val = loss.item()
        training_losses.append(loss.item())

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        progress_bar.set_description(f"## Training Epoch {epoch}## -> Loss: {loss_val:.4f} - "
                                     f"Total loss: {sum(training_losses) / (i + 1):.4f} - "
                                     f"Accuracy: {accuracy:.4f}% - "
                                     # f"Precision: {precision:.4f} - "
                                     # f"Recall: {recall:.4f} - "
                                     # f"F1 score: {f1_score:.4f} - "
                                     f"IOU Score: {iou_score:.4f} - "
                                     f"Overall IOU Score: {compute_measure_values(training_iou_score)[0]:.4f}"
                                     )

    print("Time elapsed this epoch: " + str(datetime.now() - start_epoch))
    training_losses = sum(training_losses) / num_iterations
    training_accuracy = sum(training_accuracy) / num_iterations
    training_iou_score = sum(training_iou_score) / num_iterations
    return training_losses, training_accuracy, training_iou_score


def main():
    with torch.cuda.device(1):
        batch_size = 1
        epochs = 100
        threshold = 0.5
        torch.manual_seed(0)
        torch.cuda.empty_cache()

        train_data_path = './ovules-dataset/train/'
        val_data_path = './ovules-dataset/val/'
        test_data_path = './ovules-dataset/test/'
        save_model_path = './trained_models/UNet-ADAM-FOCAL-DICE-Batch - 3D - new.pth'

        model = UNet3D(1, 1).cuda()

        input_transforms = transforms.Compose([
            Normalize3D(standardize=False),
            ToTensor3D(torch.float32)
        ])
        input_transforms_val = transforms.Compose([
            Normalize3D(standardize=True),
            ToTensor3D(torch.float32)
        ])
        output_transforms = transforms.Compose([
            GroundTruthToBoundary3D(),
            GaussianBlur3D(),
            ToTensor3D(torch.float32)
        ])

        data_transforms = transforms.Compose({
            RandomFlip3D(orientation="horizontal"),
            RandomFlip3D(orientation="vertical"),
            ElasticDeformation3D(),
            # AdditiveGaussianNoise(),
            # AdditivePoissonNoise()
        })

        criterion = DiceFocalLoss()
        # criterion = DiceBCELoss()
        # criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), 1e-4, betas=(0.9, 0.999), weight_decay=0.00001)

        training_dataset = load_dataset(train_data_path,
                                        x_transforms=input_transforms,
                                        y_transforms=output_transforms,
                                        data_augmentation=data_transforms)

        val_dataset = load_dataset(val_data_path,
                                   x_transforms=input_transforms_val,
                                   y_transforms=output_transforms)

        training_data_loader = get_data_loader(training_dataset, batch_size=batch_size, num_workers=4)
        val_data_loader = get_data_loader(val_dataset, batch_size=1, num_workers=4)

        total_training_losses = []
        total_training_accuracy = []
        total_training_iou_score = []

        total_validation_accuracy = []
        total_validation_iou_score = []

        model = load_model(model, save_model_path)

        for epoch in range(epochs):
            training_losses, training_accuracy, training_iou_score = train_model(model, epoch,
                                                                                 training_data_loader,
                                                                                 criterion, optimizer, threshold)

            print(f"## Training Statistics: Training loss: {training_losses} - "
                  f"Training Accuracy: {training_accuracy} - "
                  f"Training IOU Score: {training_iou_score}")
            total_training_losses.append(training_losses)
            total_training_accuracy.append(training_accuracy)
            total_training_iou_score.append(training_iou_score)
            save_model(model, save_model_path)

            val_accuracy, val_iou_score, val_precision, val_recall, val_f1_score = val_model(model,
                                                                                             val_data_loader,
                                                                                             epoch, threshold)
            print(f"## Validation Statistics: "
                  f"Avg Accuracy: {val_accuracy} - "
                  f"Avg IOU Score: {val_iou_score}"
                  f"Avg Precision: {val_precision}"
                  f"Avg Recall: {val_recall}"
                  f"Avg F1 Score: {val_f1_score}\n")
            total_validation_accuracy.append(val_accuracy)
            total_validation_iou_score.append(val_iou_score)
        save_model(model, save_model_path)


if __name__ == '__main__':
    main()
