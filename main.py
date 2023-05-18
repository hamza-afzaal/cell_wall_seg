from datetime import datetime

import torch
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms
from tqdm import tqdm

import utils.transformsV2
from models.UNet import UNet
from utils.SegmentationDataLoaderH5 import SegmentationDataLoaderH5
from utils.Utils import batch_pad_collate_v2, calculate_accuracy, calculate_iou, compute_precision_recall_f1
from utils.common import get_data_loader, save_model, load_model, compute_measure_values
from utils.loss_functions import DiceBCELoss
from utils.transforms import Normalize, GroundTruthToBoundary, GaussianBlur, OutputClassLabelling


def load_dataset(path, x_transforms, y_transforms, lim_dataset=None, data_augmentation=None, generate_patches=False):
    training_dataset = SegmentationDataLoaderH5(files_path=path,
                                                input_transforms=x_transforms,
                                                target_transforms=y_transforms,
                                                lim_dataset=lim_dataset,
                                                data_augmentation=data_augmentation,
                                                generate_patches=generate_patches)
    return training_dataset


def get_data_transforms():
    input_transforms = transforms.Compose([
        Normalize(ground_label=False)
    ])

    output_transforms = transforms.Compose([
        GroundTruthToBoundary(ground_label=False),
        GaussianBlur(ground_label=True),
        OutputClassLabelling(ground_label=True)
    ])
    return input_transforms, output_transforms


def get_data_transforms_v2():
    input_transforms = transforms.Compose([
        utils.transformsV2.Normalize(ground_label=False),
        utils.transformsV2.ToTensor({"normalized": torch.float32})
    ])

    output_transforms = transforms.Compose([
        utils.transformsV2.GroundTruthToBoundary(ground_label=True),
        utils.transformsV2.GaussianBlur(ground_label=True),
        utils.transformsV2.OutputClassLabelling(ground_label=True),
        utils.transformsV2.ToTensor({"boundaries": torch.float32, 'blurred': torch.float32,
                                     'boundary_label': torch.float32, 'background_label': torch.float32})
    ])
    return input_transforms, output_transforms


def get_data_transforms_input():
    input_transforms = transforms.Compose([
        utils.transformsV2.Normalize(ground_label=False),
        utils.transformsV2.ToTensor({"normalized": torch.float32})
    ])

    output_transforms = transforms.Compose([
        utils.transformsV2.GroundTruthToBoundary(ground_label=True),
        utils.transformsV2.GaussianBlur(ground_label=True),
        utils.transformsV2.OutputClassLabelling(ground_label=True),
        utils.transformsV2.ToTensor({"boundaries": torch.float32, 'blurred': torch.float32,
                                     'boundary_label': torch.float32, 'background_label': torch.float32})
    ])

    data_transforms = transforms.Compose([
        utils.transformsV2.RandomFlipV2(threshold=0.5, orientation='horizontal', input_keys={'normalized': torch.float32},
                                        output_keys={'boundary_label': torch.float32}),
        utils.transformsV2.RandomFlipV2(threshold=0.5, orientation='vertical', input_keys={'normalized': torch.float32},
                                        output_keys={'boundary_label': torch.float32}),
        utils.transformsV2.ElasticDeformationV2(threshold=0.5, input_keys={'normalized': torch.float32},
                                                output_keys={'boundary_label': torch.float32}),
        utils.transformsV2.AdditiveGaussianNoise(),
        utils.transformsV2.AdditivePoissonNoise()
    ])
    return input_transforms, output_transforms, data_transforms


def get_data_transforms_train():
    input_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        utils.transformsV2.ToNumpy(),
        utils.transformsV2.Normalize(ground_label=False),
        utils.transformsV2.ToTensor({"normalized": torch.float32})
    ])

    output_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        utils.transformsV2.ToNumpy(),
        utils.transformsV2.GroundTruthToBoundary(ground_label=True),
        utils.transformsV2.GaussianBlur(ground_label=True),
        utils.transformsV2.OutputClassLabelling(ground_label=True),
        utils.transformsV2.ToTensor({"boundaries": torch.float32, 'blurred': torch.float32,
                                     'boundary_label': torch.float32, 'background_label': torch.float32})
    ])
    return input_transforms, output_transforms


def print_data(raw, output):
    plt.imshow(raw[0, 0, 0], interpolation='nearest')
    plt.show()
    plt.imshow(output[0, 0, 0], interpolation='nearest')
    plt.show()
    plt.imshow(output[1, 0, 0], interpolation='nearest')
    plt.show()
    plt.imshow(output[2, 0, 0], interpolation='nearest')
    plt.show()
    plt.imshow(output[3, 0, 0], interpolation='nearest')
    plt.show()


def print_output_img(output):
    output_clone = output[1, :, :, 0].clone().detach().cpu()
    output_clone = torch.nn.Sigmoid()(output_clone)
    plt.imshow(output_clone, interpolation='nearest')
    plt.show()


def compute_precision_recall_curve(output, target):
    list_precision = []
    list_recall = []
    list_threshold = []
    for index in np.arange(0.05, 1., .1):
        precision, recall, f1_score = compute_precision_recall_f1(output, target, index)
        list_precision.append(precision)
        list_recall.append(recall)
        list_threshold.append(index)
    return np.array([list_precision, list_recall, list_threshold])


def test_model(model, test_data_loader, threshold):
    num_iterations = len(test_data_loader)
    testing_precision = []
    testing_recall = []
    testing_f1_score = []
    test_accuracy = []
    test_iou_score = []
    precision_recall_score = []
    with torch.no_grad():
        progress_bar = tqdm(test_data_loader)
        for i, (images, labels) in enumerate(progress_bar):
            images = images.cuda()
            labels = labels.cuda()

            labels = labels[:1]
            labels = torch.reshape(labels, (labels.shape[0], labels.shape[1], labels.shape[3], labels.shape[4]))
            labels = labels.permute(1, 2, 3, 0)

            output = torch.nn.Sigmoid()(model(images[0]))
            output = output.permute(0, 2, 3, 1)

            # output = output[:, :, :, 0]
            # labels = labels[:, :, :, 0]
            # output = output[:, :, :, None]
            # labels = labels[:, :, :, None]

            precision, recall, f1_score = compute_precision_recall_f1(output, labels, threshold)
            testing_precision.append(precision)
            testing_recall.append(recall)
            testing_f1_score.append(f1_score)

            accuracy = calculate_accuracy(output, labels, threshold)
            test_accuracy.append(accuracy)
            iou_score = calculate_iou(output, labels, threshold)
            test_iou_score.append(iou_score)

            if torch.tensor(iou_score).mean() > 0:
                precision_recall_score.append(compute_precision_recall_curve(output,
                                                                             labels))

            # for out in range(labels.shape[0]):
            #     plt.imshow(images[0, out, 0].cpu(), interpolation='nearest')
            #     plt.imsave(f"test_results/{i}-{out}-input-iou-{iou_score:.4f}.png", images[0, out, 0].cpu())
            #     plt.imshow(labels[out, :, :, 0].cpu(), interpolation='nearest')
            #     plt.imsave(f"test_results/{i}-{out}-gt-iou-{iou_score:.4f}.png", labels[out, :, :, 0].cpu())
            #     plt.imshow(output[out, :, :, 0].cpu() > 0.5, interpolation='nearest')
            #     plt.imsave(f"test_results/{i}-{out}-output-iou-{iou_score:.4f}.png", output[out, :, :, 0].cpu() > 0.5)

            progress_bar.set_description(f"## Testing ## ->Accuracy: {accuracy}% - "
                                         f"Precision: {precision:.4f} - "
                                         f"Recall: {recall:.4f} - "
                                         f"F1 Score: {f1_score:.4f} - "
                                         f"IOU Score: {iou_score}")
        test_accuracy = sum(test_accuracy) / num_iterations

        iou_mean, iou_std, old_iou_mean, old_iou_std = compute_measure_values(test_iou_score)
        precision_mean, precision_std, old_precision_mean, old_precision_std = compute_measure_values(testing_precision)
        recall_mean, recall_std, old_recall_mean, old_recall_std = compute_measure_values(testing_recall)
        f1_score_mean, f1_score_std, old_f1_score_mean, old_f1_score_std = compute_measure_values(testing_f1_score)

        print(f"##METRICS##\n"
              f"IOU: mean: {iou_mean:4f} - std: {iou_std:4f} - old_mean: {old_iou_mean:4f} - "
              f"old_std: {old_iou_std:4f}\n",
              f"Precision: mean: {precision_mean:4f} - std: {precision_std:4f} - old_mean: {old_precision_mean:4f} - "
              f"old_std: {old_precision_std:4f}\n",
              f"Recall: mean: {recall_mean:4f} - std: {recall_std:4f} - old_mean: {old_recall_mean:4f} - "
              f"old_std: {old_recall_std:4f}\n",
              f"F1 Score: mean: {f1_score_mean:4f} - std: {f1_score_std:4f} - old_mean: {old_f1_score_mean:4f} - "
              f"old_std: {old_f1_score_std:4f}\n")

        return test_accuracy, test_iou_score, testing_precision, testing_recall, testing_f1_score


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
            labels = labels[:1]
            labels = torch.reshape(labels, (labels.shape[0], labels.shape[1], labels.shape[3], labels.shape[4]))
            labels = labels.permute(1, 2, 3, 0)

            output = model(images[0])
            output = output.permute(0, 2, 3, 1)

            precision, recall, f1_score = compute_precision_recall_f1(output, labels, threshold)
            validation_precision.append(precision)
            validation_recall.append(recall)
            validation_f1_score.append(f1_score)

            accuracy = calculate_accuracy(output, labels, threshold)
            validation_accuracy.append(accuracy)
            iou_score = calculate_iou(output, labels, threshold)
            validation_iou_score.append(iou_score)
            progress_bar.set_description(f"## Validation epoch {epoch}## ->Accuracy: {accuracy:.4f}% - "
                                         f"Precision: {precision:.4f} - "
                                         f"Recall: {recall:.4f} - "
                                         f"F1 Score: {f1_score:.4f} - "
                                         f"IOU Score: {iou_score:.4f} - "
                                         f"Overall IOU Score: {compute_measure_values(validation_iou_score)[0]:.4f}")
        validation_accuracy = sum(validation_accuracy) / num_iterations
        validation_iou_score = compute_measure_values(validation_iou_score)[0]
        validation_precision = compute_measure_values(validation_precision)[0]
        validation_recall = compute_measure_values(validation_recall)[0]
        validation_f1_score = compute_measure_values(validation_f1_score)[0]
        return validation_accuracy, validation_iou_score, validation_precision, validation_recall, validation_f1_score


def train_model(model, batch_size, epoch, training_data_loader, criterion, optimizer, threshold):
    num_iterations = len(training_data_loader)
    # accuracy_metric = Accuracy(mdmc_average='samplewise')
    # precision_metric = Precision(mdmc_average='samplewise')  # num_classes=2
    # recall_metric = Recall(mdmc_average='samplewise')
    # f1_metric = F1(mdmc_average='samplewise')
    # iou_metric = IoU(num_classes=1)
    # print(f"The number of iterations per epoch are {num_iterations}")

    training_losses = []
    training_accuracy = []
    training_iou_score = []
    start_epoch = datetime.now()
    progress_bar = tqdm(training_data_loader)
    for i, (images, labels) in enumerate(progress_bar):

        images = images.cuda()
        labels = labels[:1].cuda()
        labels = torch.reshape(labels, (labels.shape[0], labels.shape[1], labels.shape[3], labels.shape[4]))

        output = model(images[0])

        output = output.permute(0, 2, 3, 1)  # batch, height, width, channels
        labels = labels.permute(1, 2, 3, 0)  # batch, height, width, channels
        # print_output_img(output)

        # labels = labels[:1]

        output_clone = torch.nn.Sigmoid()(output).detach()
        # output_clone = (output_clone.detach() > threshold).long()
        # output_clone = torch.stack((output_clone, torch.where(output_clone > 0, 0, 1)), dim=3)

        # precision, recall = compute_precision_recall(output, labels, threshold)

        accuracy = calculate_accuracy(output_clone, labels, threshold)
        training_accuracy.append(accuracy)
        iou_score = calculate_iou(output_clone, labels, threshold)
        training_iou_score.append(iou_score)

        # iou_score = calculate_iou(output_clone, labels)
        # training_iou_score.append(iou_score)
        #
        # accuracy = calculate_accuracy(output_clone, labels)
        # training_accuracy.append(accuracy)

        # accuracy = calculate_metric(output_clone, labels, accuracy_metric)
        # precision = calculate_metric(output_clone, labels, precision_metric)
        # recall = calculate_metric(output_clone, labels, recall_metric)
        # f1_score = calculate_metric(output_clone, labels, f1_metric)
        # iou_score = calculate_metric(output_clone, labels, iou_metric)
        # precision_other = calculate_precision(output_clone, labels)
        # recall = calculate_recall(output_clone, labels)
        # f1_score = calculate_f1_score(precision, recall)

        # labels_foreground = labels[0].reshape(batch_size * labels.shape[3] * labels.shape[4])
        # labels_background = labels[1].reshape(batch_size * labels.shape[3] * labels.shape[4])

        output = output.reshape(output.shape[0] * output.shape[1] * output.shape[2], output.shape[3])  # 2
        labels = labels.reshape(labels.shape[0] * labels.shape[1] * labels.shape[2], labels.shape[3])

        # output_clone = output_clone.reshape(batch_size * output_clone.shape[1] * output_clone.shape[2],
        #                                     output_clone.shape[3])
        #
        # SegmentationMetrics(labels.cpu().numpy(), output_clone.detach().cpu().numpy()).metrics(threshold)
        # labels_background = labels.reshape(batch_size * labels.shape[3] * labels.shape[4], 2)
        # labels = torch.stack((labels_foreground, labels_background), dim=1)

        # output_clone = (torch.max(torch.nn.Sigmoid()(output), dim=1).values > threshold).float() * 1
        # accuracy = calculate_accuracy(output_clone, labels)
        # training_accuracy.append(accuracy)

        loss = criterion(output, labels)

        loss_val = loss.item()
        training_losses.append(loss.item())

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        # f"Time elapsed for {num_iterations} iterations: {str(datetime.now() - start)}"
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

        train_data_path = './ovules-dataset/train_comp/'
        val_data_path = './ovules-dataset/val_comp/'
        test_data_path = './ovules-dataset/test/'
        save_model_path = './trained_models/UNet-ADAM-DICE-BCE - LATEST - Continue - Noise.pth'
        # test_model_path = './trained_models/UNet-ADAM-DICE-BCE - LATEST.pth'

        raw_transforms, label_transforms = get_data_transforms_v2()
        train_input_transforms, train_output_transforms, train_data_augmentations = get_data_transforms_input()

        data_augmentation = transforms.Compose([
            utils.transformsV2.RandomFlip(threshold=0.5, orientation='horizontal'),
            utils.transformsV2.RandomFlip(threshold=0.5, orientation='vertical'),
            utils.transformsV2.ElasticDeformation(threshold=0.5)
        ])

        training_dataset = load_dataset(train_data_path,
                                        train_input_transforms,
                                        train_output_transforms,
                                        data_augmentation=train_data_augmentations,
                                        generate_patches=False)
        training_data_loader = get_data_loader(training_dataset, batch_pad_collate_v2, batch_size, num_workers=0)

        val_dataset = load_dataset(val_data_path, raw_transforms, label_transforms, generate_patches=False)
        # generate patches here as well to keep the testing consistent
        val_data_loader = get_data_loader(val_dataset, batch_pad_collate_v2, batch_size=2, num_workers=4)

        test_dataset = load_dataset(test_data_path, raw_transforms, label_transforms, generate_patches=False)
        test_data_loader = get_data_loader(test_dataset, batch_pad_collate_v2, batch_size=1, num_workers=4)

        model = UNet(1, 1).cuda()

        # criterion = torch.nn.BCEWithLogitsLoss()
        criterion = DiceBCELoss()
        # optimizer = torch.optim.SGD(model.parameters(), 2e-4, momentum=0.99)
        optimizer = torch.optim.Adam(model.parameters(), 2e-4, betas=(0.9, 0.999), weight_decay=0.00001)

        total_training_losses = []
        total_training_accuracy = []
        total_training_iou_score = []

        total_validation_accuracy = []
        total_validation_iou_score = []

        # model = load_model(model, save_model_path)
        # test_model(model, test_data_loader, threshold)

        for epoch in range(epochs):
            training_losses, training_accuracy, training_iou_score = train_model(model, batch_size, epoch,
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
