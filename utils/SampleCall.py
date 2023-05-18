import numpy as np
from sklearn.metrics import precision_recall_curve


y_true = np.array([[0, 0, 1, 1]])
y_scores = np.array([[0.1, 0.4, 0.35, 0.8]])
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
print(precision, recall, thresholds)





















# from datetime import datetime
#
# import numpy
# import torch
# import h5py
# from os import listdir
# from os.path import isfile, join
# from skimage.segmentation import find_boundaries
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from torch.utils import data
# from utils.SegmentationDataLoaderH5 import SegmentationDataLoaderH5
# from utils.transforms import *
# from utils.Utils import *
# import torch.nn as nn
# from models.AlexNet import AlexNet
# from torchvision.utils import save_image
# from torchvision import transforms
# from torch.nn.utils.rnn import pad_sequence
# from torch.nn.utils.rnn import pack_padded_sequence
# import torch.nn.functional as F
#
# train_data_path = '../ovules-dataset/train/'
# val_data_path = '../ovules-dataset/val/'
# test_data_path = '../ovules-dataset/test/'
#
#
# raw_transformations = transforms.Compose([
#     Normalize(ground_label=False)
# ])
#
# label_transformations = transforms.Compose([
#     GroundTruthToBoundary(ground_label=True),
#     GaussianBlur(ground_label=True)
# ])
#
# training_dataset = SegmentationDataLoaderH5(files_path=train_data_path, input_transforms=raw_transformations,
#                                             target_transforms=label_transformations)
#
# training_dataloader = data.DataLoader(dataset=training_dataset,
#                                       batch_size=10,
#                                       collate_fn=batch_pad_collate,
#                                       num_workers=4,
#                                       shuffle=True)
#
# x, y = next(iter(training_dataloader))
#
# # output = y[0].cpu()
# # input = x[0].cpu()
#
# # plt.imshow(input[0, 0, :, :], interpolation='nearest')
# # plt.show()
# # for index in range(output.shape[0]):
# #     print(torch.max(output[index, 0]), torch.min(output[index, 0]))
# #     plt.imshow(output[index, 1], interpolation='nearest')
# #     plt.show()
# #
# #
# #
# # print(f'x = shape: {len(x[0])}; type: {x[0].dtype}')
# # print(f'x = min: {x[0].min()}; max: {x[0].max()}')
# # print(f'y = shape: {y[0].shape}; class: {y[0].unique().size()}; type: {y.dtype}')
# # print(f'x = min: {y[0].min()}; max: {y[0].max()}')
#
#
# # torch.manual_seed(0)
# # model = AlexNet().cuda()
# #
# # torch.cuda.set_device(0)
# #
# # criterion = nn.CrossEntropyLoss()
# # optimizer = torch.optim.SGD(model.parameters(), 1e-4)
# #
# # start = datetime.now()
# # total_step = len(training_dataloader)
# # for epoch in range(2):
# #     for i, (images, labels) in enumerate(training_dataloader):
# #         images = images[0].cuda(non_blocking=True)
# #         labels = labels[0].cuda(non_blocking=True)
# #
# #         outputs = model(images)
# #         loss = criterion(outputs, labels)
# #
# #         optimizer.zero_grad() #reset autograd
# #         loss.backward() #compute gradiants
# #         optimizer.step() #compute gradial descent
# #
# #         if (i + 1) % 100 == 0:
# #             print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'.format(
# #                 epoch + 1,
# #                 2,
# #                 i + 1,
# #                 total_step,
# #                 loss.item()
# #             ))
# #
# # print('Training complete in: ' + str(datetime.now() - start))
#
#
# # training_files = [train_data_path + file for file in listdir(train_data_path) if isfile(join(train_data_path, file))]
# #
# # inputs = []
# # targets = []
# # # for file_path in training_files:
# # with h5py.File(training_files[0], 'r') as data_file:
# #     print("Keys: %s" % data_file.keys())
# #     label = numpy.array(data_file['label'])
# #     print(label.shape)
# #     plt.imshow(label[:, :, 150], interpolation='nearest')
# #     plt.show()
# #
# #     label = np.expand_dims(label, axis=2)
# #     print(label.shape)
# #     label = label.transpose(3, 2, 0, 1)
# #     # label = label.reshape((label.shape[0], label.shape[1], label.shape[3], label.shape[2]))
# #     print(label.shape)
# #     plt.imshow(label[150, 0, :, :], interpolation='nearest')
# #     plt.show()
# #     # inputs.append(numpy.array(data_file['raw']))
# #     # boundaries = find_boundaries(label, connectivity=2, mode='thick')
# #     # targets.append(boundaries)
# #     # print(label[150, :, :])
# #
# # inputs = np.array(inputs)
# # print(inputs.shape)
#
#
# # test = np.random.randn(354, 592, 951)
# # print(test.shape)
# # test = np.expand_dims(test, axis=2)
# # print(test.shape)
# # test = np.reshape(test, (test.shape[3], test.shape[2], test.shape[0], test.shape[1]))
# # print(test.shape)
# # sample = test[150, 0, :, :]
# # print(sample.shape)
# #
