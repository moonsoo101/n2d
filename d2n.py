import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from time import time
from pathlib import Path
from PIL import Image
import cv2
import os, glob, re
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
torch.nn.Module.dump_patches = False
from torchsummary import summary


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(inplace=True)
    )


class TCNN2(torch.nn.Module):
    def __init__(self):
        super(TCNN2, self).__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, 3, 1)
        self._initialize_weights()

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)


class Data(Dataset):
    def __init__(self, dataSetPath, start_idx, dataset_name='n2d'):
        self.data_length = 500
        self.day_images = np.ndarray((self.data_length, 3, 256, 256), dtype=np.uint8)
        self.night_images = np.ndarray((self.data_length, 3, 256, 256), dtype=np.uint8)
        self.dataSetPath = dataSetPath
        self.start_idx = start_idx
        self.b_more = True
        self.dataset_name = dataset_name
        self.getData()

    def __getitem__(self, index):
        return self.day_images[index], self.night_images[index]

    def __len__(self):
        return len(self.night_images)

    def getData(self):
        data_list = list(Path(self.dataSetPath).rglob('*.jpg'))[self.start_idx:self.start_idx + self.data_length]
        list_len = len(data_list)

        if list_len == 0:
            self.b_more = False

        self.day_images = np.ndarray((list_len, 3, 256, 256), dtype=np.uint8)
        self.night_images = np.ndarray((list_len, 3, 256, 256), dtype=np.uint8)

        for start_idx, img_path in enumerate(data_list):
            img = np.asarray(Image.open(img_path), dtype="uint8")
            self.night_images[start_idx, 0, :, :] = img[:, 0:256, 0]
            self.night_images[start_idx, 1, :, :] = img[:, 0:256, 1]
            self.night_images[start_idx, 2, :, :] = img[:, 0:256, 2]

            self.day_images[start_idx, 0, :, :] = img[:, 256:, 0]
            self.day_images[start_idx, 1, :, :] = img[:, 256:, 1]
            self.day_images[start_idx, 2, :, :] = img[:, 256:, 2]


class TrainTest():

    def __init__(self):
        self.data_length = 500
        self.checkpoint_path = './model/n2d/checkpoint/'
        self.summary = SummaryWriter()

    def Train(self, epochs, model_name='n2d'):
        self.checkpoint_path = './model/%s/BCE/checkpoint/' % model_name
        self.model = TCNN2()
        print('Model : %s' % model_name)

        if os.path.exists(os.path.join(self.checkpoint_path, 'model.pth')):
            self.model = torch.load(os.path.join(self.checkpoint_path, 'model.pth'))

        self.model.train()

        self.criterion = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.model.cuda()
        self.error = np.ndarray((epochs,))
        initail_epoch = self.findLastCheckpoint()

        for t in range(initail_epoch, epochs+initail_epoch):
            print('Train epochs %d' % (t+1))
            epoch_loss = 0
            epochs_itr = 0

            dataset_idx = 0
            dataSetPath = './dataset/%s/train/pick' % model_name

            while True:
                data_set = Data(dataSetPath, dataset_idx * self.data_length, model_name)

                if not data_set.b_more:
                    break
                train_data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                                                num_workers=4, batch_size=4)
                dataset_idx = dataset_idx + 1
                for batch_idx, batch_data in enumerate(train_data_loader):
                    X, Y = Variable(batch_data[0].float().cuda())/255.0, Variable(batch_data[1].float().cuda())/255.0
                    Y_pred = self.model(X)
                    # fig = plt.figure(figsize=(2, 2))
                    # fig.add_subplot(1, 3, 1)
                    # plt.imshow(np.transpose(X.cpu().data[1, :, :, :], (1, 2, 0)), interpolation='bilinear')
                    # fig.add_subplot(1, 3, 2)
                    # plt.imshow(np.transpose(Y.cpu().data[1, :, :, :], (1, 2, 0)), interpolation='bilinear')
                    # fig.add_subplot(1, 3, 3)
                    # plt.imshow(np.transpose(Y_pred.cpu().data[1, :, :, :], (1, 2, 0)), interpolation='bilinear')
                    # plt.title('Reconstructed image after training')
                    # plt.show()
                    # plt.imshow(np.transpose(Y_pred.cpu().data[0, :, :, :], (1, 2, 0)), interpolation='bilinear')
                    # plt.show()
                    # Compute and print loss
                    print(torch.sum(Y_pred.cpu().data[:, :, :, :]), torch.sum(Y.cpu().data[:, :, :, :]))
                    print(Y_pred.cpu().data[:, :, :, :]+Y.cpu().data[:, :, :, :])
                    loss = self.criterion(Y_pred, Y)
                    print( loss)
                    epoch_loss += loss.item()
                    # Zero gradients
                    self.optimizer.zero_grad()
                    # Perform a backward pass (back propagation)
                    loss.backward()
                    # Update the weights.
                    self.optimizer.step()
                    #end one image batch set(500)
                torch.save(self.model, os.path.join(self.checkpoint_path, 'model.pth'))
                epochs_itr += batch_idx + 1
                print('epoch%d, itr:%d, loss:%f' % (t + 1, epochs_itr, loss.item()))
            #end one epoch
            print('END epoch%d' %(t+1))
            torch.save(self.model, os.path.join(self.checkpoint_path, "model_%d.pth" % (t+1)))
            # ...log the running loss
            self.summary.add_scalar('loss/train',
                                    epoch_loss/epochs_itr,
                                    (t+1))
            # self.Test()
        #self.summary.close()

        # plt.figure()
        # plt.plot(self.error)
        # plt.xlabel('epoch')
        # plt.ylabel('MSE')
        # plt.title('MSE Error plot')

        # if verbose:
        #     fig = plt.figure(figsize=(2, 2))
        #     fig.add_subplot(1, 2, 1)
        #     plt.imshow(np.transpose(np.uint8(X.cpu().data[0, :, :, :]), (1, 2, 0)), interpolation='bilinear')
        #     fig.add_subplot(1, 2, 2)
        #     plt.imshow(np.transpose(np.uint8(Y_pred.cpu().data[0, :, :, :]), (1, 2, 0)), interpolation='bilinear')
        #     plt.title('Reconstructed image after training')
        #     plt.show(block=False)

    def Test(self, model_name='n2d', b_save=False):
        self.model = TCNN2()
        self.checkpoint_path = './model/%s/checkpoint/' % model_name
        if os.path.exists(os.path.join(self.checkpoint_path, 'model.pth')):
            self.model = torch.load(os.path.join(self.checkpoint_path, 'model.pth'))

        self.model.eval()

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.model.cuda()

        test_idx = 0
        testPath = './dataset/%s/train/pick' % model_name
        save_path = './res/%s' % model_name
        #save_path = './dataset/deblur/train/pick'
        iter = 1

        while True:
            data_set = Data(testPath, test_idx * self.data_length, model_name)

            if not data_set.b_more:
                break
            train_data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                                            num_workers=4, batch_size=16)
            for batch_idx, batch_data in enumerate(train_data_loader):
                X, Y = Variable(batch_data[0].float().cuda())/255.0, Variable(batch_data[1].float().cuda())/255.0
                Y_pred = self.model(X)
                # Compute and print loss
                loss = self.criterion(Y_pred, Y)
                if b_save:
                    for i in range(len(Y)):
                        # new_data = np.dstack((X.cpu().data[i, :, :, :], Y_pred.cpu().data[i, :, :, :], Y.cpu().data[i, :, :, :],)) * 255.0
                        # #new_data = np.dstack((Y.cpu().data[i, :, :, :],Y_pred.cpu().data[i, :, :, :])) * 255.0
                        # new_data = np.transpose(new_data, (1, 2, 0))
                        # new_data = Image.fromarray(new_data.astype('uint8'))
                        # new_data.save(os.path.join(save_path, '%d.jpg' % iter))

                        fig = plt.figure(figsize=(8, 6), dpi=80)
                        ax1 = fig.add_subplot(1, 3, 1)
                        ax1.set_title('Input')
                        ax1.axis('off')
                        ax1.imshow(np.transpose(X.cpu().data[1, :, :, :], (1, 2, 0)), interpolation='bilinear')

                        ax2 = fig.add_subplot(1, 3, 2)
                        ax2.set_title('Output')
                        ax2.axis('off')
                        ax2.imshow(np.transpose(Y.cpu().data[1, :, :, :], (1, 2, 0)), interpolation='bilinear')
                        ax3 = fig.add_subplot(1, 3, 3)
                        ax3.set_title('Ground Truth')
                        ax3.axis('off')
                        ax3.imshow(np.transpose(Y_pred.cpu().data[1, :, :, :], (1, 2, 0)), interpolation='bilinear')
                        #plt.title('Reconstructed image after training')
                        plt.savefig(os.path.join(save_path, '%d.png' % iter))
                        iter += 1
                print("loss :", loss.item())
                # a = np.subtract(Y_pred.cpu().data[:,:,:,:], Y.cpu().data[:,:,:,:])
                # print("array:", a.sum())

            test_idx = test_idx + 1

    def findLastCheckpoint(self):
        file_list = glob.glob(os.path.join(self.checkpoint_path, 'model_*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*model_(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch = max(epochs_exist)
        else:
            initial_epoch = 0
        return initial_epoch

if __name__ == '__main__':
    tic = time()
    epochs = 100
    TT = TrainTest()
    TT.Train(epochs, 'n2d')
    #TT.Train(epochs, 'deblur')
    #TT.Test('deblur', False)
    #TT.Test('deblur', True)
    TT.Test('n2d', True)

    toc = time()
    print('Total Time of Execution %.2f min' % ((toc - tic) / 60.0))
