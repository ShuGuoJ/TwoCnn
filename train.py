import torch
from torch import nn, optim
import numpy as np
from scipy.io import loadmat
from utils import weight_init, loadLabel, splitSampleByClass, denoise
from HSIDataset import HSIDatasetV1, DatasetInfo
from Model.module import TwoCnn
from torch.utils.data import DataLoader
import os
import argparse
from visdom import Visdom

isExists = lambda path: os.path.exists(path)
EPOCHS = 10
LR = 1e-1
BATCHSZ = 128
NUM_WORKERS = 8
SEED = 971104
torch.manual_seed(SEED)
DEVICE = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
viz = Visdom()
def train(model, criterion, optimizer, dataLoader):
    '''
    :param model: 模型
    :param criterion: 目标函数
    :param optimizer: 优化器
    :param dataLoader: 批数据集
    :return: 已训练的模型，训练损失的均值
    '''
    model.train()
    model.to(DEVICE)
    trainLoss = []
    for step, ((spectra, neighbor_region), target) in enumerate(dataLoader):
        spectra, neighbor_region, target = spectra.to(DEVICE), neighbor_region.to(DEVICE), target.to(DEVICE)
        spectra = spectra.unsqueeze(1)
        neighbor_region = neighbor_region.permute((0, 3, 1, 2))
        out = model(spectra, neighbor_region)

        loss = criterion(out, target)
        trainLoss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%5 == 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print('step:{} loss:{} lr:{}'.format(step, loss.item(), lr))
    return model, float(np.mean(trainLoss))

def test(model, criterion, dataLoader):
    model.eval()
    evalLoss, correct = [], 0
    for (spectra, neighbor_region), target in dataLoader:
        spectra, neighbor_region, target = spectra.to(DEVICE), neighbor_region.to(DEVICE), target.to(DEVICE)
        spectra = spectra.unsqueeze(1)
        neighbor_region = neighbor_region.permute((0, 3, 1, 2))
        logits = model(spectra, neighbor_region)
        loss = criterion(logits, target)
        evalLoss.append(loss.item())
        pred = torch.argmax(logits, dim=-1)
        correct += torch.sum(torch.eq(pred, target).int()).item()
    acc = float(correct) / len(dataLoader.dataset)
    return acc, np.mean(evalLoss)

def main(datasetName):
    # 加载数据和标签
    info = DatasetInfo.info[datasetName]
    root = 'data/{}'.format(datasetName)
    data_path = os.path.join(root, '{}.mat'.format(datasetName))
    isExists(data_path)
    data = loadmat(data_path)[info['data_key']]
    label_path = os.path.join(root, '{}_gt.mat'.format(datasetName))
    isExists(label_path)
    gt = loadmat(label_path)[info['label_key']]
    # 去除噪声波段
    data = denoise(datasetName, data)
    # 数据集分割
    rate = 0.9 if datasetName == 'Indian' else 0.15
    train_gt, test_gt = splitSampleByClass(gt, rate, SEED)
    # 数据转换
    bands = data.shape[2]
    data, trainLabel, testLabel = data.astype(np.float32), train_gt.astype(np.int), test_gt.astype(np.int)
    nc = int(np.max(trainLabel))
    trainDataset = HSIDatasetV1(data, trainLabel, patchsz=21)
    testDataset = HSIDatasetV1(data, testLabel, patchsz=21)
    trainLoader = DataLoader(trainDataset, batch_size=BATCHSZ, shuffle=True, num_workers=NUM_WORKERS)
    testLoader = DataLoader(testDataset, batch_size=BATCHSZ, shuffle=True, num_workers=NUM_WORKERS)
    # TwoCnn(bands, nc)
    model = TwoCnn(bands, nc)
    model.apply(weight_init)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.8)

    for epoch in range(EPOCHS):
        print('*'*5 + 'Epoch:{}'.format(epoch) + '*'*5)
        model, trainLoss = train(model, criterion=criterion, optimizer=optimizer, dataLoader=trainLoader)
        acc, evalLoss = test(model, criterion=criterion, dataLoader=testLoader)
        viz.line([[trainLoss, evalLoss]], [epoch], win='train&test loss', update='append')
        viz.line([acc], [epoch], win='accuracy', update='append')
        print('epoch:{} trainLoss:{:.8f} evalLoss:{:.8f} acc:{:.4f}'.format(epoch, trainLoss, evalLoss, acc))
        print('*'*18)
        save_root = 'preTrained/{}'.format(datasetName)
        if epoch % 500 == 0:
            if not os.path.isdir(save_root):
                os.makedirs(save_root)
            torch.save(model.state_dict(),
                       os.path.join(save_root, 'preTrained_by_{}_epoch{}.pkl'.format(datasetName, epoch)))
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain TwoCnn')
    parser.add_argument('--name', type=str, default='PaviaU',
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=1,
                        help='模型的训练次数')
    parser.add_argument('--lr', type=float, default=1e-1,
                        help='learning rate')

    args = parser.parse_args()
    EPOCHS = args.epoch
    datasetName = args.name
    LR = args.lr
    viz.line([[0., 0.]], [0.], win='train&test loss', opts=dict(title='train&test loss',
                                                                legend=['train_loss', 'test_loss']))
    viz.line([0.,], [0.,], win='accuracy', opts=dict(title='accuracy',
                                                     legend=['accuracy']))
    print('*'*5 + 'PreTrained By {}'.format(datasetName) + '*'*5)
    main(datasetName)
    print('*'*5 + 'FINISH' + '*'*5)

