# coding=utf-8

import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

import random
import argparse
import torch.nn as nn
import torchvision
from torchvision import transforms as T
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils import data
# from data import VehicleID_MC, VehicleID_All, id2name
from tqdm import tqdm
import matplotlib as mpl
from matplotlib.font_manager import *

from InitRepNet import InitRepNet

# 解决负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.sans-serif'] = ['SimHei']

# --------------------------------------
# VehicleID用于MDNet
class VehicleID_All(data.Dataset):
    def __init__(self,
                 root,
                 transforms=None,
                 mode='train'):
        """
        :param root:
        :param transforms:
        :param mode:
        """
        if not os.path.isdir(root):
            print('[Err]: invalid root.')
            return

        # 加载图像绝对路径和标签
        if mode == 'train':
            txt_f_path = root + '/attribute/train_all.txt'
        elif mode == 'test':
            txt_f_path = root + '/attribute/test_all.txt'

        if not os.path.isfile(txt_f_path):
            print('=> [Err]: invalid txt file.')
            return

        # 打开vid2TrainID和trainID2Vid映射
        vid2TrainID_path = root + '/attribute/vid2TrainID.pkl'
        trainID2Vid_path = root + '/attribute/trainID2Vid.pkl'
        if not (os.path.isfile(vid2TrainID_path) \
                and os.path.isfile(trainID2Vid_path)):
            print('=> [Err]: invalid vid, train_id mapping file path.')

        with open(vid2TrainID_path, 'rb') as fh_1, \
                open(trainID2Vid_path, 'rb') as fh_2:
            self.vid2TrainID = pickle.load(fh_1)
            self.trainID2Vid = pickle.load(fh_2)

        self.imgs_path, self.lables = [], []
        with open(txt_f_path, 'r', encoding='utf-8') as f_h:
            for line in f_h.readlines():
                line = line.strip().split()
                img_path = root + '/image/' + line[0] + '.jpg'
                if os.path.isfile(img_path):
                    self.imgs_path.append(img_path)

                    tr_id = self.vid2TrainID[int(line[3])]
                    label = np.array([int(line[1]),
                                      int(line[2]),
                                      int(tr_id)], dtype=int)
                    self.lables.append(torch.Tensor(label))

        assert len(self.imgs_path) == len(self.lables)
        print('=> total %d samples loaded in %s mode' % (len(self.imgs_path), mode))

        # 加载数据变换
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        """
        关于数据缩放方式: 先默认使用非等比缩放
        :param idx:
        :return:
        """
        img = Image.open(self.imgs_path[idx])

        # 数据变换, 灰度图转换成'RGB'
        if img.mode == 'L' or img.mode == 'I':  # 8bit或32bit灰度图
            img = img.convert('RGB')

        # 图像数据变换
        if self.transforms is not None:
            img = self.transforms(img)

        return img, self.lables[idx]

    def __len__(self):
        """
        :return:
        """
        return len(self.imgs_path)

class FocalLoss(nn.Module):
    """
    Focal loss: focus more on hard samples
    """

    def __init__(self,
                 gamma=0,
                 eps=1e-7):
        """
        :param gamma:
        :param eps:
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        """
        :param input:
        :param target:
        :return:
        """
        log_p = self.ce(input, target)
        p = torch.exp(-log_p)
        loss = (1.0 - p) ** self.gamma * log_p
        return loss.mean()


# --------------------------------------- methods
def get_predict_mc(output):
    """
    softmax归一化,然后统计每一个标签最大值索引
    :param output:
    :return:
    """
    # 计算预测值
    output = output.cpu()  # 从GPU拷贝出来
    pred_model = output[:, :6]
    pred_color = output[:, 6:]

    model_idx = pred_model.max(1, keepdim=True)[1]
    color_idx = pred_color.max(1, keepdim=True)[1]

    # 连接pred
    pred = torch.cat((model_idx, color_idx), dim=1)
    return pred


def count_correct(pred, label):
    """
    :param output:
    :param label:
    :return:
    """
    assert pred.size(0) == label.size(0)
    correct_num = 0
    for one, two in zip(pred, label):
        if torch.equal(one, two):
            correct_num += 1
    return correct_num


def count_attrib_correct(pred, label, idx):
    """
    :param pred:
    :param label:
    :param idx:
    :return:
    """
    assert pred.size(0) == label.size(0)
    correct_num = 0
    for one, two in zip(pred, label):
        if one[idx] == two[idx]:
            correct_num += 1
    return correct_num


# @TODO: 可视化分类结果...

def ivt_tensor_img(input,
                   title=None):
    """
    Imshow for Tensor.
    """
    input = input.numpy().transpose((1, 2, 0))

    # 转变数组格式 RGB图像格式：rows * cols * channels
    # 灰度图则不需要转换，只有(rows, cols)而不是（rows, cols, 1）
    # (3, 228, 906)   #  (228, 906, 3)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # 去标准化，对应transforms
    input = std * input + mean

    # 修正 clip 限制inp的值，小于0则=0，大于1则=1
    output = np.clip(input, 0, 1)

    # plt.imshow(input)
    # if title is not None:
    #     plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated

    return output


def viz_results(resume,
                data_root):
    """
    :param resume:
    :param data_root:
    :return:
    """
    color_dict = {'black': u'黑色',
                  'blue': u'蓝色',
                  'gray': u'灰色',
                  'red': u'红色',
                  'sliver': u'银色',
                  'white': u'白色',
                  'yellow': u'黄色'}

    test_set = VehicleID_All(root=data_root,
                             transforms=None,
                             mode='test')
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1)
    # use RepNet
    # net = RepNet(out_ids=10086,
    #              out_attribs=257).to(device)
    # use InitRepNet
    vgg16_pretrain = torchvision.models.vgg16(pretrained=True)
    net = InitRepNet(vgg_orig=vgg16_pretrain,
                     out_ids=15,
                     out_attribs=11).to(device)
    # print('=> Mix difference network:\n', net)

    # 从断点启动
    if resume is not None:
        if os.path.isfile(resume):
            # 加载模型
            net.load_state_dict(torch.load(resume,  map_location='cpu'))
            print('=> net resume from {}'.format(resume))
        else:
            print('=> [Err]: invalid resume path @ %s' % resume)

    # 测试模式
    net.eval()

    # 加载类别id映射和类别名称
    modelID2name_path = data_root + '/attribute/modelID2name.pkl'
    colorID2name_path = data_root + '/attribute/colorID2name.pkl'
    trainID2Vid_path = data_root + '/attribute/trainID2Vid.pkl'
    if not (os.path.isfile(modelID2name_path) and \
            os.path.isfile(colorID2name_path) and \
            os.path.isfile((trainID2Vid_path))):
        print('=> [Err]: invalid file.')
        return

    with open(modelID2name_path, 'rb') as fh_1, \
            open(colorID2name_path, 'rb') as fh_2, \
            open(trainID2Vid_path, 'rb') as fh_3:
        modelID2name = pickle.load(fh_1)
        colorID2name = pickle.load(fh_2)
        trainID2Vid = pickle.load(fh_3)

    # 测试
    print('=> testing...')
    for i, (data, label) in enumerate(test_loader):
        # 放入GPU.
        data, label = data.to(device), label.to(device).long()

        # 前向运算: 预测车型、车身颜色
        output_attrib = net.forward(X=data,
                                    branch=1,
                                    label=None)
        pred_mc = get_predict_mc(output_attrib).cpu()[0]
        pred_m_id, pred_c_id = pred_mc[0].item(), pred_mc[1].item()
        pred_m_name = modelID2name[pred_m_id]
        pred_c_name = colorID2name[pred_c_id]

        # 前向运算: 预测Vehicle ID
        output_id = net.forward(X=data,
                                branch=3,
                                label=label[:, 2])
        _, pred_tid = torch.max(output_id, 1)
        pred_tid = pred_tid.cpu()[0].item()
        pred_vid = trainID2Vid[pred_tid]

        # 获取实际result
        img_path = test_loader.dataset.imgs_path[i]
        img_name = os.path.split(img_path)[-1][:-4]

        result = label.cpu()[0]
        res_m_id, res_c_id, res_vid = result[0].item(), result[1].item(), \
                                      trainID2Vid[result[2].item()]
        res_m_name = modelID2name[res_m_id]
        res_c_name = colorID2name[res_c_id]

        # 图像标题
        title = 'pred: ' + pred_m_name + ' ' + color_dict[pred_c_name] \
                + ', vehicle ID ' + str(pred_vid) \
                + '\n' + 'resu: ' + res_m_name + ' ' + color_dict[res_c_name] \
                + ', vehicle ID ' + str(res_vid)
        print('=> result: ', title)

        # 绘图
        # img = ivt_tensor_img(data.cpu()[0])
        # fig = plt.figure(figsize=(6, 6))
        # plt.imshow(img)
        # plt.title(title)
        # plt.show()

# 获取每张测试图片对应的特征向量
def gen_feature_map(resume,
                    imgs_path,
                    batch_size=16):
    """
    根据图相对生成每张图象的特征向量, 映射: img_name => img_feature vector
    :param resume:
    :param imgs_path:
    :return:
    """
    net = RepNet(out_ids=10086,
                 out_attribs=257).to(device)
    print('=> Mix difference network:\n', net)

    # 从断点启动
    if resume is not None:
        if os.path.isfile(resume):
            # 加载模型
            # net.load_state_dict(torch.load(resume,  map_location='cpu'))
            net.load_state_dict(torch.load(resume))
            print('=> net resume from {}'.format(resume))
        else:
            print('=> [Err]: invalid resume path @ %s' % resume)

    # 图像数据变换
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])

    # load model, image and forward
    data, features = None, None
    for i, img_path in tqdm(enumerate(imgs_path)):
        # load image
        img = Image.open(img_path)

        # tuen to RGB
        if img.mode == 'L' or img.mode == 'I':  # 8bit或32bit灰度图
            img = img.convert('RGB')

        # image data transformations
        img = transforms(img)
        img = img.view(1, 3, 224, 224)

        if data is None:
            data = img
        else:
            data = torch.cat((data, img), dim=0)

        if data.shape[0] % batch_size == 0 or i == len(imgs_path) - 1:

            # collect a batch of image data
            data = data.to(device)

            output = net.forward(X=data,
                                 branch=5,
                                 label=None)

            batch_features = output.data.cpu().numpy()
            if features is None:
                features = batch_features
            else:
                features = np.vstack((features, batch_features))

            # clear a batch of images
            data = None

    # generate feature map
    feature_map = {}
    for i, img_path in enumerate(imgs_path):
        feature_map[img_path] = features[i]

    print('=> feature map size: %d' % (len(feature_map)))
    return feature_map


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    """
    :param y_score:
    :param y_true:
    :return:
    """
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        print('=> th: %.3f, acc: %.3f' % (th, acc))

        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


# 统计阈值和准确率: Vehicle ID数据集
def get_th_acc_VID(resume,
                   pair_set_txt,
                   img_dir,
                   batch_size=16):
    """
    :param resume:
    :param pair_set_txt:
    :param img_dir:
    :param batch_size:
    :return:
    """
    if not os.path.isfile(pair_set_txt):
        print('=> [Err]: invalid file.')
        return

    pairs, imgs_path = [], []
    with open(pair_set_txt, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            pair = line.strip().split()

            imgs_path.append(img_dir + '/' + pair[0] + '.jpg')
            imgs_path.append(img_dir + '/' + pair[1] + '.jpg')

            pairs.append(pair)

    print('=> total %d pairs.' % (len(pairs)))
    print('=> total %d image samples.' % (len(imgs_path)))
    imgs_path.sort()

    # generate feature dict
    feature_map = gen_feature_map(resume=resume,
                                  imgs_path=imgs_path,
                                  batch_size=batch_size)

    sims, labels = [], []
    for pair in pairs:
        img_path_1 = img_dir + '/' + pair[0] + '.jpg'
        img_path_2 = img_dir + '/' + pair[1] + '.jpg'
        sim = cosin_metric(feature_map[img_path_1],
                           feature_map[img_path_2])
        label = int(pair[2])
        sims.append(sim)
        labels.append(label)

    # 统计最佳阈值及其对应的准确率
    acc, th = cal_accuracy(sims, labels)
    print('=> best threshold: %.3f, accuracy: %.3f%%' % (th, acc * 100.0))
    return acc, th


# 统计阈值和准确率: Car Match数据集
def test_car_match_data(resume,
                        pair_set_txt,
                        img_root,
                        batch_size=16):
    """
    :param resume:
    :param pair_set_txt:
    :param batch_size:
    :return:
    """
    if not os.path.isfile(pair_set_txt):
        print('=> [Err]: invalid file.')
        return

    pairs, imgs_path = [], []
    with open(pair_set_txt, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = line.strip().split()

            imgs_path.append(img_root + '/' + line[0])
            imgs_path.append(img_root + '/' + line[1])

            pairs.append(line)

    print('=> total %d pairs.' % (len(pairs)))
    print('=> total %d image samples.' % (len(imgs_path)))
    imgs_path.sort()

    # 计算特征向量字典
    feature_map = gen_feature_map(resume=resume,
                                  imgs_path=imgs_path,
                                  batch_size=batch_size)

    # 计算所有pair的sim
    sims, labels = [], []
    for pair in pairs:
        img_path_1 = img_root + '/' + pair[0]
        img_path_2 = img_root + '/' + pair[1]
        sim = cosin_metric(feature_map[img_path_1],
                           feature_map[img_path_2])
        label = int(pair[2])
        sims.append(sim)
        labels.append(label)

    # 统计最佳阈值及其对应的准确率
    acc, th = cal_accuracy(sims, labels)
    print('=> best threshold: %.3f, accuracy: %.3f%%' % (th, acc * 100.0))
    return acc, th


def test_accuracy(net, data_loader):
    """
    测试VehicleID分类在测试集上的准确率
    :param net:
    :param data_loader:
    :return:
    """
    net.eval()  # 测试模式,前向计算

    num_correct = 0
    num_total = 0

    # 每个属性的准确率
    num_model = 0
    num_color = 0
    total_time = 0.0

    print('=> testing...')
    for data, label in data_loader:
        # 放入GPU.
        data, label = data.to(device), label.to(device).long()

        # 前向运算, 预测Vehicle ID
        output = net.forward(X=data,
                             branch=3,
                             label=label[:, 2])

        # 统计总数
        num_total += label.size(0)

        # 统计全部属性都预测正确正确数
        _, pred = torch.max(output.data, 1)
        batch_correct = (pred == label[:, 2]).sum().item()
        num_correct += batch_correct

    # test-set总的统计
    accuracy = 100.0 * float(num_correct) / float(num_total)
    print('=> test accuracy: {:.3f}%'.format(accuracy))

    return accuracy

def train(resume):
    """
    :param resume:
    :return:
    """
    # net = RepNet(out_ids=10086,
    #              out_attribs=257).to(device)

    vgg16_pretrain = torchvision.models.vgg16(pretrained=True)
    net = InitRepNet(vgg_orig=vgg16_pretrain,
                     out_ids=15,
                     out_attribs=11).to(device)

    # print('=> Mix difference network:\n', net)

    # whether to resume from checkpoint
    if resume is not None:
        if os.path.isfile(resume):
            # net.load_state_dict(torch.load(resume,  map_location='cpu'))  # 加载模型
            net.load_state_dict(torch.load(resume))  # 加载模型
            print('=> net resume from {}'.format(resume))
        else:
            print('=> [Err]: invalid resume path @ %s' % resume)

    # 数据集
    train_set = VehicleID_All(root='/home/duweixin/CNN-SVM/VehicleID_CBIR/vehicle_search_dwx',
                              transforms=None,
                              mode='train')
    test_set = VehicleID_All(root='/home/duweixin/CNN-SVM/VehicleID_CBIR/vehicle_search_dwx',
                             transforms=None,
                             mode='test')
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=16,
                                               shuffle=True,
                                               num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=16,
                                              shuffle=False,
                                              num_workers=4)

    # loss function
    loss_func_1 = torch.nn.CrossEntropyLoss().to(device)
    loss_func_2 = FocalLoss(gamma=2).to(device)

    # optimization function
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=1e-3,
                                momentum=9e-1,
                                weight_decay=1e-8)
    print('=> optimize all layers.')

    # start to train
    print('\nTraining...')
    net.train()  # train模式

    best_acc = 0.0
    best_epoch = 0

    print('=> Epoch\tTrain loss\tTrain acc\tTest acc')
    for epoch_i in range(30):

        epoch_loss = []
        num_correct = 0
        num_total = 0
        for batch_i, (data, label) in enumerate(train_loader):  # 遍历每一个batch
            # ------------- put data to device
            data, label = data.to(device), label.to(device).long()

            # ------------- clear gradients
            optimizer.zero_grad()

            # ------------- forward pass of 3 branches
            output_1 = net.forward(X=data, branch=1, label=None)
            output_2 = net.forward(X=data, branch=2, label=label[:, 2])
            output_3 = net.forward(X=data, branch=3, label=label[:, 2])

            # ------------- calculate loss
            # branch1 loss
            loss_m = loss_func_1(output_1[:, :6], label[:, 0])  # vehicle model
            loss_c = loss_func_1(output_1[:, 6:], label[:, 1])  # vehicle color
            loss_br1 = loss_m + loss_c

            # branch2 loss
            loss_br2 = loss_func_2(output_2, label[:, 2])

            # branch3 loss: Vehicle ID classification
            loss_br3 = loss_func_2(output_3, label[:, 2])

            # 加权计算总loss
            loss = 0.5 * loss_br1 + 0.5 * loss_br2 + 1.0 * loss_br3

            # ------------- statistics
            epoch_loss.append(loss.cpu().item())

            # count samples
            num_total += label.size(0)

            # statistics of correct number
            _, pred = torch.max(output_3.data, 1)
            batch_correct = (pred == label[:, 2]).sum().item()
            batch_acc = float(batch_correct) / float(label.size(0))
            num_correct += batch_correct

            # ------------- back propagation
            loss.backward()
            optimizer.step()

            iter_count = epoch_i * len(train_loader) + batch_i

            # output batch accuracy
            if iter_count % 10 == 0:
                print('=> epoch {} iter {:>4d}/{:>4d}'
                      ', total_iter {:>6d} '
                      '| loss {:>5.3f} | accuracy {:>.3%}'
                      .format(epoch_i + 1,
                              batch_i,
                              len(train_loader),
                              iter_count,
                              loss.item(),
                              batch_acc))

        # total epoch accuracy
        train_acc = float(num_correct) / float(num_total)
        print('=> epoch {} | average loss: {:.3f} | average accuracy: {:>.3%}'
              .format(epoch_i + 1,
                      float(sum(epoch_loss)) / float(len(epoch_loss)),
                      train_acc))

        # calculate test-set accuracy
        test_acc = test_accuracy(net=net,
                                 data_loader=test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch_i + 1

            # save model weights
            model_save_name = 'epoch_' + str(epoch_i + 1) + '.pth'
            torch.save(net.state_dict(),
                       '/home/duweixin/CNN-SVM/VehicleID_CBIR/vehicle_search_dwx/model_save_path/' + model_save_name)
            print('<= {} saved.'.format(model_save_name))

        print('\t%d \t%4.3f \t\t%4.2f%% \t\t%4.2f%%' %
              (epoch_i + 1,
               sum(epoch_loss) / len(epoch_loss),
               train_acc * 100.0,
               test_acc))
    print('=> Best accuracy at epoch %d, test accuaray %f' % (best_epoch, best_acc))


# 单张图片数据读取
class SingleImgSet(torch.utils.data.Dataset):
    def __init__(self, img_path, transforms=None):
        # 加载图像绝对路径和标签
        # 每个这个数据共有4条数据, 分别是标号和标签...
        # 0是图片的路径, 1 2 3 分别是图片的标签, 这个标签是3维的, 那么分别就会对应图片的模型 颜色以及rid
        if not os.path.isfile(img_path):
            print('=> [Error]: invalid txt file.')
            return
        self.imgs_path = []
        self.labels = []
        self.imgs_path.append(img_path)
        self.labels.append(random.randint(1, 10))
        # 写一个断言, 看标签的长度和图片的是否是保持一致的
        print('=> 图片路径读取完毕')
        # 如果没有要求, 就对数据进行统一的处理
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

    # 其实单张图片读取的时候就是一张图片,这里的idx, 固定给一个值0
    # 这里是读取数据
    # 单张数据
    def __getitem__(self, idx):
        """
        关于数据缩放方式: 先默认使用非等比缩放
        :param idx:
        :return:
        """
        idx = 0
        img = Image.open(self.imgs_path[idx])
        # 数据变换, 灰度图转换成'RGB'
        if img.mode == 'L' or img.mode == 'I':  # 8bit或32bit灰度图
            img = img.convert('RGB')

        # 图像数据变换
        if self.transforms is not None:
            img = self.transforms(img)

        return img, self.labels[idx]

    # 这里是返回数据长度
    def __len__(self):
        """
        :return:
        """
        return len(self.imgs_path)


# 每一个车辆都有一个属于自己的id,现在的任务是给出一个车辆,然后根据这个车辆,找出和他id一样的车辆
# 1. 读取图片, 这部分的工作应该是根据torch的规则讲图片进行读取
# 2. 加载模型, 让模型保持在一个热启动的状态, 不要反复进行启动,这样性能的损耗太大
# 3. 分析图片,得到车辆的id
# 4. 循环遍历图片, 并将和该车车辆id一致的图片进行处理
# 5. 为了方便进行操作, 关于图片的内容最好是防止在阿里云服务器中
def get_Vehicles_ids(resume, data_root, img_path):
    # 颜色字典
    color_dict = {'black': u'黑色', 'blue': u'蓝色', 'gray': u'灰色',
                  'red': u'红色', 'sliver': u'银色', 'white': u'白色', 'yellow': u'黄色'}

    single_img = SingleImgSet(img_path=img_path, transforms=None)
    # 根据图中的信息显示, 也就是说有4010条的数据在测试模型中,通过测试模式的形式, 我们将这个数据转化成了VehicleID_All的这样的一个类
    # 也就是说,我这边所使用的是这么多条数据的类
    # assert(1 == 2)
    test_loader = torch.utils.data.DataLoader(dataset=single_img, batch_size=1, shuffle=False, num_workers=1)
    # 这样相当于是把这个数据加载完毕了
    # *** 数据加载的部分其实还可以再进行优化, 直接转化成为单张图片的数据 ***

    # 加载网络, 分别是输出的id数据和属性,然后放在gpu上, 并将网络结构进行输出
    # net = RepNet(out_ids=10086, out_attribs=257).to(device)
    vgg16_pretrain = torchvision.models.vgg16(pretrained=True)
    net = InitRepNet(vgg_orig=vgg16_pretrain,
                     out_ids=15,
                     out_attribs=11).to(device)
    # print('=> Mix difference network:\n', net)

    # 从断点启动. 如果指定了训练好的模型的话,就将模型参数输入进去
    if resume is not None:
        print("加载训练好的模型epoch14...")
        if os.path.isfile(resume):
            # 加载模型
            net.load_state_dict(torch.load(resume,  map_location='cpu'))
            print('=> net resume from {}'.format(resume))
            print("模型加载完毕!")
        else:
            print('=> [Err]: invalid resume path @ %s' % resume)

    # 测试模式
    print("进入模型的测试模式")
    net.eval()
    # print("测试毕毕")

    # 加载类别id映射和类别名称, 这三个文件用于产生模型 颜色以及id的映射
    modelID2name_path = data_root + '/attribute/modelID2name.pkl'
    colorID2name_path = data_root + '/attribute/colorID2name.pkl'
    trainID2Vid_path = data_root + '/attribute/trainID2Vid.pkl'
    if not (os.path.isfile(modelID2name_path) and os.path.isfile(colorID2name_path) and os.path.isfile((trainID2Vid_path))):
        print('=> [Err]: invalid file.')
        return
    with open(modelID2name_path, 'rb') as fh_1, \
            open(colorID2name_path, 'rb') as fh_2, \
            open(trainID2Vid_path, 'rb') as fh_3:
        modelID2name = pickle.load(fh_1)
        colorID2name = pickle.load(fh_2)
        trainID2Vid = pickle.load(fh_3)

    # 测试
    print('=> testing...')
    # 这边加载了数据之后就开始进行预测了,分别是预测车型 颜色和车辆的id
    # 数据要通过testloder的形式加载出来. 为了完成这步的加载.可以先关闭模型在本地进行加载
    for i, (data, label) in enumerate(test_loader):
        # 放入GPU.
        print('放入GPU')
        data, label = data.to(device), label.to(device).long()
        # 前向运算: 预测车型、车身颜色, 其实感觉这部分的label可以加入到下一步的操作中
        print('前向运算1')
        output_attrib = net.forward(X=data, branch=1, label=None)
        pred_mc = get_predict_mc(output_attrib).cpu()[0]
        pred_m_id, pred_c_id = pred_mc[0].item(), pred_mc[1].item()
        pred_m_name = modelID2name[pred_m_id]
        pred_c_name = colorID2name[pred_c_id]

        print("前向运算2")
        # 前向运算: 预测Vehicle ID, 我在想, 我要是任意模仿一波单边的数据会出现什么情况呢
        # 在预测ID的时候将车辆ID传进去, 这个操作可也太迷了
        output_id = net.forward(X=data, branch=3, label=label)
        _, pred_tid = torch.max(output_id, 1)
        pred_tid = pred_tid.cpu()[0].item()
        pred_vid = trainID2Vid[pred_tid]

        # 图像标题
        title = 'predict: ' + pred_m_name + ' ' + color_dict[pred_c_name] + ', vehicle ID ' + str(pred_vid)
        print('=> result: ', title)
        # 绘图, 通过绘图显示出来
        img = ivt_tensor_img(data.cpu()[0])
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(title)
        plt.show()



if __name__ == '__main__':
    # -----------------------------------
    # train(resume=None)  # 从头开始训练
    # train(resume='/home/duweixin/CNN-SVM/VehicleID_CBIR/vehicle_search_dwx/model_save_path/epoch_9.pth')
    # -----------------------------------
    # viz_results(resume='/home/neousys/duweixin/scia_project/vehicle_search_dwx/model_save_path/epoch_9.pth',
    #             data_root='/home/neousys/duweixin/scia_project/vehicle_search_dwx')
    # print('=> Done.')
    # -----------------------------------
    parser = argparse.ArgumentParser('单张图片测试')
    parser.add_argument('--image_path', '-i', required=True)
    args = parser.parse_args()
    print("the image path is: ")
    print(args.image_path)

    # 本地路径
    resume = '/home/neousys/duweixin/scia_project/vehicle_search_dwx/model_save_path/epoch_9.pth'
    data_root = '/home/neousys/duweixin/scia_project/vehicle_search_dwx'

    # # 服务器路径
    # # resume = '/home/neousys/duweixin/scia_project/vehicle_search_dwx/model_save_path/epoch_9.pth'
    # # data_root = '/home/neousys/duweixin/scia_project/vehicle_search_dwx'

    get_Vehicles_ids(resume=resume, data_root=data_root, img_path=args.image_path)

    print('=> Done.')

