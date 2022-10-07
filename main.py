#!/usr/bin/env python
# encoding: utf-8
"""
@project: PCAWSD
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.utils.data as pydata
from torch import nn
import torch.nn.functional as F
from models.model_CoordASPPCCA4_atten import PCAWSD, PCAWSD_NoSPP
import datetime
import random
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import preprocessing
from H_datapy import *
from autis import *


def main():
    samples_type = ['ratio', 'same_num'][0]
    model = 'PCAWSD'
    alpha = 0.01
    T = 4.0

    for (FLAG, curr_train_ratio) in [(1, 0.1)]:
        OA, AA, AC, Kappa, Confusion = [], [], [], [], []
        curr_seed = 0
        runs = 10

        if FLAG == 1:
            data_mat = sio.loadmat('./Datasets/IndianPines/Indian_pines_corrected.mat')
            data = data_mat['indian_pines_corrected']
            gt_mat = sio.loadmat('./Datasets/IndianPines/Indian_pines_gt.mat')
            gt = gt_mat['indian_pines_gt']
            label_names = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                           "Corn", "Grass-pasture", "Grass-trees",
                           "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                           "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                           "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                           "Stone-Steel-Towers"]
            # 参数预设
            class_count = 16  # 样本类别数
            learning_rate = 8e-4  # 学习率
            max_epoch = 600  # 迭代次数
            split_height = 1
            split_width = 1
            dataset_name = "indian"  # 数据集名称
            pass

        if FLAG == 2:
            data_mat = sio.loadmat('./Datasets/PaviaU/PaviaU.mat')
            data = data_mat['paviaU']
            gt_mat = sio.loadmat('./Datasets/PaviaU/PaviaU_gt.mat')
            gt = gt_mat['paviaU_gt']
            label_names = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                           'Painted metal sheets', 'Bare Soil', 'Bitumen',
                           'Self-Blocking Bricks', 'Shadows']

            # 参数预设
            class_count = 9  # 样本类别数
            learning_rate = 8e-4  # 学习率
            max_epoch = 600  # 迭代次数
            split_height = 3
            split_width = 2
            dataset_name = "paviaU"  # 数据集名称
            pass

        if FLAG == 3:
            data_mat = sio.loadmat('./Datasets/KSC/KSC.mat')
            data = data_mat['KSC']
            gt_mat = sio.loadmat('./Datasets/KSC/KSC_gt.mat')
            gt = gt_mat['KSC_gt']
            label_names = ["Undefined", "Scrub", "Willow swamp",
                           "Cabbage palm hammock", "Cabbage palm/oak hammock",
                           "Slash pine", "Oak/broadleaf hammock",
                           "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                           "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]

            # 参数预设
            class_count = 13  # 样本类别数
            learning_rate = 8e-4  # 学习率
            max_epoch = 600  # 迭代次数
            dataset_name = "KSC_BF"  # 数据集名称
            split_height = 1
            split_width = 1
            pass
        ###########
        train_samples_per_class = curr_train_ratio  # 当定义为每类样本个数时,则该参数更改为训练样本数
        train_ratio = curr_train_ratio
        if split_height == split_width == 1:
            EDGE = 0
        else:
            EDGE = 5

        ######################################################################
        # 存储位置
        time1 = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        log_dir = os.path.join('Results', model, dataset_name, str(time1))

        # 新建模型保存地址
        model_dir = os.path.join(log_dir, 'model')
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        # 新建运行结果保存地址
        record_name = os.path.join(log_dir, 'result_report.txt')
        #########################################################################################

        cmap = cm.get_cmap('jet', class_count + 1)
        plt.set_cmap(cmap)
        m, n, d = data.shape  # 高光谱数据的三个维度
        n_bands = d

        data = np.reshape(data, [m * n, d])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        data = np.reshape(data, [m, n, d])

        N_class = len(label_names)
        weights = compute_inf_weights(gt, ignored_classes=[0], n_classes=N_class)
        weights = torch.from_numpy(weights)
        weights = torch.tensor(weights, dtype=torch.float32).cuda()
        for run in range(runs):
            net_save_path = os.path.join(model_dir,
                                         '{}.best_model_run{}.hdf5'.format(time1, str(run)))
            teachernet_save_path = os.path.join(model_dir,
                                                '{}.best_teachermodel_run{}.hdf5'.format(time1, str(run)))
            # 随机10%数据作为训练样本。方式：给出训练数据与测试数据的GT
            random.seed(curr_seed)
            gt_reshape = np.reshape(gt, [-1])
            train_rand_idx = []
            if samples_type == 'ratio':
                for i in range(class_count):
                    idx = np.where(gt_reshape == i + 1)[-1]
                    samplesCount = len(idx)
                    rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
                    rand_idx = random.sample(rand_list,
                                             np.ceil(samplesCount * train_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
                    rand_real_idx_per_class = idx[rand_idx]
                    train_rand_idx.append(rand_real_idx_per_class)
                train_rand_idx = np.array(train_rand_idx)
                train_data_index = []
                for c in range(train_rand_idx.shape[0]):
                    a = train_rand_idx[c]
                    for j in range(a.shape[0]):
                        train_data_index.append(a[j])
                train_data_index = np.array(train_data_index)

                ##将测试集（所有样本，包括训练样本）也转化为特定形式
                train_data_index = set(train_data_index)
                all_data_index = [i for i in range(len(gt_reshape))]
                all_data_index = set(all_data_index)

                # 背景像元的标签
                background_idx = np.where(gt_reshape == 0)[-1]
                background_idx = set(background_idx)
                test_data_index = all_data_index - train_data_index - background_idx

                # 将训练集 测试集 整理
                test_data_index = list(test_data_index)
                train_data_index = list(train_data_index)

            if samples_type == 'same_num':
                for i in range(class_count):
                    idx = np.where(gt_reshape == i + 1)[-1]
                    samplesCount = len(idx)
                    real_train_samples_per_class = train_samples_per_class
                    rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
                    if real_train_samples_per_class > samplesCount:
                        real_train_samples_per_class = int(train_samples_per_class / 2)
                    rand_idx = random.sample(rand_list,
                                             real_train_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
                    rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
                    train_rand_idx.append(rand_real_idx_per_class_train)
                train_rand_idx = np.array(train_rand_idx)
                train_data_index = []
                for c in range(train_rand_idx.shape[0]):
                    a = train_rand_idx[c]
                    for j in range(a.shape[0]):
                        train_data_index.append(a[j])
                train_data_index = np.array(train_data_index)

                train_data_index = set(train_data_index)
                all_data_index = [i for i in range(len(gt_reshape))]
                all_data_index = set(all_data_index)

                # 背景像元的标签
                background_idx = np.where(gt_reshape == 0)[-1]
                background_idx = set(background_idx)
                test_data_index = all_data_index - train_data_index - background_idx

                # 将训练集 测试集 整理
                test_data_index = list(test_data_index)
                train_data_index = list(train_data_index)

            # 获取训练样本的标签图
            train_samples_gt = np.zeros(gt_reshape.shape)
            for i in range(len(train_data_index)):
                train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
                pass
            Train_Label = np.reshape(train_samples_gt, [m, n])

            # 获取测试样本的标签图
            test_samples_gt = np.zeros(gt_reshape.shape)
            for i in range(len(test_data_index)):
                test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
                pass

            Test_Label = np.reshape(test_samples_gt, [m, n])  # 测试样本图

            #############将train 和 test 样本标签转化为向量形式###################
            # 训练集
            train_samples_gt = np.reshape(train_samples_gt, [m * n])
            train_samples_gt_vector = np.zeros([m * n, class_count], np.float)
            for i in range(train_samples_gt.shape[0]):
                class_idx = train_samples_gt[i]
                if class_idx != 0:
                    temp = np.zeros([class_count])
                    temp[int(class_idx - 1)] = 1
                    train_samples_gt_vector[i] = temp
            train_samples_gt_vector = np.reshape(train_samples_gt_vector, [m, n, class_count])
            # 测试集
            test_samples_gt = np.reshape(test_samples_gt, [m * n])
            test_samples_gt_vector = np.zeros([m * n, class_count], np.float)
            for i in range(test_samples_gt.shape[0]):
                class_idx = test_samples_gt[i]
                if class_idx != 0:
                    temp = np.zeros([class_count])
                    temp[int(class_idx - 1)] = 1
                    test_samples_gt_vector[i] = temp
            test_samples_gt_vector = np.reshape(test_samples_gt_vector, [m, n, class_count])

            ############制作训练数据和测试数据的gt掩膜.根据GT将带有标签的像元设置为全1向量##############
            # 训练集
            train_label_mask = np.zeros([m * n, class_count])
            temp_ones = np.ones([class_count])
            train_samples_gt = np.reshape(train_samples_gt, [m * n])
            for i in range(m * n):
                if train_samples_gt[i] != 0:
                    train_label_mask[i] = temp_ones
            train_label_mask = np.reshape(train_label_mask, [m, n, class_count])

            # 测试集
            test_label_mask = np.zeros([m * n, class_count])
            temp_ones = np.ones([class_count])
            test_samples_gt = np.reshape(test_samples_gt, [m * n])
            for i in range(m * n):
                if test_samples_gt[i] != 0:
                    test_label_mask[i] = temp_ones
            test_label_mask = np.reshape(test_label_mask, [m, n, class_count])

            test_mask = np.zeros_like(Test_Label)
            test_mask[Test_Label > 0] = True
            test_mask = torch.from_numpy(test_mask)
            test_mask = test_mask.numpy().astype(np.bool)

            Train_Split_Data, Train_Split_GT = SpiltHSI(data, Train_Label, [split_height, split_width], EDGE)
            Test_Split_Data, Test_Split_GT = SpiltHSI(data, Test_Label, [split_height, split_width], EDGE)
            _, patch_height, patch_width, bands = Train_Split_Data.shape
            patch_height -= EDGE * 2
            patch_width -= EDGE * 2

            train_h = HData((np.transpose(Train_Split_Data, (0, 3, 1, 2)).astype("float32"), Train_Split_GT), None)
            test_h = HData((np.transpose(Test_Split_Data, (0, 3, 1, 2)).astype("float32"), Test_Split_GT), None)
            trainloader = torch.utils.data.DataLoader(train_h)
            testloader = torch.utils.data.DataLoader(test_h)

            # PCAWSD模型
            teacher_model = PCAWSD(class_count, n_bands, 150)
            student_model = PCAWSD(class_count, n_bands, 150)
            # # 消融实验：PCAWSD_NoSPP模型
            # teacher_model = PCAWSD_NoSPP(class_count, n_bands, 150)
            # student_model = PCAWSD_NoSPP(class_count, n_bands, 150)

            use_cuda = torch.cuda.is_available()
            if use_cuda: teacher_model.cuda()
            if use_cuda: student_model.cuda()
            criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=0)
            softmax = nn.Softmax(dim=1).cuda()
            logsoftmax = nn.LogSoftmax(dim=1).cuda()
            teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=learning_rate, weight_decay=2e-5)
            student_optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=2e-5)

            best_teacherEpoch, best_teacherOA, best_teacherAA = -1, -1, -1
            best_OA, best_AA, best_AC, best_kappa, best_eep, best_rightNum, best_testNum = -1, -1, -1, -1, -1, 0, 0
            method = 'kd-0'

            print("teacher_model 生成中........")
            for eep in range(max_epoch):
                TeacherOutput = []
                for batch_idx, (inputs, labels) in enumerate(trainloader):  # batch_idx是enumerate（）函数自带的索引，从0开始
                    if use_cuda:
                        inputs, labels = inputs.cuda(), labels.cuda()
                    inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
                    teacher_optimizer.zero_grad()
                    teacher_output = teacher_model(inputs)
                    teacher_loss = criterion(teacher_output, labels.long())
                    Teacheroutput = teacher_output.data.cpu().numpy()
                    Teacheroutput = np.transpose(Teacheroutput, (0, 2, 3, 1))
                    TeacherOutput.append(Teacheroutput[0])
                    teacher_optimizer.zero_grad()
                    teacher_loss.backward()
                    teacher_optimizer.step()

                TeacherOutputWhole, _ = PatchStack(TeacherOutput, m, n, patch_height, patch_width, split_height,
                                                   split_width, EDGE,
                                                   class_count + 1)
                teacher_AC, teacher_OA, teacher_AA, teacher_rightNum, teacher_testNum, teacher_kappa, teacher_confusion = \
                    ClassificationAccuracy(TeacherOutputWhole, Test_Label, class_count + 1)

                if eep % 10 == 0:
                    print('teacher_Epoch: {}\t'
                              'teacher_OA: {:.5f}\t'
                              'teacher_AA: {:.5f}\t'
                              "teacher_loss: {:.5f}\t".format(eep, teacher_OA, teacher_AA, teacher_loss))

                if 0.55 * teacher_OA + 0.45 * teacher_AA > 0.55 * best_teacherOA + 0.45 * best_teacherAA:
                    best_teacherOA = teacher_OA
                    best_teacherAA = teacher_AA
                    best_teacherEpoch = eep
                    torch.save(teacher_model, teachernet_save_path)
                if teacher_loss.data <= 0.00005:
                    break
            print("best train:\n"
                      'teacher_Epoch: {}\t'
                      'teacher_OA: {:.5f}\t'
                      'teacher_AA: {:.5f}\t'.format(best_teacherEpoch, best_teacherOA, best_teacherAA))
            print("teacher_model 生成!")

            for eep in range(max_epoch):
                for batch_idx, (inputs, labels) in enumerate(trainloader):  # batch_idx是enumerate（）函数自带的索引，从0开始
                    if use_cuda:
                        inputs, labels = inputs.cuda(), labels.cuda()
                    inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)

                    if method == 'kd-0':
                        teacher_model = torch.load(teachernet_save_path)
                        teacher_model.eval()
                        with torch.no_grad():
                            teacher_logits = teacher_model(inputs)
                            teacher_logits = teacher_logits.transpose(dim0=0, dim1=2).contiguous()
                            teacher_logits = teacher_logits.reshape([-1, class_count + 1])  # IP
                        student_model.train()
                        student_optimizer.zero_grad()
                        student_output = student_model(inputs)
                        student_output_ = student_output.transpose(dim0=1, dim1=3).contiguous()
                        student_output_ = student_output_.reshape([-1, class_count + 1])  # IP
                        dk_loss = F.kl_div(logsoftmax(student_output_ / T), softmax(teacher_logits / T),
                                           reduction='batchmean')
                        ce_loss = criterion(student_output, labels.long())
                        student_loss = (1.0 - alpha) * ce_loss + alpha * dk_loss * T * T
                        student_optimizer.zero_grad()
                        student_loss.backward()
                        student_optimizer.step()
                    else:
                        raise ValueError('val Err.')

                if eep % 10 == 0:
                    Output = []
                    for Testbatch_idx, (Testinputs, Testtargets) in enumerate(testloader):  # batch_idx是enumerate（）函数自带的索引，从0开始
                        if use_cuda:
                            Testinputs, Testtargets = Testinputs.cuda(), Testtargets.cuda()
                        Testinputs, Testtargets = torch.autograd.Variable(Testinputs), torch.autograd.Variable(Testtargets)
                        Testoutput = student_model(Testinputs)
                        Testoutput = Testoutput.data.cpu().numpy()
                        Testoutput = np.transpose(Testoutput, (0, 2, 3, 1))
                        Output.append(Testoutput[0])
                    OutputWhole, y_test = PatchStack(Output, m, n, patch_height, patch_width, split_height, split_width,
                                                     EDGE,class_count + 1)

                    teAC, teOA, teAA, terightNum, tetestNum, tekappa, teconfusion = ClassificationAccuracy(OutputWhole,
                                                                                                           Test_Label,
                                                                                                           class_count + 1)
                    print('Epoch [{}/{}]\t'
                              "test [{}/{}]\t"
                              'OA {:.5f}\t'
                              'AA {:.5f}\t'
                              'kappa {:.5f}\t'
                              'TrLoss {:.5f}\n'
                              "AC {}\t".format(eep, max_epoch, terightNum, tetestNum, teOA, teAA,
                                               tekappa, student_loss, teAC))

                    if 0.55 * teOA + 0.45 * teAA > 0.55 * best_OA + 0.45 * best_AA:
                        best_OA = teOA
                        best_AA = teAA
                        best_AC = teAC
                        best_kappa = tekappa
                        best_rightNum = terightNum
                        best_testNum = tetestNum
                        best_confusion = teconfusion
                        best_Output = OutputWhole
                        best_epoch = eep
                if student_loss.data <= 0.00005:
                    break

            print('The best model result at run {} is\n'
                      'Epoch: {}\t'
                      "test: [{}/{}]\t"
                      'OA: {:.5f}\t'
                      'AA: {:.5f}\t'
                      'kappa: {:.5f}\t'
                      'TrLoss: {:.5f}\n'
                      "AC:{}\n".format(run, best_epoch, best_rightNum, best_testNum, best_OA, best_AA,
                                       best_kappa,student_loss,best_AC))
            AC.append(best_AC)
            OA.append(best_OA)
            AA.append(best_AA)
            Kappa.append(best_kappa)
            Confusion = best_confusion
            Output_best = best_Output

            Output_best = np.reshape(Output_best, [-1])
            Output_best[list(background_idx)] = 0

        record_output(record_name, OA, AA, Kappa, AC, Confusion)

main()
