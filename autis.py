import torch
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda')
else:
    device = torch.device('cpu')


def SpiltHSI(data, gt, split_size, edge):
    '''
    split HSI data with given slice_number
    :param data: 3D HSI data
    :param gt: 2D ground truth
    :param split_size: [height_slice,width_slice]
    :return: splited data and corresponding gt
    '''
    e = edge  # 补边像素个数

    split_height = split_size[0]
    split_width = split_size[1]
    m, n, d = data.shape
    GT = gt

    # 将无法整除的块补0变为可整除
    if m % split_height != 0 or n % split_width != 0:
        data = np.pad(data, [[0, split_height - m % split_height], [0, split_width - n % split_width], [0, 0]],
                      mode='constant')
        GT = np.pad(GT, [[0, split_height - m % split_height], [0, split_width - n % split_width]],
                    mode='constant')
    m_height = int(data.shape[0] / split_height)
    m_width = int(data.shape[1] / split_width)

    pad_data = np.pad(data, [[e, e], [e, e], [0, 0]], mode="constant")
    pad_GT = np.pad(GT, [[e, e], [e, e]], mode="constant")
    final_data = []
    final_gt = []
    for i in range(split_height):
        for j in range(split_width):
            temp1 = pad_data[i * m_height:i * m_height + m_height + 2 * e, j * m_width:j * m_width + m_width + 2 * e, :]
            temp2 = pad_GT[i * m_height:i * m_height + m_height + 2 * e, j * m_width:j * m_width + m_width + 2 * e]
            final_data.append(temp1)
            final_gt.append(temp2)
    final_data = np.array(final_data)
    final_gt = np.array(final_gt)

    return final_data, final_gt


def PatchStack(OutPut, m, n, patch_height, patch_width, split_height, split_width, EDGE, class_count):
    HSI_stack = np.zeros([split_height * patch_height, split_width * patch_width, class_count], dtype=np.float32)
    for i in range(split_height):
        for j in range(split_width):
            if EDGE == 0:
                HSI_stack[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :] = OutPut[
                                                                                                                   i * split_width + j][
                                                                                                               EDGE:,
                                                                                                               EDGE:,
                                                                                                               :]
            else:
                HSI_stack[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :] = OutPut[
                                                                                                                   i * split_width + j][
                                                                                                               EDGE:-EDGE,
                                                                                                               EDGE:-EDGE,
                                                                                                               :]
    if EDGE != 0:
        HSI_stack = HSI_stack[0: -(split_height - m % split_height), 0: -(split_width - n % split_width)]
    y = HSI_stack
    HSI_stack = np.argmax(HSI_stack, axis=2)
    return HSI_stack, y


def compute_inf_weights(ground_truth, ignored_classes=[], n_classes=None):
    n_classes = np.max(ground_truth) if n_classes is None else n_classes
    weights = np.zeros(n_classes)
    frequencies = np.zeros(n_classes)
    for c in range(0, n_classes):
        if c in ignored_classes:
            continue
        frequencies[c] = np.count_nonzero(ground_truth == c)
    frequencies /= np.sum(frequencies)
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights = median / frequencies
    return weights



def ClassificationAccuracy(output, target, classcount):
    m, n = output.shape
    test_pre_label_list = []
    test_real_label_list = []
    correct_perclass = np.zeros([classcount - 1])
    count_perclass = np.zeros([classcount - 1])
    count = 0
    aa = 0

    correct_perclass = np.zeros([classcount - 1])

    for i in range(m):
        for j in range(n):
            if target[i, j] != 0:
                test_pre_label_list.append(output[i, j])
                test_real_label_list.append(target[i, j])
                count = count + 1
                count_perclass[int(target[i, j] - 1)] += 1
                if output[i, j] == target[i, j]:
                    aa = aa + 1
                    correct_perclass[int(target[i, j] - 1)] += 1

    test_AC_list = correct_perclass / count_perclass
    test_AA = np.average(test_AC_list)
    test_OA = aa / count
    Output = output.reshape(-1)
    Target = target.reshape(-1)
    confusion = confusion_matrix(Target, Output)
    test_pre_label_list = np.array(test_pre_label_list)
    test_real_label_list = np.array(test_real_label_list)
    kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16), test_real_label_list.astype(np.int16))
    return test_AC_list, test_OA, test_AA, aa, count, kappa, confusion


def record_output(file_name, oa_ae, aa_ae, kappa_ae, element_acc_ae, confusion, train_time=None, test_time=None):
    f = open(file_name, 'a')
    sentence0 = 'OAs for each iteration are:' + str(oa_ae) + '\n'
    f.write(sentence0)
    sentence1 = 'AAs for each iteration are:' + str(aa_ae) + '\n'
    f.write(sentence1)
    sentence2 = 'KAPPAs for each iteration are:' + str(kappa_ae) + '\n' + '\n'
    f.write(sentence2)
    sentence3 = 'mean_OA ± std_OA is: ' + str(np.mean(oa_ae)) + ' ± ' + str(np.std(oa_ae)) + '\n'
    f.write(sentence3)
    sentence4 = 'mean_AA ± std_AA is: ' + str(np.mean(aa_ae)) + ' ± ' + str(np.std(aa_ae)) + '\n'
    f.write(sentence4)
    sentence5 = 'mean_KAPPA ± std_KAPPA is: ' + str(np.mean(kappa_ae)) + ' ± ' + str(np.std(kappa_ae)) + '\n'
    f.write(sentence5)

    element_mean = np.mean(element_acc_ae, axis=0)
    element_std = np.std(element_acc_ae, axis=0)

    element_mean = ['{:.6f}'.format(x) for x in element_mean]
    sentence8 = "Mean of all elements in confusion matrix: " + str(element_mean) + '\n'
    element_std = ['{:.6f}'.format(x) for x in element_std]
    sentence9 = "Standard deviation of all elements in confusion matrix: " + str(element_std) + '\n'

    f.write(sentence8)
    f.write(sentence9)
    sentence10 = "confusion matrix: " + str(confusion) + '\n'
    f.write(sentence10)
    f.close()

