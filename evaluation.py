def two_cls_access(reference,result):
    # for Hermiston dataset
    # reference:change_value=1;unchange_value=0
    # result: predicted map:change_value=1;unchange_value=0
    # 对二类变化检测的结果进行精度评价，指标为kappad系数和OA值
    # 输入：
    #      reference：二元变化reference(二值图，H*W)
    #      resultz:算法检测得到的二类变化结果图(二值图，H*W)]
    oa_kappa = []
    m,n = reference.shape
    if reference.shape != result.shape:
        print('the size of reference shoulf be equal to that of result')
        return oa_kappa
    reference = np.reshape(reference, -1)
    result = np.reshape(result, -1)
    label_0 = np.where(reference == 0)
    label_1 = np.where(reference == 1)
    predict_0 = np.where(result == 0)
    predict_1 = np.where(result == 1)
    label_0 = label_0[0]
    label_1 = label_1[0]
    predict_0 = predict_0[0]
    predict_1 = predict_1[0]
    tp = set(label_1).intersection(set(predict_1))  # True Positive
    tn = set(label_0).intersection(set(predict_0))  # False Positive
    fp = set(label_0).intersection(set(predict_1))  # False Positive
    fn = set(label_1).intersection(set(predict_0))  # False Negative

    precision = len(tp) / (len(tp) + len(fp))
    recall = len(tp) / (len(tp) + len(fn))

    precision = round(precision, 4)
    recall = round(recall, 4)
    F1 = 2 * (precision * recall) / (precision + recall)
    F1 = round(F1, 4)
    print('F1=   ' + str(F1))
    print('recall=   ' + str(recall))
    print('precision=   ' + str(precision))

    oa = (len(tp)+len(tn))/m/n      # Overall precision
    pe = (len(label_1)*len(predict_1)+len(label_0)*len(predict_0))/m/n/m/n
    kappa = (oa-pe)/(1-pe)
    oa = round(oa, 4)
    kappa = round(kappa, 4)
    oa_kappa.append('OA')
    oa_kappa.append(oa)
    oa_kappa.append('kappa')
    oa_kappa.append(kappa)
    oa_kappa.append('F1')
    oa_kappa.append(F1)
    oa_kappa.append('recall')
    oa_kappa.append(recall)
    oa_kappa.append('precision')
    oa_kappa.append(precision)

    print('OA:  ' + str(oa) + '    ' + 'kappa:  ' + str(kappa))
    return oa_kappa

def two_cls_access_for_Bay_Barbara(reference,result):
    # for Bay & Barbra datasets
    # reference:change_value=1;unchange_value=2
    # result: predicted map:change_value=1;unchange_value=0
    # 对二类变化检测的结果进行精度评价，指标为kappad系数和OA值
    # 输入：
    #      reference：二元变化reference(二值图，H*W), change=1; unchanged=2;uncertain=0
    #      resultz:算法检测得到的二类变化结果图(二值图，H*W)]
    oa_kappa = []
    # m,n = reference.shape
    if reference.shape != result.shape:
        print('the size of reference shoulf be equal to that of result')
        return oa_kappa
    reference = np.reshape(reference, -1)
    result = np.reshape(result, -1)

    label_0 = np.where(reference == 2)  # Unchanged
    label_1 = np.where(reference == 1)  # Changed
    predict_0 = np.where(result == 0)  # Unchanged
    predict_1 = np.where(result == 1)  # Changed
    label_0 = label_0[0]
    label_1 = label_1[0]
    predict_0 = predict_0[0]
    predict_1 = predict_1[0]
    tp = set(label_1).intersection(set(predict_1))  # True Positive
    tn = set(label_0).intersection(set(predict_0))  # True Negative
    fp = set(label_0).intersection(set(predict_1))  # False Positive
    fn = set(label_1).intersection(set(predict_0))  # False Negative

    precision = len(tp) / (len(tp) + len(fp))  # (预测为1且正确预测的样本数) / (所有真实情况为1的样本数)
    recall = len(tp) / (len(tp) + len(fn))  # (预测为1且正确预测的样本数) / (所有真实情况为1的样本数)

    precision = round(precision, 4)
    recall = round(recall, 4)
    F1 = 2 * (precision * recall) / (precision + recall)
    F1 = round(F1, 4)
    print('F1=   ' + str(F1))
    print('recall=   ' + str(recall))
    print('precision=   ' + str(precision))
    total_num = len(label_0) +len(label_1)
    oa = (len(tp) + len(tn)) / total_num  # Overall precision
    pe = ((len(tp)+len(fn))*(len(tp)+len(fp)) +(len(fp)+len(tn))*(len(fn)+len(tn)))/ total_num / total_num


    kappa = (oa-pe)/(1-pe)
    oa = round(oa, 4)
    kappa = round(kappa, 4)
    oa_kappa.append('OA')
    oa_kappa.append(oa)
    oa_kappa.append('kappa')
    oa_kappa.append(kappa)
    oa_kappa.append('F1')
    oa_kappa.append(F1)
    oa_kappa.append('recall')
    oa_kappa.append(recall)
    oa_kappa.append('precision')
    oa_kappa.append(precision)

    print('OA:  ' + str(oa) + '    ' + 'kappa:  ' + str(kappa))
    # print('whole OA is' + str(oa))
    # print('whole kappa is' + str(kappa))
    return oa_kappa
