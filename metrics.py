def MSE(preds, true):
    squaredError = []
    for i in range(len(preds)):
        val = true[i] - preds[i]
        squaredError.append(val * val)  # target-prediction之差平方
    return sum(squaredError) / len(squaredError)


def MSE_ips(preds, true, user_num, item_num, inverse_propensity):
    squaredError = []
    globalNormalizer = 0
    for i in range(len(preds)):
        val = true[i] - preds[i]
        squaredError.append(val * val * inverse_propensity[int(true[i]) - 1])
        globalNormalizer += inverse_propensity[int(true[i]) - 1]
    # aaa = sum(squaredError) / globalNormalizer # SNIPS
    return sum(squaredError) / (user_num * item_num)


def MAE(preds, true):
    absError = []
    for i in range(len(preds)):
        val = true[i] - preds[i]
        absError.append(abs(val))  # 误差绝对值
    return sum(absError) / len(absError)


def MAE_ips(preds, true, user_num, item_num, inverse_propensity):
    absError = []
    for i in range(len(preds)):
        val = true[i] - preds[i]
        absError.append(abs(val) * inverse_propensity[int(true[i]) - 1])  # 误差绝对值
    return sum(absError) / (user_num * item_num)


def RMSE(preds, true):
    squaredError = []
    absError = []
    for i in range(len(preds)):
        val = true[i] - preds[i]
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值
    from math import sqrt
    return sqrt(sum(squaredError) / len(squaredError))


def RMSE_ips(preds, true, user_num, item_num, inverse_propensity):
    squaredError = []
    for i in range(len(preds)):
        val = true[i] - preds[i]
        squaredError.append(val * val * inverse_propensity[int(true[i]) - 1])
    from math import sqrt
    return sqrt(sum(squaredError) / (user_num * item_num))