from torch.autograd import Variable
import numpy as np


def compress(train, test, encode_discrete, device):
    retrievalB = list([])
    retrievalL = list([])
    for batch_step, (data, target) in enumerate(train):
        var_data = Variable(data.to(device))
        code = encode_discrete(var_data)
        retrievalB.extend(code.cpu().data.numpy())
        retrievalL.extend(target)

    queryB = list([])
    queryL = list([])
    for batch_step, (data, target) in enumerate(test):
        var_data = Variable(data.to(device))
        code = encode_discrete(var_data)
        queryB.extend(code.cpu().data.numpy())
        queryL.extend(target)

    retrievalB = np.array(retrievalB)
    retrievalL = np.stack(retrievalL)

    queryB = np.array(queryB)
    queryL = np.stack(queryL)
    return retrievalB, retrievalL, queryB, queryL


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    q = B2.shape[1] # max inner product value
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def calculate_top_map(qB, rB, queryL, retrievalL, topk):
    """
    :param qB: {-1,+1}^{mxq} query bits
    :param rB: {-1,+1}^{nxq} retrieval bits
    :param queryL: {0,1}^{mxl} query label
    :param retrievalL: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind] # reorder gnd

        tgnd = gnd[0:topk]
        tsum = int(np.sum(tgnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap
