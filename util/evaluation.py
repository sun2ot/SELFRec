import math


class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        """
        计算推荐系统的命中数(hits): 推荐列表中有多少项在真实的用户行为数据中也被记录过
        
        Args:
            origin (dict): 原始用户行为数据(测试集)
            res (dict): 推荐系统的推荐结果
        
        Returns:
            hit_count (dict): 每个用户的命中数
        """
        hit_count = {}
        for user in origin:
            # 获取用户真实行为的项列表
            items = list(origin[user].keys())
            # 每个用户的推荐项 {user: [(item1, score1), (item2, score2), ...]}
            predicted = [item[0] for item in res[user]]
            # 计算推荐结果与真实行为数据的交集大小(命中数)
            hit_count[user] = len(set(items).intersection(set(predicted)))
        
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        """
        计算命中率: 在测试集中检索到的交互次数 / 测试集中的所有交互次数

        Args:
            origin (dict): 原始数据
            hits (dict): 每个用户命中数
    
        Returns:
            hit_ratio: 返回测试集中检索到的交互次数占所有交互次数的比例，保留五位小数
        """
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return round(hit_num/total_num, 5)

    # # @staticmethod
    # def hit_ratio(origin, hits):
    #     """
    #     Note: This type of hit ratio calculates the fraction:
    #      (# users who are recommended items in the test set / #all the users in the test set)
    #     """
    #     hit_num = 0
    #     for user in hits:
    #         if hits[user] > 0:
    #             hit_num += 1
    #     return hit_num / len(origin)

    @staticmethod
    def precision(hits, N):
        """
        计算精确度: 命中数总和/用户数量*top-N

        Args:
            hits (dict): 每个用户的命中数
            N (int): top-N

        Returns:
            float: 推荐系统的精确度，保留五位小数。
        """
        prec = sum([hits[user] for user in hits])
        return round(prec / (len(hits) * N), 5)

    @staticmethod
    def recall(hits, origin):
        """
        计算平均召回率

        Args:
            hits (dict): 每个用户的命中数
            origin (dict): 测试集

        Returns:
            float: 平均召回率，保留5位小数
        """
        # 计算每个用户的召回率: 命中数/真实行为数
        recall_list = [hits[user]/len(origin[user]) for user in hits]
        # 计算平均召回率: 召回率总和/用户数量
        recall = round(sum(recall_list) / len(recall_list), 5)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return round(2 * prec * recall / (prec + recall),5)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error+=abs(entry[2]-entry[3])
            count+=1
        if count==0:
            return error
        return round(error/count,5)

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3])**2
            count += 1
        if count==0:
            return error
        return round(math.sqrt(error/count),5)

    @staticmethod
    def NDCG(origin,res,N):
        """
        计算归一化折损累积增益(NDCG)

        Args:
            origin (dict): 测试集
            res (dict): 推荐结果
            N (int): top-N

        Returns:
            float: 平均NDCG值
        """
        # 初始化NDCG总和变量
        sum_NDCG = 0
        # 遍历推荐结果中的每个用户
        for user in res:
            # 初始化当前用户的DCG和IDCG值
            DCG = 0
            IDCG = 0
            # 对于每个用户的推荐列表，计算DCG值
            for n, item in enumerate(res[user]):
                # 如果推荐的项目测试集中，增加DCG值
                # item -> (item_id, score)
                if item[0] in origin[user]:
                    DCG += 1.0/math.log(n+2, 2)
            # 计算理想的DCG值（即相关项目按相关性排序时的DCG值）的前N个项
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG += 1.0/math.log(n+2, 2)

            # 将当前用户的NDCG值加到总和中
            sum_NDCG += DCG / IDCG

        # 返回平均NDCG值
        return round(sum_NDCG/len(res), 5)

    # @staticmethod
    # def MAP(origin, res, N):
    #     sum_prec = 0
    #     for user in res:
    #         hits = 0
    #         precision = 0
    #         for n, item in enumerate(res[user]):
    #             if item[0] in origin[user]:
    #                 hits += 1
    #                 precision += hits / (n + 1.0)
    #         sum_prec += precision / min(len(origin[user]), N)
    #     return sum_prec / len(res)

    # @staticmethod
    # def AUC(origin, res, rawRes):
    #
    #     from random import choice
    #     sum_AUC = 0
    #     for user in origin:
    #         count = 0
    #         larger = 0
    #         itemList = rawRes[user].keys()
    #         for item in origin[user]:
    #             item2 = choice(itemList)
    #             count += 1
    #             try:
    #                 if rawRes[user][item] > rawRes[user][item2]:
    #                     larger += 1
    #             except KeyError:
    #                 count -= 1
    #         if count:
    #             sum_AUC += float(larger) / count
    #
    #     return float(sum_AUC) / len(origin)


def ranking_evaluation(origin, res, N):
    """
    通过计算top-N的各种评估指标来评估排名结果的质量

    Args:
        origin (dict): 真实结果(测试集)
        res (dict): 模型预测的结果集
        N (list): top-N 值

    Returns:
        measure (list): 逐元素对应逐行评估指标输出
    """
    measure = []
    for n in N:
        measure.append('Top ' + str(n) + '\n')
        # 取出预测结果的前N个(top-N)
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        # 初始化一个列表来存储当前 Top-N 长度下的评估指标
        indicators = []
        # 检查真实集与预测集中用户的数量是否匹配
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        hits = Metric.hits(origin, predicted)
        hr = Metric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        # F1 = Metric.F1(prec, recall)
        # indicators.append('F1:' + str(F1) + '\n')
        #MAP = Measure.MAP(origin, predicted, n)
        #indicators.append('MAP:' + str(MAP) + '\n')
        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        # AUC = Measure.AUC(origin,res,rawRes)
        # measure.append('AUC:' + str(AUC) + '\n')
        #! 注意这里是列表相加
        measure += indicators
    return measure

def rating_evaluation(res):
    measure = []
    mae = Metric.MAE(res)
    measure.append('MAE:' + str(mae) + '\n')
    rmse = Metric.RMSE(res)
    measure.append('RMSE:' + str(rmse) + '\n')
    return measure