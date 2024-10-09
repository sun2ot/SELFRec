from base.recommender import Recommender
from data.ui_graph import Interaction
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from data.loader import FileIO
import os
from os.path import abspath
from util.evaluation import ranking_evaluation
from dotenv import load_dotenv
from qywx_bot.bot import Bot

load_dotenv()
key = os.getenv('WEBHOOK_KEY')
if key is not None:
    bot: Bot = Bot(key)


class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super().__init__(conf, training_set, test_set, **kwargs)
        self.data = Interaction(conf, training_set, test_set, **kwargs)
        self.bestPerformance = []  # [epoch, performance{recall:0.0, precision:0.0, ...}]
        self.topN = [int(num) for num in self.ranking]  # 10, 20
        self.max_N = max(self.topN)

    def print_model_info(self):
        """重写父类方法输出模型配置及数据集统计信息"""
        super().print_model_info()
        # print dataset statistics
        print(f'Training Set Size: (user number: {self.data.training_size()[0]}, '
              f'item number: {self.data.training_size()[1]}, '
              f'interaction number: {self.data.training_size()[2]})')
        print(f'Test Set Size: (user number: {self.data.test_size()[0]}, '
              f'item number: {self.data.test_size()[1]}, '
              f'interaction number: {self.data.test_size()[2]})')
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        raise NotImplementedError

    def test(self):
        """
        测试模型

        Returns:
            rec_list (dict): 推荐列表 `{user: [(item1, score1), (item2, score2), ...]}`
        """
        def process_bar(num, total):
            """
            绘制CLI进度条

            Args:
                num (int): 当前完成的任务数量
                total (int): 任务的总数量
            """
            # 计算任务完成的比率
            rate = float(num) / total
            # 根据比率计算需要显示的进度条长度(最长50个字符)
            ratenum = int(50 * rate)
            print(f'\rProgress: [{"+" * ratenum}{" " * (50 - ratenum)}]{ratenum * 2}%', end='', flush=True)

        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            # 生成用户推荐候选
            candidates = self.predict(user)  # (38048,)
            #? 这是否意味着评分其实不影响预测结果
            #* ans: yes，因为压根没有用到，没看到变量名都是 _ 吗🤣
            rated_list, _ = self.data.user_rated(user)
            # 根据用户历史评分，排除已评分项目(赋极小值)
            #? 所以测试集里为什么会存在已评分项目
            #* ans: ui_graph.__generate_set() 中已过滤
            for item in rated_list:
                #* 以item_id为索引获取对应评分并修改
                candidates[self.data.item[item]] = -10e8
            # 找到评分最高的max_N个物品
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            # 存储推荐结果
            rec_list[user] = list(zip(item_names, scores))
            # 每处理1000个用户，更新进度条
            if i % 1000 == 0:
                process_bar(i, user_count)
        # 完成所有用户推荐后，进度条拉到100%(因为正常遍历是无法显示出100%的)
        process_bar(user_count, user_count)
        print('')
        return rec_list

    def evaluate(self, rec_list):
        #* 仅在所有训练epoch结束后执行一次
        """
        输出推荐指标及结果

        Args:
            rec_list (dict): 推荐列表 `{user: [(item1, score1), (item2, score2), ...]}`
        """
        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n')
        # 结果输出样例
        # 0: (771,3.333857297897339) (649,3.1552059650421143) (...)
        for user in self.data.test_set:
            line = user + ':' + ''.join(
                f" ({item[0]},{item[1]}){'*' if item[0] in self.data.test_set[user] else ''}"
                for item in rec_list[user]
            )
            line += '\n'
            self.recOutput.append(line)
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        out_dir = self.output
        file_name = f"{self.config['model']['name']}@{current_time}-top-{self.max_N}items.txt"
        FileIO.write_file(out_dir, file_name, self.recOutput)
        print('The result has been output to ', abspath(out_dir), '.')
        # 输出评估指标到文件
        file_name = f"{self.config['model']['name']}@{current_time}-performance.txt"
        self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        # 日志输出评估指标
        self.model_log.add('###Evaluation Results###')
        result_format_str = '\n'
        for r in self.result:
            result_format_str += r
        self.model_log.add(result_format_str)
        FileIO.write_file(out_dir, file_name, self.result)
        # CLI 输出评估指标
        print(f'The result of {self.model_name}:\n{"".join(self.result)}')
        bot.send_text(f'The result of {self.model_name}:\n{"".join(self.result)}')

    def fast_evaluation(self, epoch):
        """
        输出单轮评估指标并记录最佳性能

        Returns:
            measure (list): 逐元素对应逐行评估指标输出
        """
        print('Evaluating the model...')
        rec_list = self.test()
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])

        performance = {k: float(v) for m in measure[1:] for k, v in [m.strip().split(':')]}

        if self.bestPerformance:
            count = sum(1 if self.bestPerformance[1][k] > performance[k] else -1 for k in performance)
        # 如果存在之前的最佳性能记录
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            # 解析当前性能指标，存储到performance字典中
            #? measure[0]为日志，但是top-N值有两个，也就是说后面应该还有一个日志记录，如何处理的
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            # 比较当前性能和最佳性能，更新count值
            for k in self.bestPerformance[1]:
                # 如果当前性能指标小于最佳性能指标(损失函数值越小越好)，则count-1
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            # 通过count值粗略判断是否整体更优(即多数指标更优)
            if count < 0:
                self.bestPerformance = [epoch + 1, performance]
                #? 继承自父类的save函数是干嘛的
                self.save()
        else:
            self.bestPerformance = [epoch + 1, performance]
            # 不存在历史最佳性能记录，则直接保存
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            self.bestPerformance.append(performance)
            self.save()

        print('-' * 80)
        print(f'Real-Time Ranking Performance (Top-{self.max_N} Item Recommendation)')
        measure_str = ', '.join([f'{k}: {v}' for k, v in performance.items()])
        print(f'*Current Performance*\nEpoch: {epoch + 1}, {measure_str}')
        bp = ', '.join([f'{k}: {v}' for k, v in self.bestPerformance[1].items()])
        print(f'*Best Performance*\nEpoch: {self.bestPerformance[0]}, {bp}')
        print('-' * 80)
        return measure
