## 运行环境

- Python 3.9.19
- mamba 1.5.8
- conda 24.3.0

pip 部署环境：

```bash
mamba create -n selfrec python=3.9
mamba activate selfrec
pip install -r requirements.txt
```

conda/mamba 部署环境：

```bash
conda env create -f environment.yml
```

> [!warning]
> requirements.txt 是根据 `mamba list` 的内容导出的，也许存在版本号差异的问题，请自行尝试解决。

## 一些迷思

### 关于predict

数据集划分训练、测试，这没什么好说的。但是，训练集训练后，可以得到里面的user, item嵌入；可对于测试集，你只能查到训练集中存在的用户embedding 和 item embedding。对于训练集中不存在的 user/item，因为并没有投入训练，自然也就不知道对应的嵌入了。

这就是为什么，predict中是 `score = torch.matmul(self.user_emb[user_id], self.item_emb.transpose(0, 1))`。

这里容易引起误解的点就在于，训练集有项目数 38048.这里的 self.item_emb -> torch.Size([38048, 64])，但是测试集中明明有 36073 个 item，这里却没有体现。

此时再回头看 `__generate_set()` 中的内容，"遍历测试数据，只为**训练数据中已有的用户和物品**添加评分记录至测试集"，这就是为什么测试集可以在没有对应item embedding的情况下，评估模型性能。

综上，测试模型时的测试集，靠的不是模型直接去计算出对未知item的评分，而是通过测试集中用户对特定物品的交互来验证模型的推荐效果。换句话说，测试集的作用是用来衡量模型在看过的物品以外的泛化能力，而不是用于直接生成推荐列表。