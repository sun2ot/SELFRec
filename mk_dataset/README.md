## 说明

`mk_dataset` 记录了从原始数据到数据集的分析及转换过程。

## 统计数据

- `user_review_count.txt`：`yelp_academic_dataset_user.json` 中的用户在 `yelp_academic_dataset_review.json` 中存在的评论数
  - 格式：`user_id` `review_count`
  - 数据条目: 1987897
- `core_users.txt`：`user_review_count.txt` 中评论数大于等于 10 的用户
  - 格式：`user_id` `review_count`
  - 数据条目: 117370
- `item_review_count.txt`：参考 `user_review_count.txt`
  - 格式： `busuiness_id` `review_count`
  - 数据条目：150346
- `core_items.txt`：参考 `core_users.txt`
  - 格式：`busuiness_id` `review_count`
  - 数据条目：33198
- `yelp_interactions.txt`：按照 10-core 设定从 `yelp_academic_dataset_review.json` 中提取的 user-item 数据
  - 格式：`user_id` `business_id` `rating(stars)`
  - 数据条目：2041668