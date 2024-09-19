## 说明

`mk_dataset` 记录了从原始数据到数据集的分析及转换过程。

原始数据集：https://www.yelp.com/dataset
- `yelp_dataset.tgz` SHA256：7196433b8a43dd1cbbc3054c5ee85447a56d5fdee8652f8dc7aa5aad579ad7cd
- `yelp_photos.tgz` SHA256：8b9fc60c64078f6db532bf9a26c65fe90cd1cfe72617c02df47c96d1755e964f  

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
- `core_items.txt`：`item_review_count.txt` 中评论数大于等于 10 的商家
- `re_core_items.txt`：`core_items.txt` 存在对应图片的商家
  - 格式：`busuiness_id` `review_count`
  - 数据条目：33198
- `yelp_interactions(_same).txt`：根据 `core_users/core_items` 从 `yelp_academic_dataset_review.json` 中提取的 user-item 数据
  - 格式：`user_id` `business_id` `[rating(stars)]`
  - 数据条目：2989981
- `re_yelp_interactions.txt`：`yelp_interactions.txt` 过滤掉**无图片**的商家
  - 数据条目：2041668
- `filter_yelp_interactions.txt`：`re_yelp_interactions.txt` 过滤掉**有图片但无法加载**的商家
  - 数据条目：2041590
- `photos.txt`：所有图片的文件名
  - 格式：`photo_id.jpg`
  - 数据条目：200098
- `item2photos.txt`：从 `yelp_interactions.txt` 中提取的映射关系
  - 格式：`business_id` `photo_id1` `photo_id2` ...
  - 数据条目：33190 
- `re_item2photos.txt`：从 `item2photos.txt` 过滤掉无法加载的图片
  - 数据条目：33183