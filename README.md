## Gran-Pytorch
此项目是[GRAN: A Graph-based Approach to N-ary Relational Learning](https://github.com/PaddlePaddle/Research/tree/master/KG/ACL2021_GRAN)的Pytorch实现
 
The purpose of this project is to implement GRAN model in Pytorch [GRAN: A Graph-based Approach to N-ary Relational Learning](https://github.com/PaddlePaddle/Research/tree/master/KG/ACL2021_GRAN)

 
因为特定版本的paddlepaddle安装起来太麻烦了。
 
This is because a specific version of paddlepaddle is too cumbersome to install.
 
这个模型在jf17k数据集上,运行58个epoch，entity情况如下: 

This model runs on the jf17k dataset with 58 epoch with entity correlation results as follows:
| mrr    | hits1  | hits3  | hits5  | hits10 |
| ------ | ------ | ------ | ------ | ------ |
| 0.6218 | 0.5412 | 0.6606 | 0.7115 | 0.7774 |

可以认为完全复刻了原模型。 

It can be argued that the original model is completely reproduced.

注意:如果batch size设置的过小，模型可能会不收敛，建议维持batch size=1024的设置 

Note: If the batch size setting is too small, the model may not converge. It is recommended to maintain the batch size=1024 setting
## usage
```shell
nohup python -u run.py --train_file ./data/jf17k/train.json --predict_file ./data/jf17k/test.json --ground_truth_path ./data/jf17k/all.json --vocab_path ./data/jf17k/vocab.txt --vocab_size 29148 --num_relations 501 --max_seq_len 11 --max_arity 6 --hidden_dropout_prob 0.2 --attention_dropout_prob 0.2 --entity_soft_label 0.1 --relation_soft_label 1.0 --epoch 160 > logs/jf17k.log &
```