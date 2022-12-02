## Gran-Pytorch
此项目是[GRAN: A Graph-based Approach to N-ary Relational Learning](https://github.com/PaddlePaddle/Research/tree/master/KG/ACL2021_GRAN)的Pytorch实现
 
The purpose of this project is to implement GRAN model in Pytorch [GRAN: A Graph-based Approach to N-ary Relational Learning](https://github.com/PaddlePaddle/Research/tree/master/KG/ACL2021_GRAN)

 
因为特定版本的paddlepaddle安装起来太麻烦了。
 
This is because a specific version of paddlepaddle is too cumbersome to install.

## usage
```shell
nohup python -u run.py --train_file ./data/jf17k/train.json --predict_file ./data/jf17k/test.json --ground_truth_path ./data/jf17k/all.json --vocab_path ./data/jf17k/vocab.txt --vocab_size 29148 --num_relations 501 --max_seq_len 11 --max_arity 6 --hidden_dropout_prob 0.2 --attention_dropout_prob 0.2 --entity_soft_label 0.1 --relation_soft_label 1.0 --epoch 160 > logs/jf17k.log &
```