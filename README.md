# OutFlip
This is the implemenation of following paper:

DongHyun Choi, Myeong Cheol Shin, EungGyun Kim and Dong Ryeol Shin, OutFlip: Generating Examples for Unknown Intent Detection with Natural Language Attack, ACL Findings, 2021

## Requirements
tensorflow 2.0 <br>
nltk <br>

## How to Train with Sample training data
python main.py --config ./config/config_bilstm.json --store ./train_sample_model --train ./sample/train.txt --dev ./sample/dev.txt --test ./sample/test.txt --attack_num 2

## Contact
nlp.en@kakaoenterprise.com 
