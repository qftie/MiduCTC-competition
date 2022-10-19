cd .. && CUDA_VISIBLE_DEVICES=4 python -m src.prepare_for_upload \
--model4upload_dir '/home/phd-fan.weiquan2/works/ChineseCorection/MiduCTC-competition/model/ctc_train_2022Y07M27D10H-chinese-macbert-base/epoch10,step1,testf1_37_03%,devf1_70_19%/' \
--in_test_json_file '/148Dataset/data-tie.qianfeng/preliminary_a_data/preliminary_a_test_source.json'