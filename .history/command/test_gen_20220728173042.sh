cd .. && CUDA_VISIBLE_DEVICES=3 python -m src.prepare_for_upload \
--model4upload_dir '/home/phd-fan.weiquan2/works/ChineseCorection/MiduCTC-competition/model/ctc_train_2022Y07M26D22H-chinese-roberta-wwm-ext/epoch3,step1,testf1_40_11%,devf1_62_99%/' \
--in_test_json_file '/148Dataset/data-tie.qianfeng/preliminary_a_data/preliminary_a_test_source.json'