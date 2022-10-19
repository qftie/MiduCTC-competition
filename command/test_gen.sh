cd .. && export CUDA_DEVICE_ORDER=PCI_BUS_ID && CUDA_VISIBLE_DEVICES=5 python -m src.kfolds_prepare_for_uploads \
--model4upload_dir 'model/ctc_train_2022Y08M14D15H/epoch8,step51,testf1_63_29%,devf1_51_68%/' \
--in_test_json_file 'data/final_test_source.json'