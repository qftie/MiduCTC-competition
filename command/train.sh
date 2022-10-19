cd .. && export CUDA_DEVICE_ORDER=PCI_BUS_ID && CUDA_VISIBLE_DEVICES=0,1,2,5 python -m src.train \
--in_model_dir "pretrained_model/ch_macbert_base_epoch5,step1,testf1_39_41%,devf1_67_26%" \
--out_model_dir "model/ctc_train" \
--epochs "10" \
--batch_size "64" \
--max_seq_len "256" \
--learning_rate "4e-5" \
--train_fp "data/final_train_fusion_stage2_3.json" \
--random_seed_num "43" \
--check_val_every_n_epoch "0.5" \
--early_stop_times "20" \
--warmup_steps "100" \
--dev_data_ratio "0.2" \
--training_mode "dp" \
--amp true \
--freeze_embedding false