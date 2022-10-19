import json
import argparse
from src.corrector import Corrector
from src.evaluate import evaluate
from typing import List
import numpy as np

from src.baseline.predictor import PredictorCtc
from src.metric import final_f1_score

def prepare_for_uploadfile(in_model_dirs,
                           in_json_file, 
                           out_json_file='test_output.json',
                           to_evaluate = False):
        
    json_data_list = json.load(open(in_json_file, 'r', encoding='utf-8'))
    src_texts = [ json_data['source'] for json_data in json_data_list]
    pred_outputs_all=[]
    predictor = PredictorCtc(
        in_model_dirs=in_model_dirs,
        ctc_label_vocab_dir='src/baseline/ctc_vocab',
        use_cuda=True,
        cuda_id=None,
        )
    pred_outputs = predictor.predict(src_texts,batch_size=4)
    pred_texts = [PredictorCtc.output2text(output) for output in pred_outputs]
    if to_evaluate:
        f1_score = final_f1_score(src_texts=src_texts, 
                              pred_texts=pred_texts,
                              trg_texts=[json_data['target'] for json_data in json_data_list],
                              log_fp='logs/f1_score.log')
    output_json_data = [ {'id':json_data['id'], 'inference': pred_text} for json_data, pred_text in zip(json_data_list, pred_texts)]
    
    with open(out_json_file, 'w', encoding='utf-8') as f:
        json.dump(output_json_data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model4upload_dir', type=str, default='', help='the diretory for the model used to generate the final submission')
    parser.add_argument('--in_test_json_file', type=str, default='')
    parser.add_argument('--to_evaluate', action='store_true')
    args = parser.parse_args()
    prepare_for_uploadfile(in_model_dirs=[
        'model/ctc_train_2022Y08M18D11H/fold_2/epoch5,step1,testf1_None,devf1_54_24%/',
        'model/ctc_train_2022Y08M18D12H/fold_6/epoch5,step35,testf1_None,devf1_52_37%',
        ],
                            in_json_file=args.in_test_json_file,
                            to_evaluate= args.to_evaluate
                            )