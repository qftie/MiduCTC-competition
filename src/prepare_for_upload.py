import json
import argparse
from src.corrector import Corrector
from src.evaluate import evaluate


def prepare_for_uploadfile(in_model_dir,
                           in_json_file, 
                           out_json_file='test_output.json',
                           iter_num=1):
        
    json_data_list = json.load(open(in_json_file, 'r', encoding='utf-8'))
    src_texts = [ json_data['source'] for json_data in json_data_list]
    corrector = Corrector(in_model_dir=in_model_dir)
    pred_texts = corrector(texts=src_texts)
    for i in range(iter_num-1):
        pred_texts = corrector(texts=pred_texts)
    output_json_data = [ {'id':json_data['id'], 'inference': pred_text} for json_data, pred_text in zip(json_data_list, pred_texts)]
    
    with open(in_model_dir+out_json_file, 'w', encoding='utf-8') as f:
        json.dump(output_json_data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model4upload_dir', type=str, default='', help='the diretory for the model used to generate the final submission')
    parser.add_argument('--in_test_json_file', type=str, default='')
    parser.add_argument('--iter_num', type=int, default=1)
    args = parser.parse_args()
    val_f1 = evaluate(in_model_dir = args.model4upload_dir,
                        json_data_file ='/148Dataset/data-tie.qianfeng/preliminary_a_data/preliminary_val.json')
    prepare_for_uploadfile(in_model_dir=args.model4upload_dir,
                            in_json_file=args.in_test_json_file,
                            iter_num=args.iter_num
                            )