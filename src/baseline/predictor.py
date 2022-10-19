#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os

import torch
from src import logger
from src.baseline.modeling import ModelingCtcBert
from src.baseline.tokenizer import CtcTokenizer


class PredictorCtc:
    def __init__(
        self,
        in_model_dirs,
        ctc_label_vocab_dir='src/baseline/ctc_vocab',
        use_cuda=True,
        cuda_id=None,
    ):

        # self.in_model_dir = in_model_dir
        self.models=[]
        for in_model_dir in in_model_dirs:
            model = ModelingCtcBert.from_pretrained(
                in_model_dir)
            self._id2dtag, self._dtag2id, self._id2ctag, self._ctag2id = self.load_label_dict(
                ctc_label_vocab_dir)
            logger.info('model loaded from dir {}'.format(
                in_model_dir))
            self.use_cuda = use_cuda
            if self.use_cuda and torch.cuda.is_available():
                if cuda_id is not None:
                    torch.cuda.set_device(cuda_id)
                model.cuda()
                model.half()
            model.eval()
            self.models.append(model)
        self.tokenizer = CtcTokenizer.from_pretrained(in_model_dir)

        try:
            self._start_vocab_id = self.tokenizer.vocab['[START]']
        except KeyError:
            self._start_vocab_id = self.tokenizer.vocab['[unused1]']

    def load_label_dict(self, ctc_label_vocab_dir):
        dtag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_detect_tags.txt')
        ctag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_correct_tags.txt')

        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}

        id2ctag = [line.strip() for line in open(ctag_fp, encoding='utf8')]
        c_tag2id = {v: i for i, v in enumerate(id2ctag)}
        logger.info('d_tag num: {}, d_tags:{}'.format(len(id2dtag), d_tag2id))
        return id2dtag, d_tag2id, id2ctag, c_tag2id

    def id_list2ctag_list(self, id_list) -> list:

        return [self._id2ctag[i] for i in id_list]

    @torch.no_grad()
    def predict(self, texts, return_topk=1, batch_size=4):
        if isinstance(texts, str):
            texts = [texts]
        else:
            texts = texts
        outputs = []
        correct_sent_num = 0
        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx+batch_size]

            batch_texts = [' ' + t for t in batch_texts]  # 开头加一个占位符
            inputs = self.tokenizer(batch_texts,
                                    return_tensors='pt')
            # 把 ' ' 换成 _start_vocab_id
            inputs['input_ids'][..., 1] = self._start_vocab_id
            if self.use_cuda and torch.cuda.is_available():
                inputs['input_ids'] = inputs['input_ids'].cuda()
                inputs['attention_mask'] = inputs['attention_mask'].cuda()
                inputs['token_type_ids'] = inputs['token_type_ids'].cuda()

            # d_preds, preds, loss = self.model(
            #     input_ids=inputs['input_ids'],
            #     attention_mask=inputs['attention_mask'],
            #     token_type_ids=inputs['token_type_ids'],
            # )
            predictions = []
            d_predictions = []
            for model in self.models:
                with torch.no_grad():
                    d_probs, probs, _ = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    token_type_ids=inputs['token_type_ids']
                                        )
                    predictions.append(probs)
                    d_predictions.append(d_probs)

            preds = torch.zeros_like(predictions[0])
            d_preds = torch.zeros_like(d_predictions[0])
            for probs in predictions:
                preds+=probs
            for d_probs in d_predictions:
                d_preds+=d_probs

            preds = torch.softmax(preds[:, 1:, :], dim=-1)  # 从cls后面开始 [batch_size,seq_len-1,vocab_size]
            d_preds = torch.softmax(d_preds[:, 1:, :], dim=-1) # 判断该token是否需要改动，下标为1对应error则需要修改
            error_probs = d_preds[:,:,1]# 获得每个句子每个token的错误概率 [batch_size, seq_len]
            incorr_probs = torch.max(error_probs, dim=-1)[
            0]  # [batch_size]:取每个句子所有token的错误概率最大者，作为此句子的错误概率（用于min_error_probability的trick）
            recall_max_probs, recall_max_ids = preds.max(dim=-1) # 找topk个可能性最大的token，返回概率和id
            # for i,incorr_prob in enumerate(incorr_probs):
            #     if incorr_prob<0.7:
            #         correct_sent_num += 1
            #         # recall_max_ids[i,:] = 1 # 给该batch第i个句子的所有token的编辑操作top1 id赋为keep
            #         recall_max_ids[i,:] = torch.where(recall_max_ids[i,:]!=0, 1, 0)

            # recall_max_ids[:,:,0] = torch.where(d_preds.argmax(dim=-1)==1, recall_max_ids[:,:,0], 1) # 若detect为1（error），才执行编辑操作，否则keep
            # recall_max_ids[:,:,0] = torch.where((d_preds.max(dim=-1)[0]>=0.9)&(d_preds.max(dim=-1)[1]==1), recall_max_ids[:,:,0], 1)

            recall_max_probs = recall_max_probs.tolist()
            recall_max_ids = recall_max_ids.tolist()
            recall_max_chars = [[self._id2ctag[char_level] for char_level in sent_level] for sent_level in recall_max_ids]
            batch_texts = [['']+list(t)[1:] for t in batch_texts]  # 占位符
            batch_outputs = [list(zip(text, max_char, max_prob)) for text, max_char, max_prob in zip(
                batch_texts, recall_max_chars, recall_max_probs)] # [bz, seq_len, 3] text是未修改前的句子
            outputs.extend(batch_outputs)
        print('correct_sent_num:'+str(correct_sent_num))
        return outputs

    @staticmethod
    def output2text(output):
        pred_text = ''
        for src_token, pred_token_list, pred_prob_list in output:
            pred_token = pred_token_list # 选择topk=1的token
            if '$KEEP' in pred_token:
                pred_text += src_token
            elif '$DELETE' in pred_token:
                continue
            elif '$REPLACE' in pred_token:
                pred_text += pred_token.split('_')[-1]
            elif '$APPEND' in pred_token:
                pred_text += src_token+pred_token.split('_')[-1]

        return pred_text