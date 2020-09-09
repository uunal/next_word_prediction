# %%
import torch
import string

# thanks to @renatoviolin 
# simply changing mdoels and create a playground for tested models.

# naming --> MODEL_hidden-layerXattention-headsXhidden-size
# selected models has different dimensions and diferent epochs in pretraining
# BUT have same tokenizers trained on same data!


MODEL_6x12x768 = "models/testing/LogBERT-6x12x768" 
MODEL_8x8x512 = "models/testing/LogBERT-8x8x512"



# TODO preserve naming convention
BASE_MODEL_2X2 = "models/testing/LogBERT-base-2epoch" #
BASE_MODEL_2X1 = "models/testing/LogBERT-base-1epoch" #
SMALL_MODEL = "models/testing/LogBERT-small-1epoch" #

# TODO change naming convention for tokenizers
# TODO tokenizers can be defined differently but no --
# need to preserve them inside model directory.


MODELa_6x12x768 = "models/testing/LogBERT-6x12x768-from-checkpoint" 

from transformers import BertTokenizer, BertForMaskedLM

bert_tokenizer_6x12x768 = BertTokenizer.from_pretrained(MODEL_6x12x768)
bert_model_6x12x768 = BertForMaskedLM.from_pretrained(MODEL_6x12x768).eval()

bert_tokenizer_8x8x512 = BertTokenizer.from_pretrained(MODEL_8x8x512)
bert_model_8x8x512 = BertForMaskedLM.from_pretrained(MODEL_8x8x512).eval()

# bert_model_12x12x768_2epoch
bert_tokenizer_2x2  =BertTokenizer.from_pretrained(BASE_MODEL_2X2)
bert_model_2x2 = BertForMaskedLM.from_pretrained(BASE_MODEL_2X2).eval()

# bert_model_12x12x768_1epoch
bert_tokenizer_2x1 = BertTokenizer.from_pretrained(BASE_MODEL_2X1)
bert_model_2x1 = BertForMaskedLM.from_pretrained(BASE_MODEL_2X1).eval()

# bert_model_4x8x512_1epoch
bert_sma_tokenizer = BertTokenizer.from_pretrained(SMALL_MODEL)
bert_sma_model = BertForMaskedLM.from_pretrained(SMALL_MODEL).eval()

# bert_model_6x12x768_4epoch different batching
roberta_tokenizer_6x12x768 = BertTokenizer.from_pretrained(MODELa_6x12x768)
roberta_model_6x12x768 = BertForMaskedLM.from_pretrained(MODELa_6x12x768).eval()

#TODO
#Maybe, Roberta models should be added to compare BPE and WordPiece

'''from transformers import XLNetTokenizer, XLNetLMHeadModel
xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
xlnet_model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased').eval()

from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
xlmroberta_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
xlmroberta_model = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base').eval()

from transformers import BartTokenizer, BartForConditionalGeneration
bart_tokenizer = BartTokenizer.from_pretrained('bart-large')
bart_model = BartForConditionalGeneration.from_pretrained('bart-large').eval()

from transformers import ElectraTokenizer, ElectraForMaskedLM
electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-generator')
electra_model = ElectraForMaskedLM.from_pretrained('google/electra-small-generator').eval()

from transformers import RobertaTokenizer, RobertaForMaskedLM
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForMaskedLM.from_pretrained('roberta-base').eval()'''

top_k = 10


def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


def get_all_predictions(text_sentence, top_clean=5):

    # ========================= LogBERT-6x12x768 =================================
    print(text_sentence)
    input_ids, mask_idx = encode(bert_tokenizer_6x12x768, text_sentence)
    with torch.no_grad():
        predict = bert_model_6x12x768(input_ids)[0]
    
    logbert_6x12x768 = decode(bert_tokenizer_6x12x768, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= LogBERT-8x8x512 =================================
    print(text_sentence)
    input_ids, mask_idx = encode(bert_tokenizer_8x8x512, text_sentence)
    with torch.no_grad():
        predict = bert_model_8x8x512(input_ids)[0]
    
    logbert_8x8x512 = decode(bert_tokenizer_8x8x512, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= LogBERT-base-2x2 (bert_model_12x12x768_2epoch) =================================
    print(text_sentence)
    input_ids, mask_idx = encode(bert_tokenizer_2x2, text_sentence)
    with torch.no_grad():
        predict = bert_model_2x2(input_ids)[0]
    logbert_2x2 = decode(bert_tokenizer_2x2, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)


    # ========================= LogBERT-base-2x1 (bert_model_12x12x768_1epoch)=================================
    print(text_sentence)
    input_ids, mask_idx = encode(bert_tokenizer_2x1, text_sentence)
    with torch.no_grad():
        predict = bert_model_2x1(input_ids)[0]
    logbert_2x1 = decode(bert_tokenizer_2x1, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= LogBERT-small (bert_model_4x8x512_1epoch)=================================
    print(text_sentence)
    input_ids, mask_idx = encode(bert_sma_tokenizer, text_sentence)
    with torch.no_grad():
        predict = bert_sma_model(input_ids)[0]
    logbert_s = decode(bert_sma_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)


    # ========================= LogBERTa-6x12x768 =================================

    input_ids, mask_idx = encode(roberta_tokenizer_6x12x768, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = roberta_model_6x12x768(input_ids)[0]
    logberta_6x12x768 = decode(roberta_tokenizer_6x12x768, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= LogBERT-mini =================================
    #print(text_sentence)
    #input_ids, mask_idx = encode(bert_min_tokenizer, text_sentence)
    #with torch.no_grad():
    #    predict = bert_min_model(input_ids)[0]
    #logbert_mi = decode(bert_min_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    '''# ========================= XLNET LARGE =================================
    input_ids, mask_idx = encode(xlnet_tokenizer, text_sentence, False)
    perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
    perm_mask[:, :, mask_idx] = 1.0  # Previous tokens don't see last token
    target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
    target_mapping[0, 0, mask_idx] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

    with torch.no_grad():
        predict = xlnet_model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)[0]
    xlnet = decode(xlnet_tokenizer, predict[0, 0, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= XLM ROBERTA BASE =================================
    input_ids, mask_idx = encode(xlmroberta_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = xlmroberta_model(input_ids)[0]
    xlm = decode(xlmroberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= BART =================================
    input_ids, mask_idx = encode(bart_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = bart_model(input_ids)[0]
    bart = decode(bart_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= ELECTRA =================================
    input_ids, mask_idx = encode(electra_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = electra_model(input_ids)[0]
    electra = decode(electra_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= ROBERTA =================================
    input_ids, mask_idx = encode(roberta_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = roberta_model(input_ids)[0]
    roberta = decode(roberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    return {'bert': bert,
            'xlnet': xlnet,
            'xlm': xlm,
            'bart': bart,
            'electra': electra,
            'roberta': roberta}'''
    
    return {'logbert-2x1': logbert_2x1,
        'logbert-6x12x768': logbert_6x12x768,
        'logbert-8x8x512': logbert_8x8x512,      
        'logbert-small': logbert_s,
        'logbert-2x2':logbert_2x2,
        'logberta-6x12x768':logberta_6x12x768}
        #'logbert-medium': logbert_m,
        #'logbert-mini': logbert_mi}
