from pytorch_pretrained_bert import BertTokenizer
import torch
import os

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

dataset_file = os.path.join(os.path.abspath('.'), 'Chinese-Dialogue.txt')

encoder_max = 400
decoder_max = 100

def seq2token_ids(source_seq, target_seq):
    
    # 可以尝试对source_seq进行切分
    source_seq = tokenizer.tokenize(source_seq) + ["[SEP]"]
    target_seq = ["[CLS]"] + tokenizer.tokenize(target_seq)

    # 设置不得超过encoder_max大小
    encoder_input = ["[CLS]"] + source_seq[-(encoder_max-1):]
    decoder_input = target_seq[:decoder_max-1] + ["[SEP]"]
    
    # conver to ids
    encoder_input = tokenizer.convert_tokens_to_ids(encoder_input)
    decoder_input = tokenizer.convert_tokens_to_ids(decoder_input)

    # mask
    mask_encoder_input = [1] * len(encoder_input)
    mask_decoder_input = [1] * len(decoder_input)

    # padding
    encoder_input += [0] * (encoder_max - len(encoder_input))
    decoder_input += [0] * (decoder_max - len(decoder_input))
    mask_encoder_input += [0] * (encoder_max - len(mask_encoder_input))
    mask_decoder_input += [0] * (decoder_max - len(mask_decoder_input))

    # turn into tensor
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input)
    
    mask_encoder_input = torch.LongTensor(mask_encoder_input)
    mask_decoder_input = torch.LongTensor(mask_decoder_input)

    return encoder_input, decoder_input, mask_encoder_input, mask_decoder_input


num = 0

def make_sentences(position, sentence):
    return {
        'position': position,
        'sentence': sentence
    }

doctor_or_patient = -1
data = []

temp_sentence = ''
doctor_flag = False
patient_flag = False
        
with open(dataset_file, 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        if line[:3] == 'id=':
            temp_sentence = ''
            doctor_flag = False
            patient_flag = False

        elif line[:3] == '病人:':
            patient_flag = True
            line = f.readline()
            sen = '病人:'+ line.rstrip()
            if sen[-1] not in '.。？，,?!！~～':
                sen += '。'
            temp_sentence += sen
                
        elif line[:3] == '医生:':
            if patient_flag: doctor_flag = True
            line = f.readline()
            sen = '医生:'+ line.rstrip()
            if sen[-1] not in '.。？，,?!！~～':
                sen += '。'
            if doctor_flag:
                data.append((temp_sentence, sen))
            
            temp_sentence += sen


def make_dataset(data, file_name='train_data.pth'):
    train_data = []

    for d in data:
        encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = seq2token_ids(d[0], d[1])
        train_data.append((encoder_input, 
                        decoder_input, 
                        mask_encoder_input, 
                        mask_decoder_input))


    encoder_input, \
    decoder_input, \
    mask_encoder_input, \
    mask_decoder_input = zip(*train_data)

    encoder_input = torch.stack(encoder_input)
    decoder_input = torch.stack(decoder_input)
    mask_encoder_input = torch.stack(mask_encoder_input)
    mask_decoder_input = torch.stack(mask_decoder_input)


    train_data = [encoder_input, decoder_input, mask_encoder_input, mask_decoder_input]

    torch.save(train_data, file_name)


make_dataset(data[:int(len(data) * 0.9)])

make_dataset(data[int(len(data) * 0.9):], 'val_data.pth')
