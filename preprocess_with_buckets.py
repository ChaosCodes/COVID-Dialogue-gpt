from pytorch_pretrained_bert import BertTokenizer
import torch
import os
import codecs

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

dataset_file = os.path.join(os.path.abspath('.'), 'COVID-Dialogue-Dataset-Chinese.txt')

MAX_ENCODER_SIZE = 400
MAX_DECODER_SIZE = 100

def seq2token_ids(source_seq, target_seq):
    
    # 可以尝试对source_seq进行切分
    source_seq = tokenizer.tokenize(source_seq) + ["[SEP]"]
    target_seq = ["[CLS]"] + tokenizer.tokenize(target_seq)

    # 设置不得超过 MAX_ENCODER_SIZE 大小
    encoder_input = ["[CLS]"] + source_seq[-(MAX_ENCODER_SIZE - 1):]
    decoder_input = target_seq[:MAX_DECODER_SIZE - 1] + ["[SEP]"]
    enc_len = len(encoder_input)
    dec_len = len(decoder_input)
    
    # conver to ids
    encoder_input = tokenizer.convert_tokens_to_ids(encoder_input)
    decoder_input = tokenizer.convert_tokens_to_ids(decoder_input)

    # mask
    mask_encoder_input = [1] * len(encoder_input)
    mask_decoder_input = [1] * len(decoder_input)

    # padding
    encoder_input += [0] * (MAX_ENCODER_SIZE - len(encoder_input))
    decoder_input += [0] * (MAX_DECODER_SIZE - len(decoder_input))
    mask_encoder_input += [0] * (MAX_ENCODER_SIZE - len(mask_encoder_input))
    mask_decoder_input += [0] * (MAX_DECODER_SIZE - len(mask_decoder_input))

    # turn into tensor
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input)
    
    mask_encoder_input = torch.LongTensor(mask_encoder_input)
    mask_decoder_input = torch.LongTensor(mask_decoder_input)
    
    enc_len, dec_len = torch.LongTensor([enc_len]), torch.LongTensor([dec_len])

    return encoder_input, decoder_input, mask_encoder_input, mask_decoder_input, enc_len, dec_len


def make_dataset(data, file_name='train_data.pth'):
    train_data = []

    for d in data:
        encoder_input, decoder_input, mask_encoder_input, mask_decoder_input, enc_len, dec_len = seq2token_ids(d[0], d[1])
        train_data.append((encoder_input, 
                           decoder_input,
                           mask_encoder_input,
                           mask_decoder_input,
                           enc_len,
                           dec_len))


    encoder_input, \
    decoder_input, \
    mask_encoder_input, \
    mask_decoder_input, \
    enc_len, \
    dec_len = zip(*train_data)

    encoder_input = torch.stack(encoder_input)
    decoder_input = torch.stack(decoder_input)
    mask_encoder_input = torch.stack(mask_encoder_input)
    mask_decoder_input = torch.stack(mask_decoder_input)
    enc_len = torch.stack(enc_len)
    dec_len = torch.stack(dec_len)


    train_data = [encoder_input, decoder_input, mask_encoder_input, mask_decoder_input, enc_len, dec_len]

    torch.save(train_data, file_name)

def get_splited_data_by_file(dataset_file, decode):
    data = [[], [], []]
    data_split_flag = 0
    num = 0

    temp_sentence = ''
    doctor_flag = False
    patient_flag = False

    total_id_num = 0
    # get the total idx num
    with codecs.open(dataset_file, 'r', decode) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line[:3] == 'id=':
                total_id_num += 1

    validate_idx = int(float(total_id_num) * 8 / 10)
    test_idx = int(float(total_id_num) * 9 / 10)

    with codecs.open(dataset_file, 'r', decode) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line[:3] == 'id=':
                num += 1
                if num >= validate_idx:
                    if num < test_idx:
                        data_split_flag = 1
                    else:
                        data_split_flag = 2
                else:
                    data_split_flag = 0
            elif line[:8] == 'Dialogue':
                temp_sentence = ''
                doctor_flag = False
                patient_flag = False

            elif line[:3] == '病人：' or line[:3] == '病人:':
                patient_flag = True
                line = f.readline()
                sen = '病人：'+ line.rstrip()
                if sen[-1] not in '.。？，,?!！~～':
                    sen += '。'
                temp_sentence += sen

            elif line[:3] == '医生：' or line[:3] == '医生:':
                if patient_flag: doctor_flag = True
                line = f.readline()
                sen = '医生：'+ line.rstrip()
                if sen[-1] not in '.。？，,?!！~～':
                    sen += '。'
                if doctor_flag:
                    data[data_split_flag].append((temp_sentence, sen))

                temp_sentence += sen
    return data


def get_file_path_by_name(name):
    return os.path.join(os.path.abspath('.'), \
                        'Medical-Dialogue-Dataset-Chinese',\
                        f'{name}.txt')


data = []

# # construct the file_config
# filename = list(range(2010, 2021))
# filename.remove(2017)
# decoding = ['utf-8'] * 10
# decoding[1] = 'GBK'
# decoding[4] = 'GBK'
# file_config = zip(filename, decoding)

# for name, decode in file_config:
#     print(f'Process the {name} file')
#     temp_data = get_splited_data_by_file(get_file_path_by_name(name), decode)
#     data[0].extend(temp_data[0])
#     data[1].extend(temp_data[1])
#     data[2].extend(temp_data[2])

data = get_splited_data_by_file(dataset_file, 'utf-8')

print(f'Process the train dataset')
make_dataset(data[0], 'train_data.pth')

print(f'Process the validate dataset')
make_dataset(data[1], 'validate_data.pth')

print(f'Process the test dataset')
make_dataset(data[2], 'test_data.pth')
