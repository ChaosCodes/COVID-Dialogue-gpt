from pytorch_pretrained_bert import BertTokenizer
import torch
import os
import codecs

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

dataset_file = os.path.join(os.path.abspath('..'), 'COVID-Dialogue-Dataset-Chinese.txt')


def seq2token_ids(source_seq, target_seq, encoder_max, decoder_max):
    
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


def make_dataset(data, buckets, file_name='train_data.pth'):
    train_data = [[] for _ in buckets]

    for d in data:
        bucket_id = len(buckets) - 1
        for idx in range(len(buckets)):
            source_len, target_len = buckets[idx]
            if len(d[0]) <= source_len and len(d[1]) <= target_len:
                bucket_id = idx
                break

        encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = seq2token_ids(d[0],\
                                                                                             d[1],\
                                                                                             buckets[bucket_id][0] + 2,\
                                                                                             buckets[bucket_id][1] + 2)
        train_data[bucket_id].append((encoder_input, 
                        decoder_input, 
                        mask_encoder_input, 
                        mask_decoder_input))

    for idx in range(len(buckets)):
        if len(train_data[idx]) == 0:
            continue
        encoder_input, \
        decoder_input, \
        mask_encoder_input, \
        mask_decoder_input = zip(*train_data[idx])

        encoder_input = torch.stack(encoder_input)
        decoder_input = torch.stack(decoder_input)
        mask_encoder_input = torch.stack(mask_encoder_input)
        mask_decoder_input = torch.stack(mask_decoder_input)


        train_data[idx] = [encoder_input, decoder_input, mask_encoder_input, mask_decoder_input]
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

# Set the buckets
buckets = [(50, 20), (75, 25), (90, 40), (120, 50), (150, 60), (200, 70), (250, 80), (300, 90), (398, 90), (398, 100)]


data = [[], [], []]

# construct the file_config
filename = list(range(2010, 2021))
filename.remove(2017)
decoding = ['utf-8'] * 10
decoding[1] = 'GBK'
decoding[4] = 'GBK'
file_config = zip(filename, decoding)

for name, decode in file_config:
    print(f'Process the {name} file')
    temp_data = get_splited_data_by_file(get_file_path_by_name(name), decode)
    data[0].extend(temp_data[0])
    data[1].extend(temp_data[1])
    data[2].extend(temp_data[2])


print(f'Process the train dataset')
make_dataset(data[0], buckets, 'train_data.pth')

print(f'Process the validate dataset')
make_dataset(data[1], buckets, 'validate_data.pth')

print(f'Process the test dataset')
make_dataset(data[2], buckets, 'test_data.pth')
