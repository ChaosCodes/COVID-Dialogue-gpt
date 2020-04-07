import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import fire
import time

# uses allennlp modules
from allennlp.nn import util

# imports chinese gpt
from chinese_gpt import TransformerDecoderLM

# uses bert chinese wordpiece tokenization
from pytorch_pretrained_bert import OpenAIAdam, BertTokenizer

def top_k_logits(logits, k):
    """Mask logits so that only top-k logits remain
    """
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

def validate_model(
    top_k = 50,
    temperature = 1.0,
    decoder_path='decoder.pth',
    batch_size=1,
    show_num=10,
    gpu_id=0
    ):
    # make sure your model is on GPU
    device = torch.device(f"cuda:{gpu_id}")

    print('load model')
    #------------------------LOAD MODEL-----------------
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    decoder = TransformerDecoderLM()
    decoder.load_state_dict(torch.load(decoder_path))
    decoder = decoder.to(device)
    decoder.eval()

    print('load success')
    #------------------------END LOAD MODEL--------------


    #------------------------LOAD VALIDATE DATA------------------
    val_data = torch.load("val_data.pth")
    val_dataset = TensorDataset(*val_data)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=batch_size)
    #------------------------END LOAD VALIDATE DATA--------------


    #------------------------START VALIDATE-------------------
    update_count = 0
    print('start validate....')
    
    for batch in val_dataloader:
        with torch.no_grad():
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask, _ = batch
            mask = F.pad(mask, (0, 4), "constant", 1.0)

            logits, past = decoder(torch.cat([encoder_input, decoder_input[:,:4]], dim=1), mask, past=None, past_length=0)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)

            sentence = [decoder_input[:,:4]]

            probs = F.softmax(logits, dim=-1)
            prob, prev_pred = torch.topk(probs, k=1, dim=-1)
            sentence.append(prev_pred)

            length = 1
            # decoding loop
            for i in range(100):
                mask = F.pad(mask, (0, 1), "constant", 1.0)
                logits, past = decoder(prev_pred, mask, past=past, past_length=length)
                logits = logits.squeeze(1) / temperature
                logits = top_k_logits(logits, k=top_k)
                probs = F.softmax(logits, dim=-1)
                prev_pred = torch.multinomial(probs, num_samples=1)
                sentence.append(prev_pred)
                length += 1

            sentence = torch.cat(sentence, dim=-1)

            res = "".join(tokenizer.convert_ids_to_tokens(sentence[0].tolist()))
            inputs = "".join(tokenizer.convert_ids_to_tokens(encoder_input[0].tolist()))
            target = "".join(tokenizer.convert_ids_to_tokens(decoder_input[0].tolist()))

            print('-'*20 + f'Case {update_count}' + '-' * 20)
            print('-'*20 + 'Input' + '-' * 20)
            print(inputs)
            print('')

            print('-'*20 + 'Predcit' + '-' * 20)
            print(res[:100])
            print('')

            print('-'*20 + 'Target' + '-' * 20)
            print(target)
            print('')

            update_count += 1
            if update_count == show_num:
                break

    #------------------------END VALIDATE-------------------


if __name__ == '__main__':
    fire.Fire(validate_model)