import torch

from pytorch_transformers import BertModel
from chinese_gpt import TransformerEncoder

model = BertModel.from_pretrained('bert-base-chinese')
encoder = TransformerEncoder()

old_state_dict = model.state_dict()
new_state_dict = encoder.state_dict()

for item in new_state_dict.keys():
    new_state_dict[item] = old_state_dict[item]

encoder.load_state_dict(new_state_dict)
torch.save(encoder.state_dict(), 'encoder.pth')
