import torch
import os.path
import numpy as np
import warnings

from transformer.models import AutoregressiveDecoder
from enwik8_chars import sample

warnings.simplefilter("ignore", DeprecationWarning)

MODEL_NAME = 'enwik8_char_large5'
CONTEXT_LEN = 128
TEMP = 0.0
GEN_CHARS_AMOUNT = 400

model = AutoregressiveDecoder(dim_embeddings=128, num_heads=12, num_layers=20, num_tokens=256, seq_length=CONTEXT_LEN)
model.cuda()

if os.path.isfile(f'models/{MODEL_NAME}.pt'):
    model.load_state_dict(torch.load(f'models/{MODEL_NAME}.pt'))
    print('Loaded model from disk.')
else:
    print('Unable to load model from disk, please check if MODEL_NAME is correct and try again.')

print('\nWelcome to the enwik8 character autoregressive model playground! Start by typing a sentence and hitting enter to generate characters.\n')

content = ''
while True:
    content += input()
    content_context = content[-CONTEXT_LEN:]
    content_context = np.fromstring(content_context, dtype=np.uint8)

    content_seq = torch.from_numpy(content_context).long().cuda()
    
    for _ in range(GEN_CHARS_AMOUNT):
        probs = model(content_seq[None, :])
        next_char = sample(probs[0, -1, :], TEMP)

        print(str(chr(max(32, next_char))), end='')
        content += str(chr(max(32, next_char)))

        content_seq = torch.cat([content_seq[1:], next_char[None]], dim=0)


