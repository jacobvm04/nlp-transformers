import gzip
import numpy as np
import torch
import torch.nn.functional as F
import os.path
from torch.utils.tensorboard import SummaryWriter
from datasets.enwik8_dataset import Enwik8CharsDataset

from transformer.models import AutoregressiveDecoder

TRAIN_SIZE = 100000
TEST_SIZE = 50000
BATCH_SIZE = 20
TRAIN_EPOCHS = 5
LEARNING_RATE = 0.0001
WARMUP_STEPS = 5000
CONTEXT_LEN = 128

MODEL_NAME = 'enwik8_char_large5'

GEN_CHARS_AMOUNT = 1500
TEMP = 0.0

def sample(probs, tempature=0.0):
    if tempature == 0.0:
        return probs.argmax()

    probs = F.softmax(probs / tempature, dim=0)
    dist = torch.distributions.Categorical(probs)

    return dist.sample()

def evaluate(model, dataset):
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=Enwik8CharsDataset.collate)

    inputs, targets = next(iter(test_dataloader))

    inputs = inputs.cuda()
    targets = targets.cuda()

    outputs = model(inputs)
    loss = loss_fn(outputs.transpose(2, 1), targets)

    return loss.item()

if __name__ == '__main__':
    dataset = Enwik8CharsDataset(CONTEXT_LEN)
    train_dataset, test_dataset, _ = torch.utils.data.random_split(dataset, (10000000, 150000, len(dataset) - 10000000 - 150000))
    print(f'Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}')

    model = AutoregressiveDecoder(dim_embeddings=128, num_heads=12, num_layers=20, num_tokens=256, seq_length=CONTEXT_LEN)
    model.cuda()

    if os.path.isfile(f'models/{MODEL_NAME}.pt'):
        model.load_state_dict(torch.load(f'models/{MODEL_NAME}.pt'))
        print('Loaded model from disk')
    else:
        print('No model found, starting from scratch')

    board = SummaryWriter(f'runs/{MODEL_NAME}')

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE  )
    optimizer.zero_grad()

    loss_fn = torch.nn.NLLLoss(reduction='mean')

    i = 0
    for _ in range(TRAIN_EPOCHS):
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=Enwik8CharsDataset.collate)

        for inputs, targets in train_dataloader:
            if WARMUP_STEPS > 0 and i < WARMUP_STEPS:
                lr = max((LEARNING_RATE / WARMUP_STEPS) * i, 1e-10)
                optimizer.lr = lr

            optimizer.zero_grad()

            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = model(inputs)
            loss = loss_fn(outputs.transpose(2, 1), targets)

            board.add_scalar('loss', loss.item(), i)

            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f'Iteration: {i}, Loss: {loss.item()}')

            if i % 100 == 0:
                validation_loss = evaluate(model, test_dataset)
                board.add_scalar('validation_loss', validation_loss, i)
                print(f'Iteration: {i}, Validation loss: {validation_loss}')

                torch.save(model.state_dict(), f'models/{MODEL_NAME}.pt')

            i += 1

    # model.eval()
    # input_seq = test_data[:CONTEXT_LEN - 15].long().cuda()

    # for char in input_seq:
    #     print(str(chr(char)), end='')

    # for _ in range(GEN_CHARS_AMOUNT):
    #     probs = model(input_seq[None, :])
    #     next_char = sample(probs[0, -1, :], TEMP)

    #     print(str(chr(max(32, next_char))), end='')

    #     input_seq = torch.cat([input_seq[1:], next_char[None]], dim=0)
        