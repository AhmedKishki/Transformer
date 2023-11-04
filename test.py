import torch
from mask import *
from hyperparameters import *
from make_model import make_model


def inference_test():
    torch.set_default_device(config['device'])
    test_model = make_model(config)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).to(config['device'])
    src_mask = torch.ones(1, 1, 10)
    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for _ in range(9):
        out = test_model.decode(
            memory, src_mask, ys, Mask((ys.size(1))).subsequent_mask().type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    print("Example Untrained Model Prediction:", ys)

def run_tests():
    for _ in range(10):
        inference_test()
        
if __name__ == "__main__":
    run_tests()