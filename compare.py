import torch
import torch.nn as nn
import asm2vec

def cosine_similarity(v1, v2):
    return (v1 @ v2 / (v1.norm() * v2.norm())).item()


def cli(ipath1, ipath2, mpath, epochs, device, lr):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model, tokens
    model, tokens = asm2vec.utils.load_model(mpath, device=device)
    functions, tokens_new = asm2vec.utils.load_data([ipath1, ipath2])
    tokens.update(tokens_new)
    model.update(2, tokens.size())
    model = model.to(device)
    
    # train function embedding
    model = asm2vec.utils.train(
        functions,
        tokens,
        model=model,
        epochs=epochs,
        device=device,
        mode='test',
        learning_rate=lr
    )

    # compare 2 function vectors
    v1, v2 = model.to('cpu').embeddings_f(torch.tensor([0, 1]))

    print(f'cosine similarity : {cosine_similarity(v1, v2):.6f}')

if __name__ == '__main__':
    cli()
