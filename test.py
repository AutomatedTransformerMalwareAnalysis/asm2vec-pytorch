import torch
import torch.nn as nn
import asm2vec

def cli(ipath, mpath, epochs, neg_sample_num, limit, device, lr, pretty):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model, tokens
    model, tokens = asm2vec.utils.load_model(mpath, device=device)
    functions, tokens_new = asm2vec.utils.load_data(ipath)
    tokens.update(tokens_new)
    model.update(1, tokens.size())
    model = model.to(device)

    # train function embedding
    model = asm2vec.utils.train(
        functions,
        tokens,
        model=model,
        epochs=epochs,
        neg_sample_num=neg_sample_num,
        device=device,
        mode='test',
        learning_rate=lr
    )

    # show predicted probability results
    x, y = asm2vec.utils.preprocess(functions, tokens)
    probs = model.predict(x.to(device), y.to(device))
    asm2vec.utils.show_probs(x, y, probs, tokens, limit=limit, pretty=pretty)

if __name__ == '__main__':
    cli()
