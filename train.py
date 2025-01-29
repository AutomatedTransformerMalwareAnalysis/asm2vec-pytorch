

import torch
from DataIngest.Asm2Vec import utils


def cli(ipath: str,
        opath: str,
        mpath: str = None,
        limit: int = None,
        embedding_size: int = 100,
        batch_size: int = 1024,
        epochs: int = 10,
        neg_sample_num: int = 25,
        calc_acc: bool = False,
        device: str = "auto",
        lr: float = 0.02):
    print("[INFO] Training DisASM Embeddings")
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if mpath:
        model, tokens = utils.load_model(mpath, device=device)
        functions, tokens_new = utils.load_data(ipath, limit=limit)
        tokens.update(tokens_new)
        model.update(len(functions), tokens.size())
    else:
        model = None
        functions, tokens = utils.load_data(ipath, limit=limit)

    def callback(context):
        progress = f'{context["epoch"]} | time = {context["time"]:.2f}, loss = {context["loss"]:.4f}'
        if context["accuracy"]:
            progress += f', accuracy = {context["accuracy"]:.4f}'
        print(progress)
        utils.save_model(opath, context["model"], context["tokens"])

    model = utils.train(
        functions,
        tokens,
        model=model,
        embedding_size=embedding_size,
        batch_size=batch_size,
        epochs=epochs,
        neg_sample_num=neg_sample_num,
        calc_acc=calc_acc,
        device=device,
        callback=callback,
        learning_rate=lr
    )

if __name__ == '__main__':
    cli()
