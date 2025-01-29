import torch
import asm2vec


def cli(ipath, opath, mpath, limit, embedding_size, batch_size, epochs, neg_sample_num, calc_acc, device, lr):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if mpath:
        model, tokens = asm2vec.utils.load_model(mpath, device=device)
        functions, tokens_new = asm2vec.utils.load_data(ipath, limit=limit)
        tokens.update(tokens_new)
        model.update(len(functions), tokens.size())
    else:
        model = None
        functions, tokens = asm2vec.utils.load_data(ipath, limit=limit)

    def callback(context):
        progress = f'{context["epoch"]} | time = {context["time"]:.2f}, loss = {context["loss"]:.4f}'
        if context["accuracy"]:
            progress += f', accuracy = {context["accuracy"]:.4f}'
        print(progress)
        asm2vec.utils.save_model(opath, context["model"], context["tokens"])

    model = asm2vec.utils.train(
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
