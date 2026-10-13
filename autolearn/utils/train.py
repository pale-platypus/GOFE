import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from autolearn.utils import log
from autolearn.utils.torch_callback import EarlyStopping, Checkpoint, LossTradeOff, ValidLoss


def pairwise_hinge_loss(y_hat, y_true, margin=1.0):
    """
    Pairwise hinge ranking loss. For every in-batch pair (i, j) where
    y_true[i] > y_true[j], penalise predictions where y_hat[i] - y_hat[j] < margin.
    Returns the mean violation over valid pairs.
    """
    y_hat = y_hat.flatten()
    y_true = y_true.flatten()
    diff_pred = y_hat.unsqueeze(1) - y_hat.unsqueeze(0)
    mask = (y_true.unsqueeze(1) > y_true.unsqueeze(0)).float()
    hinge = torch.clamp(margin - diff_pred, min=0.0)
    n_pairs = mask.sum().clamp(min=1.0)
    return (mask * hinge).sum() / n_pairs


def forward(x, y, model, loss_func, device, y_idx=None):
    x, y = x.to(device), y.to(device)
    # on batch begin
    y_hat = model(x)
    if y_idx is not None:
        y_hat = y_hat[y_idx]
    if isinstance(y, torch.DoubleTensor) or isinstance(y, torch.cuda.DoubleTensor):
        y = y.float()
        y_hat = y_hat.squeeze()
    elif isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
        y = y.reshape(-1)
        y_hat = y_hat.reshape(y.size()[0], -1)
    loss = loss_func(y_hat.squeeze(), y)
    return loss


def torch_train(
        dataset, model, optimizer, loss_func, device,
        epochs=512, batch_size=32,
        clip_grad=0,
        patience=20,
        ckp_path=None,
        time_budget=None
):
    earlystop_cb = EarlyStopping(patience=patience)
    ckp_cb = Checkpoint(ckp_path, earlystop_cb)

    with tqdm(total=epochs) as t:
        # on epoch begin
        for i in range(epochs):
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            model.train()
            epoch_loss = 0
            for x, y in data_loader:
                # on batch begin
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                loss = forward(x, y, model, loss_func, device)
                loss.backward()
                if clip_grad > 0:
                    for p in model.parameters():
                        nn.utils.clip_grad_norm_(p, clip_grad)
                optimizer.step()

                # on batch end
                epoch_loss += loss.item()

            model.eval()
            with torch.no_grad():
                y_hat = model(x)
                if isinstance(y_hat, tuple):
                    y_hat = y_hat[0]
                y_hat = y_hat.cpu().numpy()

            # on epoch end
            early_stopping = earlystop_cb.on_epoch_end(i, -epoch_loss, epoch_loss)
            ckp_cb.on_epoch_end(model)
            t.set_postfix(
                Epoch=f"{i: 03,d}",
                patience=f"{earlystop_cb.wait: 03,d}/{earlystop_cb.patience: 03,d}",
                loss=f"{epoch_loss: 0.5f}",
            )
            t.update(1)

            try:
                if time_budget is not None:
                    time_budget.check()
            except Exception as e:
                log(f"{e}")

            if early_stopping:
                break
    ckp_cb.on_train_end(model)
    return model


def multi_train(
        dataset, model, optimizer, loss_func_1, loss_func_2, device,
        epochs=512, batch_size=32, trade_off_epoch=5, trade_off=None,
        clip_grad=0,
        patience=20,
        ckp_path=None,
        valid_set=None,
        time_budget=None,
        cur_iter=0,
        kl_beta=0.0,
):
    earlystop_cb = EarlyStopping(patience=patience)
    ckp_cb = Checkpoint(ckp_path, earlystop_cb, cur_iter)
    loss_trade_off = LossTradeOff(trade_off_epoch, trade_off=trade_off)
    valid_loss_cb = ValidLoss(
        valid_set,
        forward,
        loss_func_1,
        loss_func_2,
        device
    )

    with tqdm(total=epochs) as t:
        # on epoch begin
        for i in range(epochs):
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            model.train()
            epoch_loss = 0
            epoch_loss_1 = 0
            epoch_loss_2 = 0
            trade_off = loss_trade_off.on_epoch_begin()
            valid_loss_cb.on_epoch_begin(trade_off, trade_off == loss_trade_off.default)
            earlystop_cb.on_epoch_begin(trade_off != loss_trade_off.default)
            for x, y1, x2, y2 in data_loader:
                x, y1 = x.to(device), y1.to(device)
                x2, y2 = x2.to(device), y2.to(device)
                # on batch begin
                optimizer.zero_grad()
                # loss_1 = forward(x, y1, model, loss_func_1, device, y_idx=0)
                # loss_2 = forward(x2, y2, model, loss_func_2, device, y_idx=1)
                y_hat_1, y_hat_2 = model(x)
                loss_1 = loss_func_1(y_hat_1.squeeze(), y1.float())
                y2 = x.reshape(-1)
                y_hat_2 = y_hat_2.reshape(y2.size()[0], -1)
                loss_2 = loss_func_2(y_hat_2.squeeze(), y2)
                loss_1_item = loss_1.sum().item()
                loss_2_item = loss_2.sum().item()
                loss = trade_off * loss_1.sum() + loss_2.sum()
                if kl_beta > 0.0:
                    last_kl = getattr(getattr(model, 'encoder', None), 'last_kl', None)
                    if last_kl is not None and last_kl.numel() > 0:
                        loss = loss + kl_beta * last_kl.mean()
                loss.backward()
                if clip_grad > 0:
                    for p in model.parameters():
                        nn.utils.clip_grad_norm_(p, clip_grad)
                optimizer.step()

                # on batch end
                epoch_loss_1 += trade_off * loss_1_item
                epoch_loss_2 += loss_2_item
                epoch_loss += loss.item()

            # on epoch end
            valid_loss = valid_loss_cb.on_epoch_end(
                model, epoch_loss
            )
            early_stopping = earlystop_cb.on_epoch_end(i, -valid_loss, valid_loss)
            loss_trade_off.on_epoch_end(i, epoch_loss_1, epoch_loss_2)
            ckp_cb.on_epoch_end(model)
            t.set_postfix(
                loss=f"{epoch_loss: 0.7f}",
                valid_loss=f"{valid_loss: 0.7f}",
                patience=f"{earlystop_cb.wait: 03,d}/{earlystop_cb.patience: 03,d}"
            )
            t.update(1)

            try:
                if time_budget is not None:
                    time_budget.check()
            except Exception as e:
                log(f"{e}")

            if early_stopping:
                break
    ckp_cb.on_train_end(model)
    return model
