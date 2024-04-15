import os
import sys
import random
import shutil
import logging
import argparse
import subprocess
from time import time

import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data

from test import test
from lib.config import Config
from utils.evaluator import Evaluator


def train(model, train_loader, exp_dir, cfg, val_loader, train_state=None):
    # Get initial train state
    # 初始化优化器、学习率调度器和起始 epoch
    optimizer = cfg.get_optimizer(model.parameters())
    scheduler = cfg.get_lr_scheduler(optimizer)
    starting_epoch = 1

    if train_state is not None: # default = None
        # 加载之前保存的模型、优化器和学习率调度器的状态
        # 设置起始 epoch 为上次训练结束时的 epoch 加 1
        # 调用 scheduler.step() 函数来更新学习率
        model.load_state_dict(train_state['model'])
        optimizer.load_state_dict(train_state['optimizer'])
        scheduler.load_state_dict(train_state['lr_scheduler'])
        starting_epoch = train_state['epoch'] + 1
        scheduler.step(starting_epoch)

    # Train the model
    # 损失函数的参数
    criterion_parameters = cfg.get_loss_parameters()
    # 损失函数
    criterion = model.loss
    total_step = len(train_loader)
    # iter_log_interval: 1  # Log training iteration every N iterations
    # iter_time_window: 100  # Moving average iterations window for the printed loss metric
    # model_save_interval: 1  # Save model every N epochs
    ITER_LOG_INTERVAL = cfg['iter_log_interval']
    ITER_TIME_WINDOW = cfg['iter_time_window']
    MODEL_SAVE_INTERVAL = cfg['model_save_interval']
    # 总开始时间
    t0 = time()
    total_iter = 0
    iter_times = []
    logging.info("Starting training.")
    for epoch in range(starting_epoch, num_epochs + 1):
        # 当前 epoch 开始时间
        epoch_t0 = time()
        logging.info("Beginning epoch {}".format(epoch))
        accum_loss = 0
        # idx、图片、标签、图片索引
        for i, (images, labels, img_idxs) in enumerate(train_loader):
            # 总迭代次数
            total_iter += 1
            # 当前迭代开始时间
            iter_t0 = time()
            # Tensor 移动到可用设备上
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            # 前向传播
            outputs = model(images, epoch=epoch)
            loss, loss_dict_i = criterion(outputs, labels, **criterion_parameters)
            # 张量中的标量值
            accum_loss += loss.item()

            # Backward and optimize
            # 反向传播和优化
            # 梯度归零
            optimizer.zero_grad()
            # 反向传播计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()

            # 计算每次迭代时间
            iter_times.append(time() - iter_t0)
            if len(iter_times) > 100:
                # 最多只保存最近100个
                iter_times = iter_times[-ITER_TIME_WINDOW:]
            if (i + 1) % ITER_LOG_INTERVAL == 0:
                # 打印当前训练情况
                loss_str = ', '.join(
                    ['{}: {:.4f}'.format(loss_name, loss_dict_i[loss_name]) for loss_name in loss_dict_i])
                logging.info("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} ({}), s/iter: {:.4f}, lr: {:.1e}".format(
                    epoch,
                    num_epochs,
                    i + 1,
                    total_step,
                    accum_loss / (i + 1),
                    loss_str,
                    np.mean(iter_times),
                    optimizer.param_groups[0]["lr"],
                ))
        # 记录每个 epoch 的训练时间
        logging.info("Epoch time: {:.4f}".format(time() - epoch_t0))
        if epoch % MODEL_SAVE_INTERVAL == 0 or epoch == num_epochs:
            # 每个周期结束保存模型
            model_path = os.path.join(exp_dir, "models", "model_{:03d}.pt".format(epoch))
            save_train_state(model_path, model, optimizer, scheduler, epoch)
        if val_loader is not None:
            # 如果有验证集则进行验证并记录结果
            evaluator = Evaluator(val_loader.dataset, exp_root)
            evaluator, val_loss = test(
                model,
                val_loader,
                evaluator,
                None,
                cfg,
                view=False,
                epoch=-1,
                verbose=False,
            )
            _, results = evaluator.eval(label=None, only_metrics=True)
            logging.info("Epoch [{}/{}], Val loss: {:.4f}".format(epoch, num_epochs, val_loss))
            # 模型设置为训练模式
            model.train()
        scheduler.step()
    # 打印总时间
    logging.info("Training time: {:.4f}".format(time() - t0))

    return model


def save_train_state(path, model, optimizer, lr_scheduler, epoch):
    # 保存训练状态，包括模型的参数、优化器的状态以及学习率调度器的状态，以便在需要时恢复训练
    train_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch
    }

    torch.save(train_state, path)


def parse_args():
    # exp_name: 实验名称，用于标识当前训练实验的名称。
    # cfg: 配置文件的路径，指定用于训练的配置文件。
    # resume: 是否从上次中断的训练状态继续训练。
    # validate: 是否在训练过程中进行模型验证。
    # deterministic: 是否设置 cudnn.deterministic = True 和 cudnn.benchmark = False，以确保训练过程的确定性
    parser = argparse.ArgumentParser(description="Train PolyLaneNet")
    parser.add_argument("--exp_name", default="tusimple", help="Experiment name")
    parser.add_argument("--cfg", default="./cfgs/tusimple_bezier.yaml", help="Config file")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--validate", action="store_true", help="Validate model during training")
    parser.add_argument("--deterministic",
                        action="store_true",
                        help="set cudnn.deterministic = True and cudnn.benchmark = False")

    return parser.parse_args()


def get_code_state():
    # 获取代码的状态，包括当前的 Git 版本哈希和代码的 diff
    state = "Git hash: {}".format(
        subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    state += '\n*************\nGit diff:\n*************\n'
    state += subprocess.run(['git', 'diff'], stdout=subprocess.PIPE).stdout.decode('utf-8')

    return state


def setup_exp_dir(exps_dir, exp_name, cfg_path):
    # 设置实验目录
    # 创建一个包含模型、配置文件和代码状态的目录结构
    dirs = ["models"]
    exp_root = os.path.join(exps_dir, exp_name)

    for dirname in dirs:
        os.makedirs(os.path.join(exp_root, dirname), exist_ok=True)

    shutil.copyfile(cfg_path, os.path.join(exp_root, 'config.yaml'))
    with open(os.path.join(exp_root, 'code_state.txt'), 'w') as file:
        file.write(get_code_state())

    return exp_root


def get_exp_train_state(exp_root):
    # 获取实验训练状态，它从实验目录中的模型文件中加载最后一个保存的模型的训练状态
    models_dir = os.path.join(exp_root, "models")
    models = os.listdir(models_dir)
    last_epoch, last_modelname = sorted(
        [(int(name.split("_")[1].split(".")[0]), name) for name in models],
        key=lambda x: x[0],
    )[-1]
    train_state = torch.load(os.path.join(models_dir, last_modelname))

    return train_state


def log_on_exception(exc_type, exc_value, exc_traceback):
    # 记录未捕获的异常
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


if __name__ == "__main__":
    # 获取参数
    args = parse_args()
    cfg = Config(args.cfg)

    # Set up seeds
    # 设置种子
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

    if args.deterministic: # 默认为 false
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set up experiment
    # 设置根目录
    if not args.resume:
        # 新建一个目录
        exp_root = setup_exp_dir(cfg['exps_dir'], args.exp_name, args.cfg)
    else:
        exp_root = os.path.join(cfg['exps_dir'], os.path.basename(os.path.normpath(args.exp_name)))

    # 配置日志
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(exp_root, "log.txt")),
            logging.StreamHandler(),
        ],
    )

    # 异常处理器
    sys.excepthook = log_on_exception

    # 实验名称、配置文件内容以及命令行参数的值
    logging.info("Experiment name: {}".format(args.exp_name))
    logging.info("Config:\n" + str(cfg))
    logging.info("Args:\n" + str(args))

    # Get data sets
    # 获取数据集
    train_dataset = cfg.get_dataset("train")

    # Device configuration
    # 配置GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyper parameters
    # 配置 epoch 和 batch size
    num_epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]

    # Model
    # 获取模型并移动到可用设备
    model = cfg.get_model().to(device)

    # 从先前保存的实验目录中获取状态 否则为 None
    train_state = None
    if args.resume:
        train_state = get_exp_train_state(exp_root)

    # Data loader
    # 加载数据 根据 batch size 划分批次并打乱
    # num_workers 为子进程数量
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=8)

    # 验证数据集
    if args.validate:
        val_dataset = cfg.get_dataset("val")
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=8)
    # Train regressor
    try:
        model = train(
            model,
            train_loader,
            exp_root,
            cfg,
            val_loader=val_loader if args.validate else None,
            train_state=train_state,
        )
    except KeyboardInterrupt:
        logging.info("Training session terminated.")
    test_epoch = -1
    if cfg['backup'] is not None:
        subprocess.run(['rclone', 'copy', exp_root, '{}/{}'.format(cfg['backup'], args.exp_name)])

    # Eval model after training
    test_dataset = cfg.get_dataset("test")

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=8)

    evaluator = Evaluator(test_loader.dataset, exp_root)

    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(exp_root, "test_log.txt")),
            logging.StreamHandler(),
        ],
    )
    logging.info('Code state:\n {}'.format(get_code_state()))
    _, mean_loss = test(model, test_loader, evaluator, exp_root, cfg, epoch=test_epoch, view=False)
    logging.info("Mean test loss: {:.4f}".format(mean_loss))

    evaluator.exp_name = args.exp_name

    eval_str, _ = evaluator.eval(label='{}_{}'.format(os.path.basename(args.exp_name), test_epoch))

    logging.info(eval_str)
