import os
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from dataset import MyDataSet
from model.model_zoo import model_dict

from engine import train_one_epoch, evaluate
from utils import read_dataset, create_lr_scheduler, get_params_groups, plot_training_loss
from model.DenseNet import load_state_dict

def get_args_parser():
    parser = argparse.ArgumentParser('SAC training and evaluation script for image classification', add_help=False)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--task', type=str, default="Task1_balanced_5fold")
    parser.add_argument('--data_path', type=str, default="dataset/Task1_crop_balanced_5fold")
    parser.add_argument('--img_channel', type=int, default=1)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--weights_dir', type=str, default='weights')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--model_config', type=str, default='DenseNet169')
    parser.add_argument('--pretrained', type=str, default='', help='initial weights path')
    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    return parser


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    # load dataset
    if args.fold != 0 :
        train_images_path, train_images_label = [], []
        for f in range(1, 6) :
            fold_path, fold_label = read_dataset(args.data_path, f"fold{f}")
            if args.fold == f :
                val_images_path, val_images_label = fold_path, fold_label
            else :
                train_images_path += fold_path
                train_images_label += fold_label
    else :
        train_images_path, train_images_label = read_dataset(args.data_path, "trainval")
        val_images_path, val_images_label = read_dataset(args.data_path, "test")
    
    if args.img_channel == 3 :
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else :
        mean, std = [0.5], [0.5]

    train_dataset = MyDataSet(
        images_path=train_images_path,
        images_class=train_images_label,
        is_train = True, 
        mean = mean, 
        std = std
    )

    val_dataset = MyDataSet(
        images_path=val_images_path,
        images_class=val_images_label,
        is_train = False, 
        mean = mean, 
        std = std
    )

    # build dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=val_dataset.collate_fn
    )
    
    if args.model_config.startswith("Res") :
        # ResNet
        model = model_dict[args.model_config](args.img_channel)
        if args.pretrained != "" and args.img_channel == 3:
            assert os.path.exists(args.pretrained), "file {} does not exist.".format(args.pretrained)
            model.load_state_dict(torch.load(args.pretrained))
            for param in model.parameters():
                param.requires_grad = False
        in_channel = model.fc.in_features
        model.fc = torch.nn.Linear(in_channel, args.num_classes)
    elif args.model_config.startswith("Dense") :
        # DenseNet
        model = model_dict[args.model_config](in_channels=args.img_channel, num_classes=args.num_classes).to(device)
        if args.pretrained != "" and args.img_channel == 3:
            if os.path.exists(args.pretrained):
                load_state_dict(model, args.pretrained)
            else:
                raise FileNotFoundError("not found weights file: {}".format(args.pretrained))
    elif args.model_config.startswith("Efficient") :
        model = model_dict[args.model_config](in_channels=args.img_channel, num_classes=args.num_classes).to(device)
        if args.pretrained != "" and args.img_channel == 3:
            if os.path.exists(args.pretrained):
                pretrained_dict = torch.load(args.pretrained, map_location=device)
                load_weights_dict = {k: v for k, v in pretrained_dict.items()
                                    if model.state_dict()[k].numel() == v.numel()}
                model.load_state_dict(load_weights_dict, strict=False)
            else:
                raise FileNotFoundError("not found weights file: {}".format(args.weights))
    elif args.model_config.startswith("Conv") :
        # ConvNeXt
        model = model_dict[args.model_config](in_channels=args.img_channel, num_classes=args.num_classes).to(device)
        if args.pretrained != "" and args.img_channel == 3:
            assert os.path.exists(args.pretrained), "pretrained file: '{}' not exist.".format(args.pretrained)
            pretrained_dict = torch.load(args.pretrained, map_location=device)["model"]
            for k in list(pretrained_dict.keys()):
                if "head" in k:
                    del pretrained_dict[k]
            model.load_state_dict(pretrained_dict, strict=False)
    else :
        print("model config argument fault!")
        exit()

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    
    model.to(device)

    parameters = get_params_groups(model, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=3)

    # train
    train_losses = []
    val_losses = []
    max_accuracy = 0.0
    
    log_file = open(f"{args.results_dir}/fold{args.fold}_training.txt", 'w')

    for epoch in range(1, args.epochs + 1):
        # train
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            lr_scheduler=lr_scheduler
        )

        # validate
        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch
        )
        
        # logging
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print("[epoch {}] accuracy: {}".format(epoch, round(val_acc, 4)))
        log_file.write(f"[epoch {epoch}] accuracy: {round(val_acc, 4)}")

        # save model
        if max_accuracy <= val_acc and epoch > 5:
            torch.save(model.state_dict(), os.path.join(args.weights_dir, f"fold{args.fold}_best.pth"))
            max_accuracy = val_acc
            log_file.write(", best for now !!")
        log_file.write("\n")

    # finish
    torch.save(model.state_dict(), os.path.join(args.weights_dir, f"fold{args.fold}_last.pth"))
    plot_training_loss(train_losses, val_losses, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SAC training and evaluation script for image classification', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.weights_dir:
        args.weights_dir = os.path.join(args.weights_dir, args.task, args.model_config)
        os.makedirs(args.weights_dir, exist_ok=True)
    if args.results_dir:
        args.results_dir = os.path.join(args.results_dir, args.task, args.model_config, "train")
        os.makedirs(args.results_dir, exist_ok=True)
    main(args)