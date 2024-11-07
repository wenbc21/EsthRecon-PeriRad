import os
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from dataset import MyDataSet
from model.model_zoo import model_dict

from engine import train_one_epoch, evaluate
from utils import read_dataset, create_lr_scheduler, get_params_groups, get_mean_std, plot_training_loss
from model.DenseNet import load_state_dict

def get_args_parser():
    parser = argparse.ArgumentParser('SAC training and evaluation script for image classification', add_help=False)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--task', type=str, default="Task1_crop_gray")
    parser.add_argument('--data_path', type=str, default="dataset/Task1_crop")
    parser.add_argument('--is_rgb', type=bool, default=False)
    parser.add_argument('--weights_dir', type=str, default='weights')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--model_config', type=str, default='DenseNet201')
    parser.add_argument('--pretrained', type=str, default='', help='initial weights path')
    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    return parser


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    # load dataset
    train_images_path, train_images_label = read_dataset(args.data_path, "train")
    val_images_path, val_images_label = read_dataset(args.data_path, "val")
    
    # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] if args.pretrained != "" else get_mean_std(train_images_path)
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
        model = model_dict[args.model_config]()
        if args.pretrained != "":
            assert os.path.exists(args.pretrained), "file {} does not exist.".format(args.pretrained)
            model.load_state_dict(torch.load(args.pretrained))
            for param in model.parameters():
                param.requires_grad = False
            in_channel = model.fc.in_features
            model.fc = torch.nn.Linear(in_channel, args.num_classes)
    elif args.model_config.startswith("Dense") :
        # DenseNet
        model = model_dict[args.model_config](in_channels=1, num_classes=args.num_classes).to(device)
        if args.pretrained != "":
            if os.path.exists(args.pretrained):
                load_state_dict(model, args.pretrained)
            else:
                raise FileNotFoundError("not found weights file: {}".format(args.pretrained))
    elif args.model_config.startswith("Efficient") :
        model = model_dict[args.model_config](in_channels=1, num_classes=args.num_classes).to(device)
        if args.pretrained != "":
            if os.path.exists(args.pretrained):
                pretrained_dict = torch.load(args.pretrained, map_location=device)
                load_weights_dict = {k: v for k, v in pretrained_dict.items()
                                    if model.state_dict()[k].numel() == v.numel()}
                model.load_state_dict(load_weights_dict, strict=False)
            else:
                raise FileNotFoundError("not found weights file: {}".format(args.weights))
    elif args.model_config.startswith("Conv") :
        # ConvNeXt
        model = model_dict[args.model_config](in_channels=1, num_classes=args.num_classes).to(device)
        if args.pretrained != "":
            assert os.path.exists(args.pretrained), "pretrained file: '{}' not exist.".format(args.pretrained)
            pretrained_dict = torch.load(args.pretrained, map_location=device)["model"]
            for k in list(pretrained_dict.keys()):
                if "head" in k:
                    del pretrained_dict[k]
            model.load_state_dict(pretrained_dict, strict=False)
    elif args.model_config.startswith("UNet") :
        # UNet
        model = model_dict[args.model_config](n_channels=3, n_classes=args.num_classes).to(device)
    else :
        print("argument fault!")
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

        # save model
        if max_accuracy <= val_acc and epoch > 5:
            torch.save(model.state_dict(), os.path.join(args.weights_dir, args.model_config + "_best.pth"))
            max_accuracy = val_acc

    # finish
    torch.save(model.state_dict(), os.path.join(args.weights_dir, args.model_config + "_last.pth"))
    plot_training_loss(train_losses, val_losses, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SAC training and evaluation script for image classification', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.weights_dir:
        args.weights_dir = os.path.join(args.weights_dir, args.task)
        os.makedirs(args.weights_dir, exist_ok=True)
    if args.results_dir:
        args.results_dir = os.path.join(args.results_dir, args.task, "train")
        os.makedirs(args.results_dir, exist_ok=True)
    main(args)