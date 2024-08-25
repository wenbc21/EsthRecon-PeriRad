import os
import argparse

import torch
import torch.optim as optim
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from dataset import SACDataSet
from model.model_zoo import model_dict

from engine import train_one_epoch, evaluate
from utils import read_split_data, read_sac_data, create_lr_scheduler, get_params_groups

def get_args_parser():
    parser = argparse.ArgumentParser('SAC training and evaluation script for image classification', add_help=False)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--data_path', type=str, default="dataset/Task3cls")
    parser.add_argument('--output_dir', type=str, default='weights')
    parser.add_argument('--model_config', type=str, default='resnet50')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    return parser


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    train_images_path, train_images_label = read_sac_data(args.data_path, "train")
    val_images_path, val_images_label = read_sac_data(args.data_path, "val")

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(args.input_size),
                                    #  transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(args.input_size * 1.143)),
                                   transforms.CenterCrop(args.input_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = SACDataSet(
        images_path=train_images_path,
        images_class=train_images_label,
        transform=data_transform["train"])

    val_dataset = SACDataSet(
        images_path=val_images_path,
        images_class=val_images_label,
        transform=data_transform["val"])

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

    model = model_dict[args.model_config](num_classes=args.num_classes)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
                
    model.to(device)

    parameters = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=2)

    max_accuracy = 0.0
    for epoch in range(args.epochs):
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
        
        print("[epoch {}] accuracy: {}".format(epoch, round(val_acc, 3)))

        if max_accuracy <= val_acc and epoch > 5:
            torch.save(model.state_dict(), os.path.join(args.output_dir, args.model_config + "_best.pth"))
            max_accuracy = val_acc

    torch.save(model.state_dict(), os.path.join(args.output_dir, args.model_config + "_last.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SAC model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)