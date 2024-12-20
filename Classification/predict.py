import os
import argparse
import json

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn import metrics
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model.model_zoo import model_dict
from utils import read_dataset, plot_test_metrics, tensor2img, augment_and_pad

def get_args_parser():
    parser = argparse.ArgumentParser('SAC model testing script for image classification', add_help=False)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--task', type=str, default="Task3_final")
    parser.add_argument('--data_path', type=str, default="dataset/Task3_crop")
    parser.add_argument('--img_channel', type=int, default=1)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--grad_cam', type=bool, default=True)
    parser.add_argument('--weights_dir', type=str, default='weights')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--model_config', type=str, default='DenseNet169')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    return parser


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    
    # load dataset
    test_images_path, test_images_label = read_dataset(args.data_path, "test")
    test_images_predict = []
    test_image_class = []
    
    # read class_indict
    json_path = os.path.join(args.data_path, 'class_indices.json')
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    f = open(f"{args.results_dir}/fold{args.fold}_metrics.txt", 'w')
    
    if args.img_channel == 3 :
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else :
        mean, std = [0.5], [0.5]
    
    # create model
    model = model_dict[args.model_config](in_channels=args.img_channel, num_classes=args.num_classes).to(device)
    if args.grad_cam :
        os.makedirs(os.path.join(args.results_dir, "grad_cam"), exist_ok=True)
        if args.model_config.startswith("Res") :
            target_layers = [model.layer4[-1]]
        elif args.model_config.startswith("Dense") :
            target_layers = [model.features.denseblock4.denselayer32]

    # load model weights
    model_weight_path = os.path.join(args.weights_dir, f"fold{args.fold}_last.pth")
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # inference
    for img_path, image_label in zip(test_images_path, test_images_label) :
        # load image
        img = Image.open(img_path)
        if args.img_channel == 1 :
            img = img.convert('L')
        img = augment_and_pad(img, 224, mean, std, args.img_channel)
        img = torch.unsqueeze(img, dim=0)
        
        # predict class
        with torch.no_grad():
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_class = torch.argmax(predict).numpy()
            test_images_predict.append(predict[1])
            test_image_class.append(predict_class)
            
            if args.grad_cam :
                torch.set_grad_enabled(True)
                with GradCAM(model=model, target_layers=target_layers) as cam:
                    targets = [ClassifierOutputTarget(0)] #指定查看class_num为386的热力图
                    # aug_smooth=True, eigen_smooth=True 使用图像增强是热力图变得更加平滑
                    grayscale_cams = cam(input_tensor=img.to(device), targets=targets)#targets=None 自动调用概率最大的类别显示
                    for grayscale_cam, tensorg in zip(grayscale_cams,img.to(device)):
                        #将热力图结果与原图进行融合
                        rgb_img = tensor2img(tensorg)
                        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                        imggrad = Image.fromarray(visualization)
                        imggrad.save(os.path.join(args.results_dir, "grad_cam", os.path.split(img_path)[-1]))
                torch.set_grad_enabled(False)
        
        # result
        print("label: {}, img_path: {}, class: {}, prob: {:.3}".format(
            class_indict[str(image_label)], os.path.split(img_path)[-1], class_indict[str(predict_class)], predict[predict_class].numpy()
        ))
        f.write("label: {}, img_path: {}, class: {}, prob: {:.3}\n".format(
            class_indict[str(image_label)], os.path.split(img_path)[-1], class_indict[str(predict_class)], predict[predict_class].numpy()
        ))
    
    # metrics
    accuracy = metrics.accuracy_score(test_images_label, test_image_class)
    precision = metrics.precision_score(test_images_label, test_image_class)
    recall = metrics.recall_score(test_images_label, test_image_class)
    f1 = metrics.f1_score(test_images_label, test_image_class)
    print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1:{f1}")
    f.write(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1:{f1}\n")
    # visualize
    auroc, auprc = plot_test_metrics(test_images_label, test_images_predict, args.results_dir, f"fold{args.fold}")
    print(f"AUROC: {auroc}, AUPRC: {auprc}")
    f.write(f"AUROC: {auroc}, AUPRC: {auprc}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SAC model testing script for image classification', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.weights_dir:
        args.weights_dir = os.path.join(args.weights_dir, args.task, args.model_config)
        os.makedirs(args.weights_dir, exist_ok=True)
    if args.results_dir:
        args.results_dir = os.path.join(args.results_dir, args.task, args.model_config, "evaluate")
        os.makedirs(args.results_dir, exist_ok=True)
    main(args)