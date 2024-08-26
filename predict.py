import os
import argparse
import json

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model.model_zoo import model_dict
from utils import read_sac_data, plot_test_metrics, tensor2img

def get_args_parser():
    parser = argparse.ArgumentParser('SAC model testing script for image classification', add_help=False)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--data_path', type=str, default="dataset/Task3clsAug")
    parser.add_argument('--weights_dir', type=str, default='weights')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--model_config', type=str, default='ResNet50')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    return parser

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    
    test_images_path, test_images_label = read_sac_data(args.data_path, "test")
    test_images_predict = []

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.29204324, 0.29204324, 0.29204324], [0.29269517, 0.29269517, 0.29269517])
    ])

    for img_path, image_label in zip(test_images_path, test_images_label) :
        # load image
        img = Image.open(img_path)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)

        # read class_indict
        json_path = 'dataset/class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
        with open(json_path, "r") as f:
            class_indict = json.load(f)

        # create model
        model = model_dict[args.model_config](num_classes=args.num_classes).to(device)
        # target_layers = [model.layer4[-1]]

        # load model weights
        model_weight_path = os.path.join(args.weights_dir, args.model_config + "_best.pth")
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        
        # predict class
        with torch.no_grad():
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_class = torch.argmax(predict).numpy()
            test_images_predict.append(predict[1])
            
            # torch.set_grad_enabled(True)
            # with GradCAM(model=model, target_layers=target_layers, use_cuda=True) as cam:
            #     targets = [ClassifierOutputTarget(1), ClassifierOutputTarget(1)] #指定查看class_num为386的热力图
            #     # aug_smooth=True, eigen_smooth=True 使用图像增强是热力图变得更加平滑
            #     grayscale_cams = cam(input_tensor=img.to(device), targets=None)#targets=None 自动调用概率最大的类别显示
            #     for grayscale_cam, tensorg in zip(grayscale_cams,img.to(device)):
            #         #将热力图结果与原图进行融合
            #         rgb_img = tensor2img(tensorg)
            #         visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            #         imggrad = Image.fromarray(visualization)
            #         imggrad.save(f'./cls_model/ResNet/results/grad_cam/{os.path.split(img_path)[-1]}')
            # torch.set_grad_enabled(False)
        
        print("label: {}, img_path: {}, class: {}, prob: {:.3}".format(
            class_indict[str(image_label)], os.path.split(img_path)[-1], class_indict[str(predict_class)], predict[predict_class].numpy()
        ))
        
    plot_test_metrics(test_images_label, test_images_predict, args.results_dir, args.model_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SAC model testing script for image classification', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
    main(args)