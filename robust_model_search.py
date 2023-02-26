import argparse
import itertools
import os
import subprocess
from collections import OrderedDict

import numpy as np
import open_clip
import torch
import torch.nn as nn
from torch import linalg
from torch.nn import functional as F


class ArcFace(nn.Module):
    def __init__(self, cin, cout, s=30, m=0.3):
        super().__init__()
        self.s = s
        self.sin_m = torch.sin(torch.tensor(m))
        self.cos_m = torch.cos(torch.tensor(m))
        self.cout = cout
        self.fc = nn.Linear(cin, cout, bias=False)

    def forward(self, x, label=None):
        w_L2 = linalg.norm(self.fc.weight.detach(), dim=1, keepdim=True).T
        x_L2 = linalg.norm(x, dim=1, keepdim=True)
        cos = self.fc(x) / (x_L2 * w_L2)
        if label is not None:
            sin_m, cos_m = self.sin_m, self.cos_m
            one_hot = F.one_hot(label, num_classes=self.cout)
            sin = (1 - cos**2) ** 0.5
            angle_sum = cos * cos_m - sin * sin_m
            cos = angle_sum * one_hot + cos * (1 - one_hot)
            cos = cos * self.s
        return cos


class classifier_model(nn.Module):
    def __init__(self):
        super(classifier_model, self).__init__()
        self.model = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="laion2b_s32b_b82k"
        )[0].visual
        self.fc = ArcFace(768, 9691)

    def forward(self, x, labels=None):
        x = self.model(x)
        x = self.fc(x, labels)
        return x


def load_model(model_path):
    # load fine tuned
    model_finetuned = classifier_model()

    try:
        new_state_dict = torch.load(model_path)["model_state_dict"]
    except:
        state_dict_model = torch.load(model_path)["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict_model.items():
            if "fc.fc" in k:
                continue
            name = k.replace("model.", "")
            new_state_dict[name] = v

    model_finetuned = model_finetuned.model

    # model_finetuned = torch.nn.DataParallel(model_finetuned).to('cuda')
    model_finetuned.load_state_dict(new_state_dict)
    model_finetuned.to("cuda")
    # model_finetuned = model_finetuned.module.model

    theta_1 = model_finetuned.state_dict()
    return theta_1, model_finetuned


def load_models(model_path):
    # load zero shot
    model_zeroshot, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion2b_s32b_b82k"
    )
    model_zeroshot = model_zeroshot.visual.to("cuda")

    theta_1, model_finetuned = load_model(model_path)
    theta_0 = model_zeroshot.state_dict()

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())
    return theta_0, theta_1, model_finetuned


def load_models2(model_path, model_path2):
    theta_0, model_finetuned = load_model(model_path)
    theta_1, _ = load_model(model_path2)

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())

    return theta_0, theta_1, model_finetuned


def mix_2_models():
    # interpolate between checkpoints with mixing coefficient alpha
    for alpha in np.arange(0, 1.1, 0.1):
        print("saving model for alpha = ", alpha)
        if args.model_path2 == "":
            theta_0, theta_1, model_finetuned = load_models(args.model_path)
        else:
            theta_0, theta_1, model_finetuned = load_models2(
                args.model_path, args.model_path2
            )

        theta = {
            key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_0.keys()
        }
        # update the model acccording to the new weights
        model_finetuned.load_state_dict(theta)
        torch.save(
            {
                "model_state_dict": model_finetuned.state_dict(),
            },
            "./new_models/ViT-L-14-laion2B-s32B-b82K/interp_model.pth",
        )
        subprocess.call(["python", "local_evaluation.py"])


def mix_many_models():
    model_paths = [
        os.path.join(args.model_path, file_name)
        for file_name in os.listdir(args.model_path)
    ]

    alphas = np.arange(0, 1.1, 0.1)
    alpha_combinations = list(itertools.product(alphas, repeat=len(model_paths)))
    alpha_combinations = [
        alpha_combination
        for alpha_combination in alpha_combinations
        if sum(alpha_combination) == 1.0
    ]
    # alpha_combinations = [[0.0, 0.0, 0.6000000000000001, 0.4]]
    for i, alpha_combination in enumerate(alpha_combinations):
        print(f"Combination {i + 1} of {len(alpha_combinations)}")
        print(f"Alphas: {alpha_combination}")

        # Load models and calculate weighted average
        theta = {}
        for j, model_path in enumerate(model_paths):
            alpha = alpha_combination[j]
            theta_j, model_finetuned = load_model(model_path)

            for key in theta_j.keys():
                if key not in theta:
                    theta[key] = alpha * theta_j[key]
                else:
                    theta[key] += alpha * theta_j[key]

        # update the model acccording to the new weights
        model_finetuned.load_state_dict(theta)
        torch.save(
            {
                "model_state_dict": model_finetuned.state_dict(),
            },
            "./new_models/ViT-L-14-laion2B-s32B-b82K/interp_model.pth",
        )
        subprocess.call(["python", "local_evaluation.py"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="path to model")
    parser.add_argument("--model_path2", default="", help="path to model2")
    parser.add_argument(
        "--model_path_is_folder", action="store_true", help="is path to model a folder"
    )
    args = parser.parse_args()

    if args.model_path_is_folder:
        mix_many_models()
    else:
        mix_2_models()
