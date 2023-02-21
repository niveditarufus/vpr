import open_clip
import torch
import torch.nn as nn

from torch.nn import functional as F
from torch import linalg

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
            sin = (1 - cos ** 2) ** 0.5
            angle_sum = cos * cos_m - sin * sin_m
            cos = angle_sum * one_hot + cos * (1 - one_hot)
            cos = cos * self.s
                        
        return cos

class classifier_model(nn.Module):
    def __init__(self):
        super(classifier_model, self).__init__()
        self.model = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')[0].visual
        self.fc = ArcFace(768, 9691)
    def forward(self, x, labels=None):
        x = self.model(x)
        x = self.fc(x, labels)
        return x

# load zero shot
model_zeroshot , _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
model_zeroshot = model_zeroshot.visual.to('cuda')

# load fine tuned
model_finetuned = classifier_model()
state_dict_model = torch.load('./new_models/ViT-L-14-laion2B-s32B-b82K/model_0025.pth')['state_dict']
model_finetuned = torch.nn.DataParallel(model_finetuned).to('cuda')
model_finetuned.load_state_dict(state_dict_model)
model_finetuned = model_finetuned.module.model

theta_0 = model_zeroshot.state_dict()
theta_1 = model_finetuned.state_dict()

# make sure checkpoints are compatible
assert set(theta_0.keys()) == set(theta_1.keys())

# interpolate between checkpoints with mixing coefficient alpha
alpha = 1.2
print("saving model for alpha = ", alpha)
theta = {
    key: (1-alpha) * theta_0[key] + alpha * theta_1[key]
    for key in theta_0.keys()
}
# update the model acccording to the new weights
model_finetuned.load_state_dict(theta)
torch.save({
        'model_state_dict': model_finetuned.state_dict(),
        }, './new_models/interp_model.pth')
