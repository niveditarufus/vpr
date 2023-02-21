import open_clip
import torch
import torch.nn as nn

class classifier_model(nn.Module):
    def __init__(self):
        super(classifier_model, self).__init__()
        self.model,_ ,_ = open_clip.create_model_and_transforms('ViT-L-14')
        self.model = self.model.visual
        self.fc_model = nn.Sequential(nn.Linear(768, 9691))
    def forward(self, x):
        x = self.model(x)
        x = self.fc_model(x)
        return x

model_paths = [
                "/home/unni/workspace/active/models_l_14/good_aug/model_0025.pth",
                "/home/unni/workspace/active/models_l_14/test_train_combined/model_0020.pth",
                "/home/unni/workspace/active/models_l_14/test_train_combined/model_0026.pth",
                "/home/unni/workspace/active/models_l_14/test_train_combined/model_0030.pth",
                "/home/unni/workspace/active/models_l_14/train_only/model_0025.pth",
                "/home/unni/workspace/active/models_l_14/train_only/model_0032.pth",
                "/home/unni/workspace/active/models_l_14/train_only/model_0035.pth",
                "/home/unni/workspace/active/models_l_14/train_only/model_0043.pth"
              ]

state_dicts = []

for f_path in model_paths:
    state_dict_model = torch.load(f_path)['state_dict']
    state_dicts.append(state_dict_model)

len_models = len(state_dicts) 
soup_model = state_dicts[0]

for key in state_dicts[0].keys():
    soup_model[key].zero_()
    for model in state_dicts:
        soup_model[key]+=model[key]
    soup_model[key]/=len_models

torch.save({
        'model_state_dict': soup_model,
        }, './new_models/soup_model.pth')

# finetuned_model = 0.5032424851406929
# 0.95 = 0.49958811024845223
# 0.99 = 0.5026922757422585