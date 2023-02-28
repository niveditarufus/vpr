import os
import shutil
import subprocess

input_folder = "/home/nick/Data/model_saves/vit_224_v2"
output_folder = "/home/nick/Data/visual-product-recognition-2023-starter-kit-master/new_models/ViT-L-14-laion2B-s32B-b82K"

for file_name in sorted(
    os.listdir(input_folder),
    key=lambda x: os.path.getmtime(os.path.join(input_folder, x)),
):
    if file_name.endswith(".ckpt"):
        # copy the file to the output folder with the desired name
        src_path = os.path.join(input_folder, file_name)
        dst_path = os.path.join(output_folder, "interp_model.pth")
        shutil.copyfile(src_path, dst_path)

        # print the name of the file
        print(file_name)
        subprocess.call(["python", "local_evaluation.py"])
