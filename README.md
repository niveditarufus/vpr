# Visual Product Recognition
The current objective is to fine-tune an OpenAI CLIP (Contrastive Language-Image Pre-Training) Vision Transformer (ViT) H-14 model for a classification task on the Products10k dataset. The resulting embeddings will then be used to compute similarity.

To train the model, we merged the Products10k training and test sets, and also added images from the competition's validation small dataset, which can be accessed at this link: [https://drive.google.com/file/d/1i6OyBPWqSGCR_-6p35kvqBUWT_GovPZS/view?usp=sharing](https://drive.google.com/file/d/1i6OyBPWqSGCR_-6p35kvqBUWT_GovPZS/view?usp=sharing).

To begin training, prepare the dataset as described above, update all path variables in the config class in train_lightning.py, and then run the script using the following command: 

```python train_lightning.py ```

# Some useful links:
* [https://www.kaggle.com/competitions/products-10k/discussion/188026](https://www.kaggle.com/competitions/products-10k/discussion/188026)
* [https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/359316](https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/359316)
* [https://github.com/LouieShao/1st-Place-Solution-in-Google-Universal-Image-Embedding](https://github.com/LouieShao/1st-Place-Solution-in-Google-Universal-Image-Embedding)
* [https://github.com/XL-H/ECCV2022](https://github.com/XL-H/ECCV2022)
* [https://platform.sankuai.com/foodai2021.html#index](https://platform.sankuai.com/foodai2021.html#index)
* [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
* [https://github.com/mlfoundations/wise-ft](https://github.com/mlfoundations/wise-ft)
* [https://github.com/psinger/kaggle-landmark-recognition-2020-1st-place](https://github.com/psinger/kaggle-landmark-recognition-2020-1st-place)
* [Experiment logs](https://docs.google.com/spreadsheets/d/1U8C6m4_MFcsQKSUf74rCjuw-uPTw-hPn_HuGbRi3XTE/edit#gid=0)

# Distillation and TensorRT
* [https://github.com/lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)
* [https://github.com/NVIDIA/FasterTransformer/tree/main/examples/pytorch/vit/ViT-quantization](https://github.com/NVIDIA/FasterTransformer/tree/main/examples/pytorch/vit/ViT-quantization)
