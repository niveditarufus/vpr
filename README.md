# Visual Product Recognition
Right now the idea is to fine tune an openclip ViT L-14 model for classification task on products10k and then use those embeddings to compute similarity.
ViT H-14 is better but needs to be converted to tensorrt+8bitQuant or onnx+quantization for faster inference (should give roughly 3% increase).
# Some useful links:
* [https://www.kaggle.com/competitions/products-10k/discussion/188026](https://www.kaggle.com/competitions/products-10k/discussion/188026)
* [https://github.com/XL-H/ECCV2022](https://github.com/XL-H/ECCV2022)
* [https://platform.sankuai.com/foodai2021.html#index](https://platform.sankuai.com/foodai2021.html#index)
* [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
* [https://github.com/mlfoundations/wise-ft](https://github.com/mlfoundations/wise-ft)
* Lion optimizer [https://arxiv.org/pdf/2302.06675.pdf](https://arxiv.org/pdf/2302.06675.pdf)
* [https://github.com/psinger/kaggle-landmark-recognition-2020-1st-place](https://github.com/psinger/kaggle-landmark-recognition-2020-1st-place)
* [https://docs.google.com/spreadsheets/d/1U8C6m4_MFcsQKSUf74rCjuw-uPTw-hPn_HuGbRi3XTE/edit#gid=0](Experiments Log)
