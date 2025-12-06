## 1. Dataset Preparation

We use the **COCO 2017 Dataset** for training.

1.  Download the dataset from Kaggle: [COCO 2017 Dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset/data).
2.  Extract the `train2017` folder.
3.  Organize your directory structure as follows:

project_root/ data/ train/ <-- Put your original high-res COCO images here


## 2. Preprocessing (Crucial for Speed)

To accelerate training (up to 10x faster), we perform **offline resizing** first.

Run the provided script to resize all images to `256x256`. The processed images will be saved to `data/train_256`.

```bash
python resize_data.py
Note: This process might take 20-40 minutes depending on your CPU/Disk speed, but it will save you days of training time.
```
## 3. Training
Once the data is resized, you can start training. The model uses Feature Matching Loss and Hinge Loss by default.

Run the following command:

```bash

python train.py --data_dir data/train_256 --out_dir outputs --epochs 100 --batch_size 32
```
Arguments
You can adjust the parameters based on your GPU memory:

--data_dir: Path to the pre-resized dataset (default: data/train_256).

--out_dir: Where to save sample images and logs (default: outputs).

--epochs: Number of training epochs (default: 100).

--batch_size: Batch size (default: 32. Reduce to 16 or 8 if you run out of VRAM).

--lambda_fm: Weight for Feature Matching Loss (higher values improve structure).
