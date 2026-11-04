ITCS6169-A2

CV assignment 2 — Self-supervised visual representation learning (MoCo, MAE-style, DINO, joint rotation + classification) with a randomly initialized ResNet18

Environment

* Python 3.12.10
* PyTorch 2.12.0 (dev build, cu128) — note: a nightly build was required for RTX 50-series (Blackwell) GPU support; a stable PyTorch release may behave slightly differently
* Torchvision 0.27.0 (dev build)
* Seed: `SEED = 0` (set via `set_random_seed()`, applied before each training run)

Install
Adjust CUDA version to match your driver via `nvidia-smi`:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib jupyter
```

For a normal (non-Blackwell) CUDA version:

```
pip install torch torchvision
# or, to target a specific CUDA version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu<x>
# x = your driver's CUDA version
```

Files

* `ITCS_6169_8169_Assignment2.ipynb` — all code: data loading, ResNet18 model definition, baseline, MoCo, MAE-style, DINO, joint rotation + classification, and the final test evaluation
* `checkpoints/resnet18_ssl_final.pth` — trained checkpoint of the best model (DINO pretraining + fine-tuning)
* `requirements.txt`

Dataset

```
data/
train/
...
test/
...
```

Dataset folder should be named `data` as per code requirements. The `data` folder should be placed in the same directory as the Python notebook. It is not committed (see `.gitignore`).

Checkpoints
Only the final checkpoint, `checkpoints/resnet18_ssl_final.pth`, is committed. The notebook also saves intermediate checkpoints (baseline, MoCo, MAE-style, DINO, joint) to `checkpoints/`; these are ignored by `.gitignore` and are recreated when the notebook is run.

Train
Open `ITCS_6169_8169_Assignment2.ipynb`, restart the kernel, and run all cells top to bottom. The SSL data loaders use `num_workers=0` because MoCo hung on Windows with multiple workers. Section 8 picks the best model by validation accuracy, evaluates it once on the test set, and saves `checkpoints/resnet18_ssl_final.pth`.

Evaluate the saved checkpoint
Run the cells in Sections 1 to 4 (imports, paths, data, model and evaluation utilities), then run the last cell of Section 8:

```
ckpt = torch.load('checkpoints/resnet18_ssl_final.pth', map_location=device)
model_final = build_resnet18(ckpt['num_classes'], ckpt['config']['model']['dropout']).to(device)
model_final.load_state_dict(ckpt['model_state_dict'])
loss, acc = evaluate(model_final, test_loader, device)
print(f'Test accuracy from saved checkpoint: {acc:.4f}')
```

This reproduces the reported test accuracy of 0.7725 without retraining.

Hyperparameters
Shared: randomly initialized ResNet18 (no pretrained weights), dropout 0.3 before the classifier, 224px input, batch size 32, 80/20 train/validation split (seed 0), SGD with momentum 0.9 and weight decay 1e-4, cosine LR schedule.

* Baseline (supervised only): lr 0.01, up to 150 epochs, early stopping patience 15, label smoothing 0.1, Mixup/CutMix alpha 1.0
* MoCo pretraining: 2,048-entry queue, momentum 0.999, temperature 0.2, lr 0.03, 100 epochs, 128-d projector (hidden 512)
* MAE-style pretraining: 32px patches, 75% mask ratio, lr 0.01, up to 100 epochs, patience 15
* DINO pretraining: 2,048 prototypes, 2 global (224px) + 2 local (96px) crops, student/teacher temperature 0.1/0.04, teacher EMA 0.99 to 1.0, lr 0.03, 5 warmup epochs, 100 epochs, kNN check every 5 epochs
* Joint: classification loss + 0.5 x rotation-prediction loss, lr 0.01, up to 150 epochs, patience 15, label smoothing 0.1, Mixup/CutMix alpha 1.0 on the classification branch
* Fine-tuning after MoCo, MAE-style and DINO pretraining: lr 0.01, up to 100 epochs, early stopping patience 10, label smoothing 0.1, Mixup/CutMix alpha 1.0
