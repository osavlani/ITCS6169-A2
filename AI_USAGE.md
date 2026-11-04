# AI Usage

## Tool(s) used
Claude (Anthropic) — used as a coding/research assistant throughout the project: implementing the SSL pipelines, debugging, explaining unfamiliar methods, git setup, and drafting the README and report text.

## Representative prompts and how AI assisted

### 1. Report citations and verification
**Prompt:** "this is my report, do the references/citations like last time ... flag anything concerning you wanna address"

**Outcome:** AI added 16 numbered inline citations and a References list (SimCLR, MoCo, BYOL, SimSiam, DINO, MAE, BEiT, VAE, rotation prediction, ViT, ResNet, SGDR, label smoothing, Mixup, CutMix, Cole et al.), each with its arXiv ID so it can be checked directly. While doing this it also flagged problems in the report itself: the baseline architecture (ResNet18) was never named, "41% accuracy" was actually a kNN number, the paragraphs mixed validation and test numbers, and there was no limitations statement. I accepted those edits and kept the report within two pages up to the results table. Only one reference (Cole et al.) was checked against its arXiv page during the session. The other entries came from the AI's memory, so I check each arXiv ID against its listed title before submitting.

### 2. Git: `.gitignore` not working and the unrelated-histories push error
**Prompt:** "gitignore isnt working, data/ and .ipynb_checkpoints/ still show up when i do git add ."

**Outcome:** AI found that the `.gitignore` had been written as UTF-16 with a BOM (PowerShell's `echo >>` does this), so Git did not read it. I confirmed this with `Format-Hex .gitignore`, which showed the `FF FE` byte-order mark, and the file was rewritten as ASCII, after which `git status` no longer listed the dataset or notebook checkpoints. In the same setup the first `git push` was rejected because the GitHub repo already held a LICENSE commit and the remote URL pointed at a wrong repo name. AI explained this and the fix: `git remote set-url` to the correct repo, then `git pull origin main --allow-unrelated-histories` (including how to save the merge message in Vim with `:wq`) before pushing. Later it also gave the rule for committing only the final checkpoint (`checkpoints/*` plus `!checkpoints/resnet18_ssl_final.pth`).

### 3. Listing SSL approaches and what fits on 8 GB of VRAM
**Prompt:** "so what were all possible ssl approaches that you can list, and which all are feasible to implement on 8gb vram"

**Outcome:** AI listed the main SSL families: contrastive (SimCLR, MoCo), self-distillation (BYOL, SimSiam, DINO), clustering (SwAV), redundancy reduction (Barlow Twins, VICReg), masked modeling (MAE, BEiT), pretext tasks (rotation, inpainting, colorization), generative (AE/VAE), and joint objectives. For each it gave the idea, pros and cons, and whether it was realistic on an 8 GB laptop GPU. SimCLR was not feasible because it needs a very large batch. VAE sits in the same reconstruction family as MAE and was expected to give weaker features. BYOL and SimSiam are feasible on this hardware but are the same family as DINO (EMA teacher, no negatives), so they add little new, and BYOL is also sensitive to hyperparameters and BatchNorm with a collapse risk at batch size 32. This gave me the information I needed to pick which methods to implement myself (see the decision section below).

### 4. MoCo training stuck on Windows
**Prompt:** "moco seems stuck, been a while no epochs to be seen"

**Outcome:** AI identified the most likely cause as PyTorch DataLoader multiprocessing on Windows with `num_workers=4` and a custom dataset, and told me exactly which cells to change to `num_workers=0`. I did not take the diagnosis on trust: after the change MoCo trained its 100 epochs in about 2,257 seconds, which confirmed it. The same setting was applied to the MAE and DINO loaders before they were run, and the README notes why.

### 5. kNN encoder accuracy before fine-tuning
**Prompt:** "knn encoder accuracy before fine tuning, explain why its important"

**Outcome:** AI explained that the kNN probe freezes the encoder, extracts features for the train images, and classifies each validation image by its nearest neighbors (weighted cosine similarity, k=20, T=0.1). No weights are trained, so it measures the pretrained representation itself, separate from fine-tuning choices such as learning rate, early stopping and Mixup/CutMix. Results before fine-tuning: random init 41.0%, MoCo 55.4%, MAE-style 31.0%, DINO 70.5%. Compared with fine-tuned validation accuracy (MoCo 83.2%, MAE 79.2%, DINO 85.8%), the ranking of the three pretrained encoders matches: DINO > MoCo > MAE. The pretext kNN numbers are far below the fine-tuned ones because fine-tuning trains every layer with labels. MAE-style scored below random init on kNN yet still fine-tuned to 79.2%, so kNN is a useful sanity check but not a reliable predictor of final accuracy. The supervised baseline (80.7%) and joint model (84.9%) were trained with labels, so their kNN scores are not comparable to the pretext ones.

## Decision made independently

From the list of possible SSL approaches, I chose to run one method from each different family: MoCo (contrastive), MAE-style (masked modeling), DINO (self-distillation) and a joint rotation + classification model (multi-task with a pretext task). I picked these because I was interested in these particular methods and wanted the comparison to cover genuinely different ideas, instead of several variants of the same one. I left out SimCLR because its large-batch requirement does not fit on an 8 GB GPU, and BYOL, SimSiam and VAE because they overlap with a family I was already running (DINO and MAE).
