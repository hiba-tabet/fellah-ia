import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import classification_report

DATA = Path("data/prepared")
PLANTDOC = Path("data/plantdoc")

EPOCHS = 12
BS = 32
LR = 5e-4

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

CLASSES = json.load(open("models/classes.json"))

PD_MAP = {
    "Potato leaf early blight": "potato_early_blight",
    "Potato leaf late blight": "potato_late_blight",
    "Tomato Early blight leaf": "tomato_early_blight",
    "Tomato Septoria leaf spot": "tomato_septoria_leaf_spot",
    "Tomato leaf": "tomato_healthy",
    "Tomato leaf bacterial spot": "tomato_bacterial_spot",
    "Tomato leaf late blight": "tomato_late_blight",
    "Tomato leaf mosaic virus": "tomato_mosaic_virus",
    "Tomato leaf yellow virus": "tomato_yellow_leaf_curl_virus",
    "Tomato mold leaf": "tomato_leaf_mold",
    "Tomato two spotted spider mites leaf": "tomato_spider_mites",
}

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class PlantDocDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []

        for folder, cls in PD_MAP.items():
            d = Path(root) / folder
            if not d.exists():
                continue

            label = CLASSES.index(cls)

            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix.lower() in EXTS:
                    self.samples.append((f, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            preds.extend(out.argmax(1).cpu().tolist())
            labels.extend(y.tolist())

    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    return acc, preds, labels


def main():
    print("Device:", device)
    print("Classes:", len(CLASSES))

    norm = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            224, scale=(0.35, 1.0), ratio=(0.75, 1.33)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomPerspective(0.25, p=0.4),
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.08),
        transforms.RandomGrayscale(0.05),
        transforms.RandomApply(
            [transforms.GaussianBlur(5, (0.1, 2.0))],
            p=0.4
        ),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(
            p=0.4, scale=(0.02, 0.2), value="random"
        ),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm,
    ])

    # Dataset principal
    base_ds = datasets.ImageFolder(DATA / "train", train_tf)

    # PlantDoc supplémentaire
    pd_ds = PlantDocDataset(
        PLANTDOC / "train",
        train_tf
    )

    print("Base train:", len(base_ds))
    print("PlantDoc train ajoute:", len(pd_ds))

    # Mélange des deux datasets
    train_ds = torch.utils.data.ConcatDataset([base_ds, pd_ds])

    # Validation / test propres de notre dataset
    val_ds = datasets.ImageFolder(DATA / "val", eval_tf)
    test_ds = datasets.ImageFolder(DATA / "test", eval_tf)

    # PlantDoc test : jamais utilisé pour l'entraînement
    pd_test_ds = PlantDocDataset(
        PLANTDOC / "test",
        eval_tf
    )

    print("Combined train:", len(train_ds))
    print("Val propre:", len(val_ds))
    print("Test propre:", len(test_ds))
    print("PlantDoc test:", len(pd_test_ds))

    # Construire les labels pour le sampler
    base_targets = list(base_ds.targets)
    pd_targets = [label for _, label in pd_ds.samples]
    all_targets = torch.tensor(base_targets + pd_targets)

    counts = torch.bincount(
        all_targets,
        minlength=len(CLASSES)
    ).float()

    weights = torch.zeros_like(counts)
    weights[counts > 0] = 1.0 / counts[counts > 0]

    sample_weights = weights[all_targets]

    sampler = WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True
    )

    train_dl = DataLoader(
        train_ds,
        BS,
        sampler=sampler,
        num_workers=4,
        persistent_workers=True
    )

    val_dl = DataLoader(
        val_ds,
        64,
        num_workers=4,
        persistent_workers=True
    )

    test_dl = DataLoader(
        test_ds,
        64,
        num_workers=4
    )

    pd_dl = DataLoader(
        pd_test_ds,
        64,
        num_workers=4
    )

    model = timm.create_model(
        "mobilenetv3_large_100",
        pretrained=True,
        num_classes=len(CLASSES)
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=EPOCHS
    )

    best_real = 0.0
    best_val = 0.0

    for ep in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        total = 0.0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

            total += loss.item() * len(y)

        sched.step()

        val_acc, _, _ = evaluate(model, val_dl)
        real_acc, _, _ = evaluate(model, pd_dl)

        print(
            f"Epoch {ep}/{EPOCHS} "
            f"loss={total/len(train_ds):.3f} "
            f"val_acc={val_acc:.4f} "
            f"val_reel_top1={real_acc:.4f} "
            f"time={time.time()-t0:.0f}s",
            flush=True
        )

        # Priorité à la généralisation PlantDoc
        if real_acc > best_real:
            best_real = real_acc
            best_val = val_acc
            torch.save(
                model.state_dict(),
                "models/best_v3.pth"
            )

    model.load_state_dict(
        torch.load(
            "models/best_v3.pth",
            map_location="cpu"
        )
    )

    test_acc, preds, labels = evaluate(model, test_dl)
    real_acc, real_preds, real_labels = evaluate(model, pd_dl)

    print("\nRÉSULTATS FINAUX")
    print(f"BEST PlantDoc top1={best_real:.4f}")
    print(f"Val propre={best_val:.4f}")
    print(f"Test propre={test_acc:.4f}")
    print(f"PlantDoc test top1={real_acc:.4f}")

    print("\nRAPPORT TEST PROPRE")
    print(
        classification_report(
            labels,
            preds,
            target_names=CLASSES,
            digits=3,
            zero_division=0
        )
    )

    print("\nRAPPORT PLANTDOC")
    real_classes = sorted(set(real_labels))
    real_names = [CLASSES[i] for i in real_classes]
    print(
        classification_report(
            real_labels,
            real_preds,
            labels=real_classes,
            target_names=real_names,
            digits=3,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()
