import json, time
from pathlib import Path
import torch, torch.nn as nn, timm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import classification_report

DATA = Path("data/prepared")
EPOCHS, BS, LR = 10, 32, 5e-4
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            preds += out.argmax(1).cpu().tolist()
            labels += y.tolist()

    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    return acc, preds, labels


def main():
    print("Device:", device)

    norm = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )

    # Augmentations fortes pour améliorer la robustesse
    # aux photos prises dans des conditions réelles.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.35, 1.0),
            ratio=(0.75, 1.33)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomPerspective(0.25, p=0.4),
        transforms.ColorJitter(
            0.5, 0.5, 0.5, 0.08
        ),
        transforms.RandomGrayscale(0.05),
        transforms.RandomApply(
            [transforms.GaussianBlur(5, (0.1, 2.0))],
            p=0.4
        ),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(
            p=0.4,
            scale=(0.02, 0.2),
            value="random"
        )
    ])

    # Pas d'augmentation sur validation/test.
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm
    ])

    train_ds = datasets.ImageFolder(DATA / "train", train_tf)
    val_ds = datasets.ImageFolder(DATA / "val", eval_tf)
    test_ds = datasets.ImageFolder(DATA / "test", eval_tf)

    classes = train_ds.classes

    json.dump(
        classes,
        open("models/classes.json", "w"),
        indent=2
    )

    targets = torch.tensor(train_ds.targets)
    counts = torch.bincount(targets).float()

    sampler = WeightedRandomSampler(
        (1.0 / counts)[targets],
        len(targets),
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

    model = timm.create_model(
        "mobilenetv3_large_100",
        pretrained=True,
        num_classes=len(classes)
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

    best = 0.0

    for ep in range(1, EPOCHS + 1):
        model.train()
        t0, total = time.time(), 0.0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()

            loss = loss_fn(
                model(x),
                y
            )

            loss.backward()
            opt.step()

            total += loss.item() * len(y)

        sched.step()

        val_acc, _, _ = evaluate(
            model,
            val_dl
        )

        print(
            f"Epoch {ep}/{EPOCHS}  "
            f"loss={total/len(train_ds):.3f}  "
            f"val_acc={val_acc:.4f}  "
            f"time={time.time()-t0:.0f}s",
            flush=True
        )

        if val_acc > best:
            best = val_acc

            torch.save(
                model.state_dict(),
                "models/best_v2.pth"
            )

    model.load_state_dict(
        torch.load("models/best_v2.pth")
    )

    test_acc, preds, labels = evaluate(
        model,
        test_dl
    )

    print(
        f"\nBEST val_acc={best:.4f}   "
        f"TEST acc={test_acc:.4f}\n"
    )

    print(
        classification_report(
            labels,
            preds,
            target_names=classes,
            digits=3
        )
    )


if __name__ == "__main__":
    main()