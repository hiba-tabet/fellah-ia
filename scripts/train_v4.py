import sys, json, random, time
from pathlib import Path

import torch
import torch.nn as nn
import timm

from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T

sys.path.insert(0, "scripts")
from eval_real import plantdoc_label

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

classes = json.load(open("models/classes.json"))
cidx = {c: i for i, c in enumerate(classes)}

H = cidx["olive_healthy"]

D = [
    cidx["olive_aculus_olearius"],
    cidx["olive_peacock_spot"]
]

EPOCHS = 10
BS = 32
LR = 1e-4
PER_EPOCH = 6400

random.seed(42)
torch.manual_seed(42)

norm = T.Normalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225]
)

train_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.35, 1.0), ratio=(0.75, 1.33)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(30),
    T.RandomPerspective(0.25, p=0.4),
    T.ColorJitter(0.5, 0.5, 0.5, 0.08),
    T.RandomGrayscale(0.05),
    T.RandomApply(
        [T.GaussianBlur(5, (0.1, 2.0))],
        p=0.4
    ),
    T.ToTensor(),
    norm,
    T.RandomErasing(
        p=0.4,
        scale=(0.02, 0.2),
        value="random"
    )
])

eval_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    norm
])


class DS(Dataset):
    def __init__(self, items, tf):
        self.items = items
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def folder_items(root, mapper):
    items = []
    root = Path(root)

    if not root.exists():
        return items

    for d in sorted(root.iterdir()):
        if d.is_dir():
            c = mapper(d.name)

            if c is not None:
                items += [
                    (f, cidx[c])
                    for f in sorted(d.iterdir())
                    if f.suffix.lower() in EXTS
                ]

    return items


def zeytin_items(split):
    data = json.load(
        open("data/zeytin_split.json")
    )[split]

    return [
        (p, H if lab == "healthy" else -1)
        for p, lab in data
    ]


def make_weights(sources):
    w = []

    for items, share in sources:
        cnt = {}

        for _, y in items:
            cnt[y] = cnt.get(y, 0) + 1

        for _, y in items:
            w.append(
                share / (len(cnt) * cnt[y])
            )

    return w


def loss_fn(logits, y):
    logp = torch.log_softmax(logits, 1)

    normal = y >= 0

    loss = logits.new_zeros(())

    if normal.any():
        loss = loss + nn.functional.cross_entropy(
            logits[normal],
            y[normal],
            label_smoothing=0.1,
            reduction="sum"
        )

    dis = ~normal

    if dis.any():
        loss = loss - torch.logsumexp(
            logp[dis][:, D],
            dim=1
        ).sum()

    return loss / len(y)


def load(path):
    m = timm.create_model(
        "mobilenetv3_large_100",
        pretrained=False,
        num_classes=len(classes)
    )

    m.load_state_dict(
        torch.load(path, map_location="cpu")
    )

    return m.to(device).eval()


def probs_of(model, items):
    dl = DataLoader(
        DS(items, eval_tf),
        64,
        num_workers=0
    )

    model.eval()

    P = []
    Y = []

    with torch.no_grad():
        for x, y in dl:
            P.append(
                torch.softmax(
                    model(x.to(device)),
                    1
                ).cpu()
            )

            Y += y.tolist()

    return torch.cat(P), torch.tensor(Y)


def top1(model, items):
    P, Y = probs_of(model, items)

    return (
        (P.argmax(1) == Y)
        .float()
        .mean()
        .item()
    )


def zeytin_metrics(P, Y):
    pd_ = P[:, D].sum(1)
    ph = P[:, H]

    hl = Y == H

    rh = (
        (ph[hl] > pd_[hl])
        .float()
        .mean()
        .item()
    )

    rd = (
        (pd_[~hl] > ph[~hl])
        .float()
        .mean()
        .item()
    )

    return rh, rd, (rh + rd) / 2


def main():

    pv = folder_items(
        "data/prepared/train",
        lambda n: n if n in cidx else None
    )

    pd_all = folder_items(
        "data/plantdoc/train",
        plantdoc_label
    )

    random.shuffle(pd_all)

    nv = int(0.15 * len(pd_all))

    pd_val = pd_all[:nv]
    pd_train = pd_all[nv:]

    pd_test = folder_items(
        "data/plantdoc/test",
        plantdoc_label
    )

    z_tr = zeytin_items("train")
    z_val = zeytin_items("val")
    z_te = zeytin_items("test")

    print(
        f"PV={len(pv)} "
        f"PlantDoc train={len(pd_train)} "
        f"val={len(pd_val)} "
        f"test={len(pd_test)} "
        f"| Zeytin train={len(z_tr)} "
        f"val={len(z_val)} "
        f"test={len(z_te)} "
        f"| device={device}",
        flush=True
    )

    sampler = WeightedRandomSampler(
        make_weights([
            (pv, 0.4),
            (pd_train, 0.4),
            (z_tr, 0.2)
        ]),
        PER_EPOCH,
        replacement=True
    )

    dl = DataLoader(
        DS(
            pv + pd_train + z_tr,
            train_tf
        ),
        BS,
        sampler=sampler,
        num_workers=0
    )

    model = load("models/best_v3.pth")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=EPOCHS
    )

    best = -1.0

    for ep in range(1, EPOCHS + 1):

        model.train()

        t0 = time.time()
        tot = 0.0
        n = 0

        for x, y in dl:

            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()

            loss = loss_fn(
                model(x),
                y
            )

            loss.backward()
            opt.step()

            tot += loss.item() * len(y)
            n += len(y)

        sched.step()

        a = top1(
            model,
            pd_val
        )

        rh, rd, bal = zeytin_metrics(
            *probs_of(model, z_val)
        )

        score = 0.5 * a + 0.5 * bal

        print(
            f"Epoch {ep}/{EPOCHS} "
            f"loss={tot/n:.3f} "
            f"plantdoc_val={a:.3f} "
            f"zeytin_val: "
            f"saines={rh:.3f} "
            f"malades={rd:.3f} "
            f"equilibre={bal:.3f} "
            f"time={time.time()-t0:.0f}s",
            flush=True
        )

        if score > best:

            best = score

            torch.save(
                model.state_dict(),
                "models/best_v4.pth"
            )

    print(
        "\n===== COMPARAISON FINALE "
        "(tests jamais vus) =====",
        flush=True
    )

    clean = folder_items(
        "data/prepared/test",
        lambda n: n if n in cidx else None
    )

    for name, path in (
        ("v3", "models/best_v3.pth"),
        ("v4", "models/best_v4.pth")
    ):

        m = load(path)

        a = top1(
            m,
            pd_test
        )

        rh, rd, bal = zeytin_metrics(
            *probs_of(m, z_te)
        )

        c = top1(
            m,
            clean
        )

        print(
            f"{name}: "
            f"PlantDoc test top1={a:.3f} "
            f"(n={len(pd_test)}) | "
            f"Zeytin test "
            f"saines={rh:.3f} "
            f"malades={rd:.3f} "
            f"equilibre={bal:.3f} "
            f"(n={len(z_te)}) | "
            f"test propre={c:.3f} "
            f"(n={len(clean)})",
            flush=True
        )


if __name__ == "__main__":
    main()
