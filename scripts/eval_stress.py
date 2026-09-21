import sys, json
import torch, timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
classes = json.load(open("models/classes.json"))

norm = T.Normalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225]
)

clean = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    norm
])

stress = T.Compose([
    T.Resize((224, 224)),
    T.RandomAffine(25, scale=(0.55, 1.0), fill=(95, 105, 70)),
    T.ColorJitter(0.5, 0.5, 0.4, 0.05),
    T.RandomApply(
        [T.GaussianBlur(7, (0.5, 3.0))],
        p=0.5
    ),
    T.ToTensor(),
    norm,
    T.RandomErasing(
        p=0.5,
        scale=(0.02, 0.15),
        value="random"
    )
])

def run(model, tf):
    ds = datasets.ImageFolder("data/prepared/test", tf)
    assert ds.classes == classes

    torch.manual_seed(0)

    dl = DataLoader(
        ds,
        64,
        num_workers=4
    )

    ok, tot = {}, {}

    with torch.no_grad():
        for x, y in dl:
            p = model(x.to(device)).argmax(1).cpu()

            for pi, yi in zip(p.tolist(), y.tolist()):
                crop = classes[yi].split("_")[0]

                for k in (crop, "TOUT"):
                    tot[k] = tot.get(k, 0) + 1
                    ok[k] = ok.get(k, 0) + (pi == yi)

    return {
        k: ok[k] / tot[k]
        for k in tot
    }

def main():
    path = sys.argv[1]

    model = timm.create_model(
        "mobilenetv3_large_100",
        pretrained=False,
        num_classes=len(classes)
    )

    model.load_state_dict(
        torch.load(path, map_location="cpu")
    )

    model.to(device).eval()

    for name, tf in (
        ("PROPRE", clean),
        ("STRESS", stress)
    ):
        r = run(model, tf)

        print(
            f"{path} | {name:6s} | "
            + " | ".join(
                f"{k}={r[k]:.3f}"
                for k in ("TOUT", "olive", "tomato", "potato")
            )
        )

if __name__ == "__main__":
    main()
