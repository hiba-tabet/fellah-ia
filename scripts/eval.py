import json
from pathlib import Path
import torch, timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
classes = json.load(open("models/classes.json"))
tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def main():
    model = timm.create_model("mobilenetv3_large_100", pretrained=False,
                              num_classes=len(classes))
    model.load_state_dict(torch.load("models/best.pth", map_location="cpu"))
    model.to(device).eval()
    for split in ("val", "test"):
        ds = datasets.ImageFolder(Path("data/prepared") / split, tf)
        dl = DataLoader(ds, 64, num_workers=0)
        preds, labels = [], []
        with torch.no_grad():
            for x, y in dl:
                preds += model(x.to(device)).argmax(1).cpu().tolist()
                labels += y.tolist()
        acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
        print(f"\n=== {split.upper()} accuracy = {acc:.4f} ===")
        if split == "test":
            print(classification_report(labels, preds, target_names=classes, digits=3))
            cm = confusion_matrix(labels, preds)
            print("Principales confusions (vrai -> prédit : nombre):")
            pairs = [(cm[i][j], classes[i], classes[j])
                     for i in range(len(classes)) for j in range(len(classes))
                     if i != j and cm[i][j] > 0]
            for n, a, b in sorted(pairs, reverse=True)[:8]:
                print(f"  {a} -> {b} : {n}")

if __name__ == "__main__":
    main()
