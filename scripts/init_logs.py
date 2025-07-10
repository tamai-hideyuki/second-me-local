import os
from datetime import date

d = date.today().strftime("%Y-%m-%d")
base = f"logs/{d}"
os.makedirs(base, exist_ok=True)
for n, name in [("01", "chat"), ("02", "thoughts"), ("03", "tasks")]:
    path = f"{base}/{n}_{name}.md"
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(f"# {name.title()}（{d}）\n\n")
print(f"Initialized log templates under {base}/")
