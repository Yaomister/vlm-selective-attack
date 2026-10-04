import shutil
from pathlib import Path
import fiftyone as fo
import fiftyone.zoo as foz

# images/safe inside the project root, regardless of where you run this from
ZOO_DIR = Path(__file__).resolve().parent.parent / "images" / "safe"
OLD = Path.home() / "fiftyone" / "coco-2017"
NEW = ZOO_DIR / "coco-2017"

ZOO_DIR.mkdir(parents=True, exist_ok=True)

# move the complete copy from your first run, replacing the broken download
if OLD.exists():
    shutil.rmtree(NEW, ignore_errors=True)
    shutil.move(str(OLD), str(NEW))

# delete any leftover broken zips
for z in NEW.rglob("*.zip"):
    z.unlink()

fo.config.dataset_zoo_dir = str(ZOO_DIR)

dataset = foz.load_zoo_dataset(
    "coco-2017",
    splits=["validation"],
    drop_existing_dataset=True,  # rebuild so FiftyOne points to the new location
)
print(dataset)
