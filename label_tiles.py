"""Interactive multi-species tile labeler.

Divide each frame into a grid; paint tiles with the ACTIVE species. Add species
on the fly via the in-window text box, switch with number keys, each species a
distinct color. Labels saved per-image as JSON; species list persists in
species.json. Resumes and loads the bootstrapped cogongrass labels.

Run:  python label_tiles.py

Controls:
  left click          paint active species onto a tile (click same again = erase)
  right click         erase a tile
  1..9                select active species
  type in "New species" box + Enter   add a new species (becomes active)
  n / Enter / Spc     save & next image
  b                   save & previous image
  c                   clear all tiles on this image
  q                   save & quit
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import TextBox
from PIL import Image

IMG_DIR = Path("drone_dataset/images")
OUT_DIR = Path("tile_labels")
SPECIES_FILE = OUT_DIR / "species.json"
TILE = 160
PALETTE = ["lime", "red", "deepskyblue", "yellow", "magenta", "orange",
           "springgreen", "white", "violet", "cyan"]


def load_species():
    if SPECIES_FILE.exists():
        return json.loads(SPECIES_FILE.read_text()).get("species", ["cogongrass"])
    return ["cogongrass"]


def save_species(species):
    OUT_DIR.mkdir(exist_ok=True)
    SPECIES_FILE.write_text(json.dumps({"species": species}))


class Labeler:
    def __init__(self, images):
        self.images = images
        self.idx = 0
        self.species = load_species()
        self.active = 0
        self.tiles = {}                       # (r, c) -> species name

        self.fig = plt.figure(figsize=(13, 9))
        self.ax = self.fig.add_axes([0.03, 0.10, 0.94, 0.82])
        tb_ax = self.fig.add_axes([0.30, 0.025, 0.40, 0.045])
        self.textbox = TextBox(tb_ax, "New species: ")
        self.textbox.on_submit(self.on_submit)

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.load()
        plt.show()

    def color(self, name):
        i = self.species.index(name) if name in self.species else 0
        return PALETTE[i % len(PALETTE)]

    def label_path(self, img):
        return OUT_DIR / f"{img.stem}.json"

    def load(self):
        img = self.images[self.idx]
        self.im = Image.open(img).convert("RGB")
        self.W, self.H = self.im.size
        self.cols = -(-self.W // TILE)
        self.rows = -(-self.H // TILE)
        self.tiles = {}
        lp = self.label_path(img)
        if lp.exists():
            d = json.loads(lp.read_text())
            if "tiles" in d:                  # new multi-species format
                self.tiles = {tuple(int(x) for x in k.split(",")): v for k, v in d["tiles"].items()}
            elif "cogongrass" in d:           # backward-compat: old binary format
                self.tiles = {tuple(t): "cogongrass" for t in d["cogongrass"]}
        for name in set(self.tiles.values()):
            if name not in self.species:
                self.species.append(name)
        self.draw()

    def draw(self):
        self.ax.clear()
        self.ax.imshow(self.im)
        for c in range(self.cols + 1):
            self.ax.axvline(c * TILE, color="white", lw=0.5, alpha=0.4)
        for r in range(self.rows + 1):
            self.ax.axhline(r * TILE, color="white", lw=0.5, alpha=0.4)
        for (r, c), name in self.tiles.items():
            col = self.color(name)
            self.ax.add_patch(Rectangle((c * TILE, r * TILE), TILE, TILE,
                                        facecolor=col, alpha=0.40, edgecolor=col, lw=2))
        legend = "   ".join(f"[{i + 1}]{'>' if i == self.active else ' '}{s}"
                            for i, s in enumerate(self.species))
        self.ax.set_title(
            f"[{self.idx + 1}/{len(self.images)}]  {self.images[self.idx].name}\n"
            f"ACTIVE: {self.species[self.active]}    species:  {legend}\n"
            "click=paint  right-click=erase  1-9=select  n=next  b=back  c=clear  q=quit",
            fontsize=9)
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def on_submit(self, text):
        name = (text or "").strip()
        if name:
            if name not in self.species:
                self.species.append(name)
                save_species(self.species)
            self.active = self.species.index(name)
            self.textbox.set_val("")          # clear box (re-fires on_submit with "" -> no-op)
            self.draw()

    def on_click(self, e):
        if e.inaxes != self.ax or e.xdata is None:
            return
        c, r = int(e.xdata // TILE), int(e.ydata // TILE)
        if not (0 <= c < self.cols and 0 <= r < self.rows):
            return
        key = (r, c)
        name = self.species[self.active]
        if e.button == 3 or self.tiles.get(key) == name:   # right-click or same -> erase
            self.tiles.pop(key, None)
        else:
            self.tiles[key] = name
        self.draw()

    def save(self):
        img = self.images[self.idx]
        OUT_DIR.mkdir(exist_ok=True)
        self.label_path(img).write_text(json.dumps({
            "image": img.name, "tile_px": TILE, "rows": self.rows, "cols": self.cols,
            "tiles": {f"{r},{c}": n for (r, c), n in sorted(self.tiles.items())},
        }))

    def on_key(self, e):
        if getattr(self.textbox, "capturekeystrokes", False):   # typing in the box -> ignore hotkeys
            return
        if e.key and e.key.isdigit() and e.key != "0":
            i = int(e.key) - 1
            if i < len(self.species):
                self.active = i; self.draw()
        elif e.key in ("n", "enter", " ", "right"):
            self.save()
            if self.idx < len(self.images) - 1:
                self.idx += 1; self.load()
            else:
                print("reached last image - saved."); plt.close(self.fig)
        elif e.key in ("b", "left"):
            self.save()
            if self.idx > 0:
                self.idx -= 1; self.load()
        elif e.key == "c":
            self.tiles.clear(); self.draw()
        elif e.key == "q":
            self.save(); save_species(self.species); print("saved & quit."); plt.close(self.fig)


def main():
    all_imgs = sorted(IMG_DIR.glob("*.jpg"))
    if not all_imgs:
        print(f"no images in {IMG_DIR}"); return
    print("multi-species tile labeler: type a name in the 'New species' box + Enter to add, "
          "1-9 to switch, click to paint, n=next, q=quit.")
    Labeler(all_imgs)


if __name__ == "__main__":
    main()
