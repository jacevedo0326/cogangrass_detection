"""Generate a comprehensive PDF report of the cogongrass detection project."""
from fpdf import FPDF

ACCENT = (30, 90, 50)
GREY = (90, 90, 90)


class PDF(FPDF):
    def multi_cell(self, *args, **kwargs):
        kwargs.setdefault("new_x", "LMARGIN")
        kwargs.setdefault("new_y", "NEXT")
        return super().multi_cell(*args, **kwargs)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8); self.set_text_color(*GREY)
        self.cell(0, 6, "Cogongrass Detection - Project Report", align="L")
        self.cell(0, 6, f"p.{self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def h1(self, t):
        self.ln(3); self.set_font("Helvetica", "B", 15); self.set_text_color(*ACCENT)
        self.multi_cell(0, 8, t); self.set_draw_color(*ACCENT); self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y()); self.ln(3)

    def h2(self, t):
        self.ln(2); self.set_font("Helvetica", "B", 12); self.set_text_color(20, 20, 20)
        self.multi_cell(0, 6, t); self.ln(1)

    def body(self, t):
        self.set_font("Helvetica", "", 10); self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, t); self.ln(1)

    def bullet(self, t):
        self.set_font("Helvetica", "", 10); self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.cell(5, 5, "-")
        self.multi_cell(0, 5, t)
        self.set_x(x)

    def table(self, headers, rows, widths):
        self.set_font("Helvetica", "B", 8.5); self.set_fill_color(*ACCENT); self.set_text_color(255, 255, 255)
        for w, hdr in zip(widths, headers):
            self.cell(w, 6, hdr, border=1, align="C", fill=True)
        self.ln()
        self.set_font("Helvetica", "", 8.5); self.set_text_color(20, 20, 20)
        fill = False
        for row in rows:
            self.set_fill_color(238, 244, 238) if fill else self.set_fill_color(255, 255, 255)
            for w, cell in zip(widths, row):
                self.cell(w, 5.5, str(cell), border=1, align="C", fill=True)
            self.ln(); fill = not fill
        self.ln(2)


p = PDF()
p.set_auto_page_break(auto=True, margin=15)
p.add_page()

# ---- Title ----
p.ln(40)
p.set_font("Helvetica", "B", 24); p.set_text_color(*ACCENT)
p.multi_cell(0, 12, "Cogongrass Detection from Drone Imagery", align="C")
p.set_font("Helvetica", "", 13); p.set_text_color(*GREY)
p.multi_cell(0, 8, "Project Report: Models, Results, Findings & Roadmap", align="C")
p.ln(6); p.set_font("Helvetica", "I", 10)
p.multi_cell(0, 6, "Generated June 2026", align="C")
p.ln(10)
p.set_font("Helvetica", "", 10); p.set_text_color(30, 30, 30)
p.multi_cell(0, 5, "Bottom line: a deployable tile-classification model detects cogongrass on a "
            "genuinely unseen drone collection at ~0.84 balanced accuracy (with AdaBN test-time "
            "adaptation). Architecture and preprocessing are exhausted; the remaining bottlenecks "
            "are DATA DIVERSITY and LABEL QUALITY, not the model.", align="C")

# ---- 1. Executive summary ----
p.add_page(); p.h1("1. Executive Summary")
p.body("Goal: automatically detect the invasive grass cogongrass (Imperata cylindrica) in drone "
       "imagery. The project evolved from an initial detect-then-classify plan into a TILE "
       "CLASSIFICATION pipeline: cut each frame into a grid, classify each tile cogongrass / not, "
       "stitch into a coverage heatmap.")
p.h2("What works")
for b in ["Deployable model: ResNet18 + domain-adaptation training + AdaBN ~= 0.84 balanced "
          "accuracy on a held-out collection (true cross-collection test).",
          "Reproducible pipeline: import -> tile -> train -> AdaBN inference -> heatmap.",
          "Honest evaluation: cross-collection (train one flight, test another) instead of a "
          "misleading random split that inflated scores to ~0.97.",
          "Test-time AdaBN is the single most effective technique (+2 pts, free, deployment-realistic)."]:
    p.bullet(b)
p.h2("The core finding")
p.body("Every architecture and preprocessing lever we tried converges to ~0.82-0.84 cross-collection "
       "while in-collection stays ~0.97. That flat ceiling across two backbones (ResNet, DINOv2) and "
       "every knob proves the bottleneck is DATA, not the model. Separately, visual inspection showed "
       "most 'false positives' are unlabeled cogongrass -> the metrics are a floor limited by LABEL QUALITY.")

# ---- 2. Data ----
p.add_page(); p.h1("2. Data")
p.h2("Close-up reference data (web)")
p.body("ResNet50 trained on iNaturalist close-ups: cogongrass vs 10 confuser grass species. "
       "Test 91.9% - but it is a DIFFERENT domain (sharp close-ups) and does not transfer to oblique/aerial drone crops.")
p.h2("Drone imagery (the real target)")
for b in ["Format: YOLO-labeled frames, 4096x3072 (12 MP), DJI camera (24 mm-equiv).",
          "Latest dataset (D:/data/invasive): 1095 images = 655 with cogongrass boxes + 440 negative "
          "(no cogongrass). Single class 'cogon'.",
          "Combines TWO flights/dates: 2026-04-22 (262 frames) and 2026-06-06 (819 frames) + 14 misc.",
          "Imagery is a near-continuous grass carpet; SAM2 confirmed it has NO separable plant "
          "instances (segments the whole field as one object) - which is why object detection was abandoned.",
          "Resolution: at ~30 ft flight, ground sample distance ~3.35 mm/px (cogongrass blade ~2-4 px wide - marginal)."]:
    p.bullet(b)
p.h2("Tiling")
p.body("Frames are cut into 512 px tiles (~1.7 m ground at 30 ft), each resized to 224 for the CNN. "
       "A green-index (ExG) filter drops sky / non-vegetation tiles. Tile labels are derived from the "
       "boxes: a tile is cogongrass if >=30% covered by a box. Negatives (440 frames) yield all-negative tiles.")

# ---- 3. Methodology ----
p.add_page(); p.h1("3. Methodology")
p.h2("Pipeline (scripts)")
p.table(["Script", "Role"],
        [["prep_images.py", "Import + (optionally) downscale frames; full-res supported (PREP_MAX=4096)"],
         ["boxes_to_tiles.py", "Cut tiles, ExG sky filter, derive tile labels from boxes"],
         ["train_tiles.py / _da.py", "Train tile classifier; _da adds CLAHE + domain-randomization aug + regularization"],
         ["train_tiles_collection.py", "Cross-collection split: train 0606, TEST held-out 0422"],
         ["train_tiles_dino_spatial.py", "DINOv2 patch-feature backbone + conv head"],
         ["tta_eval.py", "Test-time adaptation (AdaBN / TENT) evaluation"],
         ["threshold_sweep.py", "Recall/precision vs decision threshold"],
         ["heatmap_infer.py", "Deployable: tile -> AdaBN -> classify -> coverage heatmap"],
         ["label_tiles.py", "Interactive multi-species tile labeler"]],
        [55, 130])
p.h2("Evaluation protocol (critical)")
p.body("The honest metric is CROSS-COLLECTION: train on one flight, test on an entirely held-out "
       "flight. A random tile/frame split inflated accuracy to ~0.97 because the dataset is visually "
       "homogeneous (few distinct scenes) - that 0.97 does NOT predict performance on a new field.")

# ---- 4. Architecture ----
p.add_page(); p.h1("4. Model Architecture")
p.h2("Best deployable model: ResNet18 + DA + AdaBN")
p.body("Input 224x224 RGB tile -> ResNet18 backbone (ImageNet-pretrained, BatchNorm throughout) -> "
       "head: Dropout(0.4) + Linear(512 -> 2). Trained with class-weighted / balanced loss, label "
       "smoothing (0.1), AdamW, heavy augmentation. The BatchNorm layers are what enable AdaBN: at "
       "inference, BN running statistics are recomputed on the NEW field's tiles (no labels), which "
       "cancels cross-collection covariate shift.")
p.body("Flow:  tile(512px) -> resize 224 -> [Conv/BN/ReLU x ResNet18] -> global avg pool -> "
       "Dropout -> Linear -> softmax -> P(cogongrass).")
p.h2("Alternative: DINOv2-spatial")
p.body("Frozen DINOv2 ViT-S/14 -> 16x16x384 PATCH feature map -> conv head (Conv-BN-ReLU x2 -> "
       "global pool -> Dropout -> Linear). Uses spatial patch tokens (not the CLS vector - that "
       "mistake cost us 11 points). Competitive (~0.83) but did not beat the ResNet on our data.")
p.h2("Earlier approaches (superseded)")
for b in ["Close-up ResNet50 classifier (web data) - wrong domain for drone.",
          "YOLOv8 patch detector - detection is ill-posed for continuous grass (no object instances).",
          "DINOv2 frozen + CLS linear probe - wrong way to use DINOv2 (0.717)."]:
    p.bullet(b)

# ---- 5. Results ----
p.add_page(); p.h1("5. Results Across All Models")
p.h2("Headline models")
p.table(["Model / approach", "Metric", "Score", "Notes"],
        [["Close-up classifier (ResNet50)", "test bal acc", "0.919", "web close-ups, wrong domain"],
         ["YOLOv8 patch detector", "test mAP50", "0.39", "detection ill-suited to grass"],
         ["Tile clf v1 (old data)", "bal acc", "0.97", "HOLLOW (negatives were sky)"],
         ["Tile clf v2 (random split)", "test bal acc", "0.967", "INFLATED by homogeneity"]],
        [70, 35, 25, 55])
p.h2("Cross-collection results (the honest numbers; train 0606, test held-out 0422)")
p.table(["Configuration", "SOURCE", "+ AdaBN"],
        [["Baseline (blurry 1280, no DA)", "0.804", "-"],
         ["DA: CLAHE + aug + reg (blurry)", "0.817", "-"],
         ["Full-res 256 px, no CLAHE", "0.793", "-"],
         ["Full-res 256 px, CLAHE", "0.794", "-"],
         ["Full-res 512 px, no CLAHE", "0.815", "0.839"],
         ["Full-res 512 px, CLAHE (best)", "0.820", "0.838"],
         ["DINOv2 frozen, CLS-linear", "0.717", "-"],
         ["DINOv2-spatial + conv head", "0.801", "0.828"]],
        [95, 45, 45])
p.body("AdaBN consistently adds ~+2 pts. TENT test-time adaptation COLLAPSED (~0.53). Best deployable "
       "= 512 px + AdaBN ~= 0.84. The ~0.82-0.84 ceiling holds across BOTH backbones and every knob.")
p.h2("Decision-threshold sweep (best model + AdaBN, held-out 0422)")
p.table(["Threshold", "Recall", "Precision", "Cogongrass missed"],
        [["0.50 (default)", "0.87", "0.65", "13%"],
         ["0.30", "0.92", "0.59", "8%"],
         ["0.20 (recommended)", "0.94", "0.55", "6%"],
         ["0.15", "0.95", "0.52", "5%"],
         ["0.10", "0.96", "0.48", "4%"]],
        [40, 35, 35, 45])
p.body("False negatives (missed cogongrass) are far costlier than false positives for an invasive "
       "species, so operate at a LOW threshold (~0.20): catch 94% of cogongrass.")

# ---- 6. Key findings ----
p.add_page(); p.h1("6. Key Findings & Lessons")
for b in ["DOMAIN SHIFT is the core problem. In-collection ~0.97 vs cross-collection ~0.84. Random "
          "splits lie; always evaluate train-one-collection / test-another.",
          "ARCHITECTURE IS TAPPED OUT. Augmentation, CLAHE, resolution, tile size, regularization, and "
          "two backbones (ResNet + DINOv2) all land at ~0.82-0.84.",
          "DATA DIVERSITY is the dominant lever. Only 2 collections, visually homogeneous.",
          "LABELS ARE NOISY. Isolated 'false positives' are visually indistinguishable from labeled "
          "cogongrass -> most are UNLABELED cogongrass. Measured precision (~0.55) is a pessimistic floor; "
          "true performance is likely better.",
          "RECALL is the stuck metric (~0.64-0.74 raw; ~0.87+ with AdaBN at 0.5; up to 0.94 at threshold 0.20). "
          "Lowering the threshold buys recall cheaply.",
          "AdaBN is the best single technique - free, deployment-realistic, stacks with everything.",
          "Bigger tiles (512, more context) beat smaller (256, finer texture) for this task.",
          "512px tiles: false positives cluster near real cogongrass (61% vs 45% baseline) -> boundary/label effect."]:
    p.bullet(b); p.ln(1)

# ---- 7. Routes & next steps ----
p.add_page(); p.h1("7. All Possible Routes & Next Steps")
p.h2("A. Data (highest value - this is the real bottleneck)")
for b in ["MORE COLLECTIONS / SITES: the single biggest lever. Each new field de-confounds the model.",
          "FLY LOWER (~20 ft) + higher-res sensor: cogongrass blade texture needs resolution "
          "(Purdue & BASF papers fly ~16-20 ft at 0.5-1.7 mm/px vs our 3.35).",
          "MULTI-TEMPORAL RGB (phenology): fly the SAME fields 2-3x across the season. Cogongrass has "
          "distinctive timing (early green-up, persistent thatch, reddish senescence). Literature: beats "
          "adding spectral bands for telling similar grasses apart. Uses existing camera.",
          "MODEL-ASSISTED LABEL CLEANING: pre-fill tiles with model predictions, human corrects. Fixes "
          "the label gaps inflating false positives; gives a TRUE performance number."]:
    p.bullet(b); p.ln(1)
p.h2("B. Sensors")
for b in ["MULTISPECTRAL (NIR + red-edge): NDVI for vegetation-soil separation (the #1 cross-domain "
          "failure mode), NDRE/red-edge for grass-vs-grass. More illumination-invariant -> helps domain "
          "shift. Combine via channel-stacking (RGB+NIR+indices); adapt the first conv layer. Needs a "
          "multispectral sensor + reflight + band registration; store tiles as TIFF/npy."]:
    p.bullet(b); p.ln(1)
p.h2("C. Model / algorithm (lower priority - architecture is tapped out)")
for b in ["DINOv2 SEGMENTATION with a dense decoder (needs pixel labels) - the BASF SOTA for ground->aerial shift.",
          "Domain-generalization aug: MixStyle, Fourier/amplitude-mix (style) augmentation.",
          "IBN-Net (InstanceNorm) for style-invariant features; self-supervised pretrain on unlabeled drone frames.",
          "Fix TENT (lower lr / class-balanced entropy) to stack on AdaBN."]:
    p.bullet(b); p.ln(1)
p.h2("D. Inference / deployment")
for b in ["Operate at threshold ~0.20 (prioritize recall - don't miss the invasive).",
          "Merge adjacent positive tiles into stands -> boundary false-positives vanish at field level.",
          "Overlapping tiles at inference + hierarchical fallback (flag low-confidence tiles for human review)."]:
    p.bullet(b); p.ln(1)

# ---- 8. Recommended path ----
p.add_page(); p.h1("8. Recommended Path Forward")
p.body("The model is in good shape and likely better than its metrics (label noise). Effort should "
       "now go to DATA and LABELS, not architecture:")
p.table(["Priority", "Action", "Why"],
        [["1", "Collect 1-2 more sites/collections", "De-confounds; the proven lever"],
         ["2", "Model-assisted label cleaning pass", "FPs are mostly unlabeled cogongrass; get true score"],
         ["3", "Fly ~20 ft + multi-temporal (phenology)", "Cheapest capture change; hits grass-vs-grass"],
         ["4", "Deploy ResNet-DA + AdaBN at thr 0.20", "Usable now; high recall"],
         ["5", "Add multispectral (NIR/red-edge)", "Veg-soil + domain robustness; needs hardware"]],
        [20, 80, 85])
p.body("Current deliverable: a validated ~0.84 cross-collection model with an AdaBN inference tool "
       "(heatmap_infer.py) and a full, reproducible pipeline. The project went from 'is this possible?' "
       "to a rigorously-measured, deployable system that knows exactly what it needs next.")

p.output("cogongrass_report.pdf")
print("wrote cogongrass_report.pdf")
