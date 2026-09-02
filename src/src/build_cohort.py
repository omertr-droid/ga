"""Build the GA validation set from the advRPE cohort zip:
  1) ga_cohort_manifest.csv  - one row per eye (subject, visit, eye, date, advRPE area, mask area, QC ratio)
  2) cohort_masks/*.png       - clean BINARY GA masks (filled) recovered from the yellow GA_seg_outline PNGs
Each mask's filled area is cross-checked against the advRPE Total_GA_Area to validate extraction."""
import zipfile, csv, io, re, os
from paths import REPO_ROOT as ROOT, DATA_DIR, RESULTS_DIR
import numpy as np
from PIL import Image
try:
    from scipy import ndimage as ndi
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

ZIP = os.path.join(DATA_DIR, "Zeiss GA Algorithm Run PLEX 6x6 only.zip")
OUTDIR = ROOT
MASKDIR = os.path.join(OUTDIR, "cohort_masks")
os.makedirs(MASKDIR, exist_ok=True)

z = zipfile.ZipFile(ZIP)

# group entries by (subject, eye)
groups = {}   # (nh, eye) -> {"outline": name, "csv_area": float, "date":, "pid":, "mm":}
for n in z.namelist():
    m = re.search(r"(NHAMD-\d+-\d+-V\d+).*?_6x6_(O[DS])_\d{8}_\d{6}", n)
    if not m:
        continue
    key = (m.group(1), m.group(2))
    g = groups.setdefault(key, {})
    low = n.lower()
    if low.endswith("ga_seg_outline.png"):
        g["outline"] = n
    elif low.endswith(".csv"):
        try:
            data = z.read(n).decode("utf-8", "replace").replace("\x00", "")
        except Exception:
            continue
        if "Total_GA_Area" in data:
            r = list(csv.reader(io.StringIO(data)))
            d = dict(zip(r[0], r[1]))
            g["csv_area"] = float(d.get("Total_GA_Area", "nan") or "nan")
            g["date"] = d.get("acquisitionDateTime", "")
            g["pid"] = d.get("patientID", "")
            g["mmX"] = float(d.get("mmX", 6) or 6)
            g["mmY"] = float(d.get("mmY", 6) or 6)

def fill_outline(rgb):
    """Return (binary_mask, n_tint_px). GA is a translucent YELLOW FILL (R~=G, B reduced),
    not a thin outline, so detect the yellow-tinted pixels directly."""
    a = rgb.astype(np.int16)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    tint = (R - B > 12) & (G - B > 12) & (np.abs(R - G) < 12)   # desaturated yellow overlay
    n = int(tint.sum())
    if n == 0:
        return np.zeros(rgb.shape[:2], bool), 0
    if HAVE_SCIPY:                                              # fill interior speckles only
        tint = ndi.binary_fill_holes(tint)
    return tint, n

rows = []
for (nh, eye), g in sorted(groups.items()):
    rec = {"subject": nh, "visit": nh.split("-")[-1], "eye": eye,
           "date": g.get("date", ""), "patientID": g.get("pid", ""),
           "advRPE_area_mm2": g.get("csv_area", float("nan"))}
    if "outline" in g:
        rgb = np.array(Image.open(io.BytesIO(z.read(g["outline"]))).convert("RGB"))
        h, w = rgb.shape[:2]
        mm_px2 = (g.get("mmX", 6) / w) * (g.get("mmY", 6) / h)
        mask, n_out = fill_outline(rgb)
        mask_area = float(mask.sum() * mm_px2)
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            os.path.join(MASKDIR, f"{nh}_{eye}_GAmask.png"))
        csv_a = g.get("csv_area", float("nan"))
        ratio = (mask_area / csv_a) if (csv_a and csv_a > 0.02) else float("nan")
        rec.update({"img_w": w, "img_h": h, "mask_area_mm2": round(mask_area, 4),
                    "mask_px": int(mask.sum()), "outline_px": n_out,
                    "area_ratio_mask_over_csv": round(ratio, 3) if ratio == ratio else ""})
    else:
        rec.update({"img_w": "", "img_h": "", "mask_area_mm2": "", "mask_px": "",
                    "outline_px": "", "area_ratio_mask_over_csv": ""})
    rows.append(rec)

cols = ["subject", "visit", "eye", "date", "advRPE_area_mm2", "mask_area_mm2",
        "area_ratio_mask_over_csv", "mask_px", "outline_px", "img_w", "img_h", "patientID"]
man = os.path.join(RESULTS_DIR, "ga_cohort_manifest.csv")
with open(man, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in cols})

print(f"scipy available: {HAVE_SCIPY}")
print(f"\n{'subject':<18}{'eye':<4}{'advRPE':>9}{'mask':>9}{'ratio':>8}")
for r in rows:
    print(f"{r['subject']:<18}{r['eye']:<4}{r['advRPE_area_mm2']:>9.3f}"
          f"{(r['mask_area_mm2'] if r['mask_area_mm2']!='' else float('nan')):>9}"
          f"{str(r['area_ratio_mask_over_csv']):>8}")
print(f"\nWrote {man}")
print(f"Wrote {len([r for r in rows if r['mask_px']!=''])} masks -> {MASKDIR}")
z.close()
