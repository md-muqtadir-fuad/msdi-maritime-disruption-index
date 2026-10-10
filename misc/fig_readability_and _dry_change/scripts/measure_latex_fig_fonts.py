import sys, os, math, collections
from pypdf import PdfReader

TEXTWIDTH_BP = 6.5 * 72  # letter paper, 1in margins

def mult(a, b):
    return [a[0]*b[0]+a[1]*b[2], a[0]*b[1]+a[1]*b[3],
            a[2]*b[0]+a[3]*b[2], a[2]*b[1]+a[3]*b[3],
            a[4]*b[0]+a[5]*b[2]+b[4], a[4]*b[1]+a[5]*b[3]+b[5]]

for path in sys.argv[1:]:
    r = PdfReader(path)
    page = r.pages[0]
    w = float(page.mediabox.width); h = float(page.mediabox.height)
    scale = TEXTWIDTH_BP / w
    sizes = collections.Counter()
    def visit(text, cm, tm, fdict, fsize):
        if not text or not text.strip():
            return
        m = mult(tm, cm)
        eff = fsize * math.hypot(m[2], m[3])  # vertical scale of text space
        sizes[round(eff, 1)] += len(text.strip())
    try:
        page.extract_text(visitor_text=visit)
    except Exception as e:
        print(f"{os.path.basename(path)}: ERROR {e}")
        continue
    if not sizes:
        print(f"{os.path.basename(path)}: native {w:.0f}x{h:.0f}bp, scale {scale:.2f}, NO TEXT (outlined/raster)")
        continue
    native = sorted(sizes)
    printed = sorted({round(s*scale, 1) for s in native})
    common = sizes.most_common(1)[0][0]
    print(f"{os.path.basename(path)}: native {w:.0f}x{h:.0f}bp, scale {scale:.2f} | "
          f"native pt {native[0]}-{native[-1]} (most text {common}) | "
          f"printed pt {printed[0]}-{printed[-1]} (most text {round(common*scale,1)})")
