"""
Difumina los datos sensibles de las capturas del Planner, dejando la interfaz intacta.

Localiza el texto con OCR y difumina SOLO lo que coincide con datos sensibles:
nombres de personas, códigos y nombres de proyecto, clientes, correos e importes.
Después vuelve a pasar OCR sobre la imagen difuminada y avisa si algo sigue siendo legible.

Uso:  python sanitizar_capturas.py            (procesa todas las capturas)
      python sanitizar_capturas.py Fichajes   (solo las que contengan ese texto)
"""
import re
import sys
import unicodedata
from pathlib import Path

from PIL import Image, ImageFilter
from paddleocr import PaddleOCR

ORIGEN = Path(r"C:\Users\sperez\Desktop\Capturas planner")
DESTINO = Path(__file__).resolve().parent / "img"
DESTINO.mkdir(exist_ok=True)

# Nombres de pila frecuentes en la plantilla: si una caja empieza por uno de ellos, se difumina
NOMBRES = {
    "aaron", "alejandro", "byron", "carlos", "cristina", "david", "irune", "ivan", "jesus",
    "laura", "luis", "mayra", "oscar", "roberto", "sergio", "simon", "administrador",
}
PATRONES = [
    re.compile(r"\d{4,5}\.\d{2}"),                 # códigos de proyecto 00042.26 / 0042.26.01
    re.compile(r"\b[A-Z]{3}\d{5,}\b"),             # referencias tipo DZ136135
    re.compile(r"[\w.\-]+@[\w.\-]+"),              # correos
    re.compile(r"\d[\d.,]*\s*€"),                  # importes
    re.compile(r"(?i)\b(cliente|proveedor)\b"),    # etiquetas que suelen acompañar al dato
]


def sin_tildes(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def es_sensible(texto: str) -> bool:
    t = sin_tildes(texto).strip()
    if not t:
        return False
    if any(p.search(t) for p in PATRONES):
        return True
    palabras = [w.lower().strip(".,:;()") for w in t.split()]
    return any(w in NOMBRES for w in palabras)


def difuminar(img: Image.Image, caja, margen=2) -> None:
    x0, y0, x1, y1 = caja
    x0, y0 = max(0, int(x0) - margen), max(0, int(y0) - margen)
    x1, y1 = min(img.width, int(x1) + margen), min(img.height, int(y1) + margen)
    if x1 <= x0 or y1 <= y0:
        return
    region = img.crop((x0, y0, x1, y1))
    radio = max(4, (y1 - y0) // 2)
    img.paste(region.filter(ImageFilter.GaussianBlur(radio)), (x0, y0))


def cajas(ocr, ruta: Path):
    res = ocr.ocr(str(ruta))
    salida = []
    for pagina in res or []:
        if isinstance(pagina, dict):                      # PaddleOCR 3.x
            for poly, txt in zip(pagina.get("rec_polys", []), pagina.get("rec_texts", [])):
                xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                salida.append(((min(xs), min(ys), max(xs), max(ys)), txt))
        else:                                             # PaddleOCR 2.x
            for poly, (txt, _conf) in pagina:
                xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                salida.append(((min(xs), min(ys), max(xs), max(ys)), txt))
    return salida


def main() -> int:
    filtro = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    ocr = PaddleOCR(
        lang="es",
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        enable_mkldnn=False,   # bug PIR+oneDNN en paddle 3.3.x Windows/CPU (visto en Admin Finanzas)
    )
    for ruta in sorted(ORIGEN.glob("*.png")):
        if filtro and filtro not in ruta.stem.lower():
            continue
        img = Image.open(ruta).convert("RGB")
        encontradas = cajas(ocr, ruta)
        tocadas = [t for c, t in encontradas if es_sensible(t)]
        for caja, txt in encontradas:
            if es_sensible(txt):
                difuminar(img, caja)
        destino = DESTINO / (sin_tildes(ruta.stem).lower().replace(" ", "-") + ".png")
        img.save(destino)
        # Verificación: ¿queda algo sensible legible en la imagen ya difuminada?
        resto = [t for _c, t in cajas(ocr, destino) if es_sensible(t)]
        print(f"{ruta.name}: {len(tocadas)} cajas difuminadas -> {destino.name}"
              f" | comprobación posterior: {'LIMPIA' if not resto else 'REVISAR ' + str(resto[:6])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
