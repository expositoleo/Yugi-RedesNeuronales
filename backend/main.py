"""Back-end web del trabajo integrador (FastAPI).

Expone el modelo entrenado como servicio HTTP y sirve el front-end estatico.
Corresponde a la etapa de "deploy" del curso (FCC 09, model deployment): el
modelo deja de vivir en un notebook y pasa a responder peticiones reales.

Endpoints:
- GET  /api/health    -> estado del servicio y tamano de la galeria.
- POST /api/identify  -> recibe una foto (multipart) y devuelve el top-k de
                         cartas mas parecidas con sus distancias.
- GET  /cards/{id}.jpg -> imagen de una carta de la galeria (para mostrarla
                          en los resultados del front-end).
- GET  /              -> front-end (frontend/index.html).

Ejecucion:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

import io
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from backend.card_detector import CardDetector
from backend.catalog import CardCatalog
from backend.inference import CardIdentifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_DIR / "frontend"
IMAGES_DIR = PROJECT_DIR / "data" / "yugioh_card_images"

app = FastAPI(title="Reconocedor de cartas Yu-Gi-Oh!",
              description="Red siamesa + Triplet Loss + retrieval por embeddings")

# El modelo, la galeria y el catalogo se cargan UNA sola vez al arrancar el servidor
# (cargar ~45 MB de ResNet en cada peticion seria inviable).
identifier = CardIdentifier()
catalog = CardCatalog()
detector = CardDetector()


@app.get("/api/health")
def health() -> dict:
    """Chequeo rapido: confirma que el modelo y la galeria estan cargados."""
    return {
        "status": "ok",
        "device": str(identifier.device),
        "gallery_size": len(identifier.names),
        "catalog_size": len(catalog.cards),
    }


@app.get("/api/cards/search")
def search_cards(q: str = "", limit: int = 25) -> dict:
    """Buscador y filtro de cartas para autocompletado y asignacion de embeddings."""
    results = catalog.search(
        query=q,
        gallery_ids=set(identifier.ids),
        gallery_names=identifier.names,
        limit=limit,
    )
    return {"cards": results}


@app.post("/api/identify")
async def identify(file: UploadFile = File(...), top_k: int = 5) -> dict:
    """Identifica la carta de la foto subida.

    Pipeline completo (GuiaEstudio/06_YuGiOh_Triplet.md, "Idea central"):
        Imagen -> ResNet -> vector 128-D -> distancia -> nombre de carta
    """
    # Validacion en el borde del sistema: el archivo debe ser una imagen decodificable.
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="El archivo no es una imagen valida.")

    # Toda la logica de ML vive en inference.py; la API solo traduce HTTP <-> modelo.
    result = identifier.identify(image, top_k=top_k)

    # Se agrega la URL local de la imagen de cada candidata para que el
    # front-end pueda mostrarla junto al nombre y la distancia.
    for match in result["matches"]:
        match["image_url"] = f"/cards/{match['image_id']}.jpg"

    return result


@app.post("/api/identify-multi")
async def identify_multi(file: UploadFile = File(...)) -> dict:
    """Detecta multiples cartas en una imagen completa (mesa/tapete), recorta y

    rectifica la perspectiva de cada una, y devuelve UNICAMENTE las cartas
    que hayan sido reconocidas positivamente por el modelo (distancia <= umbral).
    """
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="El archivo no es una imagen valida.")

    # 1. Deteccion de cuadrilateros y dewarping por perspectiva
    detected_cards = detector.detect(image)

    # 2. Inferencia y filtrado estricto de coincidencias
    valid_matches = []
    for card in detected_cards:
        res = identifier.identify(card.crop, top_k=1)
        # Solo incluir si fue reconocida dentro del umbral calibrado
        if res.get("recognized") and res.get("matches"):
            best = res["matches"][0]
            dist = float(best["distance"])
            sim_pct = round(max(0.0, 1.0 - dist / 2.0) * 100.0, 1)

            valid_matches.append({
                "name": best["name"],
                "image_id": best["image_id"],
                "distance": dist,
                "similarity_pct": sim_pct,
                "image_url": f"/cards/{best['image_id']}.jpg",
                "box": {k: int(v) for k, v in card.box.items()},
                "points": [[int(coord) for coord in pt] for pt in card.points],
                "is_defense": bool(card.is_defense_mode),
            })

    return {
        "total_detected": len(detected_cards),
        "total_matches": len(valid_matches),
        "threshold": round(identifier.threshold, 4),
        "matches": valid_matches,
        "detected_cards": [
            {
                "box": {k: int(v) for k, v in c.box.items()},
                "points": [[int(coord) for coord in pt] for pt in c.points],
                "is_defense": bool(c.is_defense_mode),
            }
            for c in detected_cards
        ],
    }


@app.post("/api/gallery/average")
async def average_embedding(
    file: UploadFile = File(...),
    target_id: Optional[str] = Form(None),
    target_name: Optional[str] = Form(None),
    alpha: float = Form(0.5),
) -> dict:
    """Promedia la foto subida con el embedding de una carta existente."""
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="El archivo no es una imagen valida.")

    try:
        res = identifier.average_embedding(
            target_id=target_id,
            target_name=target_name,
            image=image,
            alpha=alpha,
        )
        res["image_url"] = f"/cards/{res['image_id']}.jpg"
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al promediar embedding: {e}")


@app.post("/api/gallery/add")
async def add_card(
    file: UploadFile = File(...),
    name: str = Form(...),
    id: Optional[str] = Form(None),
) -> dict:
    """Registra una nueva carta y su embedding en la galeria."""
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="El archivo no es una imagen valida.")

    try:
        res = identifier.add_card(name=name, image=image, image_id=id)
        catalog.register_new_card(cid=res["image_id"], name=res["card_name"])
        res["image_url"] = f"/cards/{res['image_id']}.jpg"
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al agregar nueva carta: {e}")


# /cards/{id}.jpg -> las imagenes descargadas por data/cards_downloader.py.
app.mount("/cards", StaticFiles(directory=IMAGES_DIR), name="cards")

TEST_IMAGES_DIR = PROJECT_DIR / "data" / "test_table_images"
TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/test-images", StaticFiles(directory=TEST_IMAGES_DIR), name="test_images")

# La raiz sirve el front-end completo (html=True hace que "/" devuelva index.html).
# Se monta al final para que no tape las rutas /api/* y /cards/*.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
