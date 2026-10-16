"""Servicio de catalogo de cartas.

Carga y busca cartas en `data/cards.json` y las sincroniza con la galeria
activa de embeddings para responder a busquedas y filtros en tiempo real.
"""

import json
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
CARDS_JSON_PATH = PROJECT_DIR / "data" / "cards.json"
IMAGES_DIR = PROJECT_DIR / "data" / "yugioh_card_images"


class CardCatalog:
    """Gestiona el catalogo completo de cartas para autocompletado y busqueda."""

    def __init__(self):
        self.cards: list[dict[str, Any]] = []
        self.by_id: dict[str, dict[str, Any]] = {}
        self.load_catalog()

    def load_catalog(self) -> None:
        """Carga el dataset de cartas desde data/cards.json si existe."""
        if not CARDS_JSON_PATH.exists():
            self.cards = []
            return

        try:
            with open(CARDS_JSON_PATH, "r", encoding="utf-8") as f:
                raw_cards = json.load(f)

            self.cards = []
            self.by_id = {}
            for item in raw_cards:
                cid = str(item.get("id"))
                name = item.get("name", "")
                card_type = item.get("type", "Card")
                
                # Imagen local o fallback a URL remota de YGOPRODeck
                local_img = IMAGES_DIR / f"{cid}.jpg"
                if local_img.exists():
                    img_url = f"/cards/{cid}.jpg"
                elif item.get("card_images") and len(item["card_images"]) > 0:
                    img_url = item["card_images"][0].get("image_url_small", f"/cards/{cid}.jpg")
                else:
                    img_url = f"/cards/{cid}.jpg"

                entry = {
                    "id": cid,
                    "name": name,
                    "type": card_type,
                    "image_url": img_url,
                }
                self.cards.append(entry)
                self.by_id[cid] = entry
        except Exception as e:
            print(f"Advertencia: No se pudo cargar cards.json: {e}")
            self.cards = []

    def search(self, query: str, gallery_ids: set[str], gallery_names: list[str], limit: int = 25) -> list[dict[str, Any]]:
        """Busca cartas por nombre o id, marcando si estan en la galeria activa.

        Prioriza:
        1. Coincidencias exactas de prefijo.
        2. Coincidencias de subcadena.
        3. Cartas que ya estan en la galeria activa.
        """
        q = query.strip().lower()
        results: list[dict[str, Any]] = []

        gallery_names_set = {n.lower() for n in gallery_names}

        # 1. Incluir cartas agregadas dinamicamente a la galeria que quizas no esten en cards.json
        custom_seen = set()
        for idx, gid in enumerate(gallery_ids):
            if gid not in self.by_id:
                name = gallery_names[idx] if idx < len(gallery_names) else f"Carta {gid}"
                if not q or q in name.lower() or q in gid:
                    custom_seen.add(gid)
                    results.append({
                        "id": gid,
                        "name": name,
                        "type": "Custom / En Galería",
                        "in_gallery": True,
                        "image_url": f"/cards/{gid}.jpg",
                    })

        # 2. Filtrar sobre el catalogo de cards.json
        matches_prefix: list[dict[str, Any]] = []
        matches_substr: list[dict[str, Any]] = []

        for card in self.cards:
            cid = card["id"]
            if cid in custom_seen:
                continue
            name_lower = card["name"].lower()

            in_gallery = cid in gallery_ids or name_lower in gallery_names_set

            if not q:
                # Si no hay query, mostrar primeras cartas (priorizando las que estan en galeria)
                matches_substr.append({
                    **card,
                    "in_gallery": in_gallery,
                })
            elif name_lower.startswith(q) or cid.startswith(q):
                matches_prefix.append({
                    **card,
                    "in_gallery": in_gallery,
                })
            elif q in name_lower or q in cid:
                matches_substr.append({
                    **card,
                    "in_gallery": in_gallery,
                })

        # Ordenar: primero prefijo, luego subcadena; dentro de cada uno, preferir in_gallery==True
        def sort_key(item: dict[str, Any]):
            return (not item.get("in_gallery", False), item["name"])

        matches_prefix.sort(key=sort_key)
        matches_substr.sort(key=sort_key)

        combined = results + matches_prefix + matches_substr
        return combined[:limit]

    def register_new_card(self, cid: str, name: str, card_type: str = "Carta de Usuario") -> None:
        """Registra una nueva carta en la lista en memoria si no existia."""
        if cid not in self.by_id:
            entry = {
                "id": cid,
                "name": name,
                "type": card_type,
                "image_url": f"/cards/{cid}.jpg",
            }
            self.cards.insert(0, entry)
            self.by_id[cid] = entry
