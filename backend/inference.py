"""Motor de inferencia: identifica una carta buscando su vecino mas cercano.

Encapsula lo que en el prototipo Colab era la funcion ``identify()``:
    foto -> transform -> EmbeddingNet -> vector 128-D -> torch.cdist contra la
    galeria -> top-k cartas mas cercanas.

Mejoras:
- Transform DETERMINISTA de inferencia (compartido con build_gallery).
- Top-k + umbral de rechazo CALIBRADO (guardado en gallery.pt por
  build_gallery.py). La constante de abajo es solo fallback si el archivo
  es antiguo y no trae el campo.

El retrieval por vecino mas cercano se explica en
GuiaEstudio/06_YuGiOh_Triplet.md (concepto 6, "Nearest neighbor:
torch.cdist -> argmin").
"""

import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
# Se agrega dev/ al path para reutilizar model.py y transforms.py sin duplicar codigo.
sys.path.insert(0, str(PROJECT_DIR / "dev"))

from model import EmbeddingNet          # noqa: E402
from transforms import inference_transform  # noqa: E402

ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
IMAGES_DIR = PROJECT_DIR / "data" / "yugioh_card_images"

# Fallback si gallery.pt no trae distance_threshold (galeria vieja).
# El valor real se calibra en build_gallery.py (U1/9 Seleccion de Modelos).
FALLBACK_THRESHOLD = 0.75


class CardIdentifier:
    """Carga modelo + galeria UNA sola vez y responde consultas."""

    def __init__(self):
        # En un servidor web la inferencia suele correr en CPU; si hay GPU se usa.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()

        # --- Modelo entrenado (dev/train.py) ---
        checkpoint = torch.load(ARTIFACTS_DIR / "model.pt", map_location=self.device,
                                weights_only=False)
        # pretrained=False: load_state_dict pisa los pesos con los entrenados.
        self.model = EmbeddingNet(embedding_dim=checkpoint["embedding_dim"],
                                  pretrained=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        # eval(): BatchNorm en modo inferencia (U2/2 Redes_Convolucionales.ipynb).
        self.model.eval()

        # --- Galeria de referencia (dev/build_gallery.py) ---
        gallery = torch.load(ARTIFACTS_DIR / "gallery.pt", map_location="cpu",
                             weights_only=False)
        self.embeddings = gallery["embeddings"].to(self.device)  # [N, 128]
        self.ids = [str(i) for i in gallery["ids"]]
        self.names = list(gallery["names"])
        # Umbral calibrado; fallback si el .pt es de una version anterior.
        self.threshold = float(gallery.get("distance_threshold", FALLBACK_THRESHOLD))

    def compute_embedding(self, image: Image.Image) -> torch.Tensor:
        """Calcula el vector de embedding 128-D L2-normalizado para una imagen."""
        query = inference_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_emb = self.model(query)  # [1, 128] (ya normalizado en EmbeddingNet)
        return query_emb

    def identify(self, image: Image.Image, top_k: int = 5) -> dict:
        """Devuelve las top_k cartas mas parecidas a la imagen consultada."""
        query_emb = self.compute_embedding(image)

        # torch.cdist: matriz de distancias L2 entre la consulta y las N cartas
        # de la galeria -> [1, N] -> squeeze -> [N].
        with self.lock:
            embeddings_snap = self.embeddings
            names_snap = self.names
            ids_snap = self.ids

        dists = torch.cdist(query_emb, embeddings_snap).squeeze(0)

        # topk con largest=False = los k vecinos MAS CERCANOS (menor distancia).
        k = min(top_k, len(names_snap))
        top_dists, top_idx = torch.topk(dists, k=k, largest=False)

        matches = [
            {
                "name": names_snap[i],
                "image_id": ids_snap[i],
                # La distancia se informa al usuario como medida de confianza:
                # ~0 = practicamente identica, cerca de 2 = nada que ver.
                "distance": round(d.item(), 4),
            }
            for d, i in zip(top_dists, top_idx)
        ]

        # Umbral calibrado: si ni el mejor candidato esta cerca -> no reconocida.
        recognized = matches[0]["distance"] <= self.threshold if matches else False

        return {
            "recognized": recognized,
            "threshold": round(self.threshold, 4),
            "matches": matches,
        }

    def average_embedding(self, target_id: str | None = None, target_name: str | None = None,
                          image: Image.Image = None, alpha: float = 0.5) -> dict:
        """Promedia el embedding de una carta existente con una nueva foto real.

        Matematica:
        v = alpha * e_viejo + (1 - alpha) * e_nuevo
        e_final = v / ||v||_2  (Renormalizacion obligatoria a la hiperesfera S^127)
        """
        if image is None:
            raise ValueError("Se requiere una imagen para promediar.")

        target_str_id = str(target_id).strip() if target_id else ""
        target_name_clean = target_name.strip().lower() if target_name else ""

        target_idx = -1
        # 1. Buscar por target_id
        if target_str_id:
            for idx, cid in enumerate(self.ids):
                if cid == target_str_id:
                    target_idx = idx
                    break

        # 2. Si no encontro por id, buscar por nombre
        if target_idx == -1 and target_name_clean:
            for idx, name in enumerate(self.names):
                if name.strip().lower() == target_name_clean:
                    target_idx = idx
                    break

        if target_idx == -1:
            raise ValueError(f"No se encontro la carta objetivo (id={target_id}, name={target_name}) en la galeria.")

        new_emb = self.compute_embedding(image)  # [1, 128]

        with self.lock:
            old_emb = self.embeddings[target_idx:target_idx + 1]  # [1, 128]
            # Promedio ponderado en el espacio métrico
            blended = alpha * old_emb + (1.0 - alpha) * new_emb
            # Renormalización L2
            norm_emb = F.normalize(blended, p=2, dim=-1)  # [1, 128]
            self.embeddings[target_idx] = norm_emb.squeeze(0)
            self._save_gallery()

            card_name = self.names[target_idx]
            card_id = self.ids[target_idx]
            gallery_len = len(self.names)

        # Distancia entre la nueva foto y el embedding promediado actualizado
        updated_dist = torch.cdist(new_emb, norm_emb).item()

        return {
            "success": True,
            "action": "averaged",
            "card_name": card_name,
            "image_id": card_id,
            "new_distance": round(updated_dist, 4),
            "threshold": round(self.threshold, 4),
            "gallery_size": gallery_len,
        }

    def add_card(self, name: str, image: Image.Image, image_id: str | None = None) -> dict:
        """Agrega una nueva carta a la galeria con su embedding L2-normalizado."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("El nombre de la carta no puede estar vacio.")

        cid = str(image_id).strip() if image_id else str(int(time.time() * 1000))

        new_emb = self.compute_embedding(image)  # [1, 128]

        with self.lock:
            self.embeddings = torch.cat([self.embeddings, new_emb], dim=0)
            self.ids.append(cid)
            self.names.append(clean_name)
            self._save_gallery()
            gallery_len = len(self.names)

        # Guardar imagen en data/yugioh_card_images si no existe para que se pueda servir via /cards/{id}.jpg
        try:
            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            img_path = IMAGES_DIR / f"{cid}.jpg"
            image.convert("RGB").save(img_path, format="JPEG", quality=92)
        except Exception as e:
            print(f"Advertencia: no se pudo guardar imagen de nueva carta: {e}")

        return {
            "success": True,
            "action": "added",
            "card_name": clean_name,
            "image_id": cid,
            "gallery_size": gallery_len,
        }

    def _save_gallery(self) -> None:
        """Persistencia atomica de la galeria a artifacts/gallery.pt."""
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = ARTIFACTS_DIR / "gallery.pt.tmp"
        out_path = ARTIFACTS_DIR / "gallery.pt"

        torch.save({
            "embeddings": self.embeddings.detach().cpu(),
            "ids": list(self.ids),
            "names": list(self.names),
            "distance_threshold": self.threshold,
        }, tmp_path)
        tmp_path.replace(out_path)

