"""Benchmark y comparacion empirica de metricas de distancia entre embeddings.

Responde directamente a la pregunta de la mesa de examen:
¿Por que se eligio la distancia Euclidiana L2 y no Coseno, L1 u otra?

Compara:
1. Distancia Euclidiana (L2) con normalizacion L2
2. Distancia Coseno (1 - cos(theta)) con normalizacion L2
3. Distancia Manhattan (L1) con normalizacion L2
4. Distancia Chebyshev (L-inf) con normalizacion L2
5. Distancia Euclidiana (L2) SIN normalizacion L2 (proyeccion cruda)
6. Distancia Manhattan (L1) SIN normalizacion L2
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "dev"))

from dataset import CardImageDataset
from model import EmbeddingNet
from transforms import inference_transform, train_transform

ARTIFACTS_DIR = PROJECT_DIR / "artifacts"


def load_model(device):
    checkpoint = torch.load(ARTIFACTS_DIR / "model.pt", map_location=device, weights_only=False)
    model = EmbeddingNet(embedding_dim=checkpoint["embedding_dim"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def compute_distances(query_emb: torch.Tensor, gallery_emb: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "euclidean_l2":
        return torch.cdist(query_emb, gallery_emb, p=2).squeeze(0)
    elif metric == "cosine":
        sim = F.cosine_similarity(query_emb, gallery_emb, dim=-1)
        return 1.0 - sim
    elif metric == "manhattan_l1":
        return torch.cdist(query_emb, gallery_emb, p=1).squeeze(0)
    elif metric == "chebyshev_linf":
        return torch.cdist(query_emb, gallery_emb, p=float("inf")).squeeze(0)
    else:
        raise ValueError(f"Metrica desconocida: {metric}")


def run_benchmark(sample_size=150):
    device = torch.device("cpu")
    print(f"--- Iniciando Benchmark Consistente en {device} (n={sample_size}) ---", flush=True)

    model = load_model(device)
    dataset_inf = CardImageDataset(transform=inference_transform)
    names = [dataset_inf.id_to_name[p.stem] for p in dataset_inf.files]
    n = len(names)
    k = min(sample_size, n)

    loader_inf = DataLoader(dataset_inf, batch_size=64, shuffle=False, num_workers=0)
    gallery_raw_list = []

    print(f"Extrayendo representaciones de las {n} cartas en lotes...", flush=True)
    with torch.no_grad():
        for batch_imgs, _, _ in loader_inf:
            raw_emb = model.backbone(batch_imgs.to(device))
            gallery_raw_list.append(raw_emb)
    gallery_raw = torch.cat(gallery_raw_list, dim=0)               # [N, 128] crudos
    gallery_l2 = F.normalize(gallery_raw, p=2, dim=-1)            # [N, 128] normalizados L2

    torch.manual_seed(42)
    indices = torch.randperm(n)[:k].tolist()
    dataset_base = CardImageDataset(transform=None)

    print(f"Generando {k} consultas con augmentacion realista...", flush=True)
    queries_l2 = []
    queries_raw = []
    ground_truth = []

    with torch.no_grad():
        for idx in indices:
            path = dataset_base.files[idx]
            img = Image.open(path).convert("RGB")
            t_img = train_transform(img).unsqueeze(0).to(device)
            raw_q = model.backbone(t_img)
            norm_q = F.normalize(raw_q, p=2, dim=-1)
            queries_l2.append(norm_q)
            queries_raw.append(raw_q)
            ground_truth.append((idx, names[idx]))

    metrics_to_test = [
        ("Euclidiana (L2) + Norm L2", "euclidean_l2", queries_l2, gallery_l2),
        ("Coseno (1 - cos) + Norm L2", "cosine", queries_l2, gallery_l2),
        ("Manhattan (L1) + Norm L2", "manhattan_l1", queries_l2, gallery_l2),
        ("Chebyshev (L-inf) + Norm L2", "chebyshev_linf", queries_l2, gallery_l2),
        ("Euclidiana (L2) SIN Norm L2 (Cruda)", "euclidean_l2", queries_raw, gallery_raw),
        ("Manhattan (L1) SIN Norm L2 (Cruda)", "manhattan_l1", queries_raw, gallery_raw),
    ]

    results = []

    for name_label, metric_code, q_list, gal in metrics_to_test:
        top1_count = 0
        top5_count = 0
        total_time_ms = 0.0
        genuine_dists = []
        impostor_dists = []

        for i, (true_idx, true_name) in enumerate(ground_truth):
            q = q_list[i]
            t0 = time.perf_counter()
            dists = compute_distances(q, gal, metric_code)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            total_time_ms += elapsed_ms

            sorted_indices = dists.argsort().tolist()
            pred_names = [names[idx] for idx in sorted_indices[:5]]

            if pred_names[0] == true_name:
                top1_count += 1
            if true_name in pred_names:
                top5_count += 1

            d_gen = dists[true_idx].item()
            # Distancia al impostor mas cercano
            mask_other = torch.tensor([n != true_name for n in names], dtype=torch.bool)
            d_imp = dists[mask_other].min().item()

            genuine_dists.append(d_gen)
            impostor_dists.append(d_imp)

        avg_gen = sum(genuine_dists) / len(genuine_dists)
        avg_imp = sum(impostor_dists) / len(impostor_dists)
        separation_ratio = (avg_imp - avg_gen) / (avg_imp + 1e-8)
        avg_latency = total_time_ms / k

        results.append({
            "Configuracion": name_label,
            "Top-1 (%)": round(100.0 * top1_count / k, 2),
            "Top-5 (%)": round(100.0 * top5_count / k, 2),
            "Separabilidad (Imp-Gen)/Imp": round(separation_ratio, 4),
            "Dist Media Genuina": round(avg_gen, 4),
            "Dist Media Impostora": round(avg_imp, 4),
            "Latencia (ms/query)": round(avg_latency, 3),
        })

    print("\n" + "=" * 90, flush=True)
    print("RESULTADOS EXPERIMENTALES: COMPARACIÓN DE MÉTRICAS DE DISTANCIA", flush=True)
    print("=" * 90, flush=True)
    header = f"{'Configuracion':<38} | {'Top-1':<7} | {'Top-5':<7} | {'Separacion':<10} | {'Latencia':<8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in results:
        print(f"{r['Configuracion']:<38} | {r['Top-1 (%)']:<5}% | {r['Top-5 (%)']:<5}% | {r['Separabilidad (Imp-Gen)/Imp']:<10} | {r['Latencia (ms/query)']:<5} ms", flush=True)
    print("=" * 90, flush=True)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = ARTIFACTS_DIR / "distance_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en: {out_json}", flush=True)


if __name__ == "__main__":
    run_benchmark(sample_size=150)
