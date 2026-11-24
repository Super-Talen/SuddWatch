"""
SuddWatch - latency and cost benchmark of the live Azure endpoint (rubric 5).

Latency: N warm requests per endpoint; client-side round trip (includes the network from where you run it) and
server-side processing time (X-Process-Time-ms header, set by the API middleware). p50 / p95 / p99.
Cold start: run with --cold after the app has been idle for 10+ minutes (it scales to zero); the first request
then includes starting a container and loading the model.
Cost: Azure retail prices (public API, no login) for Container Apps and Container Registry in the deployment region,
applied to two scenarios: scale-to-zero (as deployed) and one always-on replica.

Usage: .venv\\Scripts\\python.exe -m src.benchmark_api [--url URL] [--n 50] [--cold]
Output: phase3_modeling/deployment/latency_cost.json
"""
import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "phase3_modeling/deployment"
URL = "https://suddwatch-api.wittypond-9ed3ebe4.southafricanorth.azurecontainerapps.io"
REGION = "southafricanorth"
VCPU, MEM_GIB = 0.5, 1.0                      # container size as deployed
FREE_VCPU_S, FREE_GIB_S, FREE_REQ = 180_000, 360_000, 2_000_000   # Container Apps monthly free grant (consumption)
MONTH_S = 30 * 24 * 3600


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, math.ceil(q / 100 * len(xs)) - 1)], 1)


def payload(n_rows):
    df = pd.read_csv(ROOT / "api/latest_features.csv").drop(columns=["adm2_name", "issue_date"])
    rows = []
    for _, r in df.head(n_rows).iterrows():
        d = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.to_dict().items()}
        d["month"] = int(d["month"])
        rows.append(d)
    return {"rows": rows}


def bench(client, method, path, n, body=None):
    client_ms, server_ms = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        r = client.request(method, path, json=body)
        client_ms.append(1000 * (time.perf_counter() - t0))
        r.raise_for_status()
        server_ms.append(float(r.headers.get("X-Process-Time-ms", "nan")))
    return {"endpoint": f"{method} {path}" + (f" ({len(body['rows'])} rows)" if body else ""), "n": n,
            "client_p50_ms": pct(client_ms, 50), "client_p95_ms": pct(client_ms, 95), "client_p99_ms": pct(client_ms, 99),
            "server_p50_ms": pct(server_ms, 50), "server_p95_ms": pct(server_ms, 95), "server_mean_ms": round(statistics.mean(server_ms), 1)}


def prices():
    """Pay-as-you-go retail prices (USD) from https://prices.azure.com (public, unauthenticated)."""
    def query(f):
        items, url = [], "https://prices.azure.com/api/retail/prices"
        params = {"$filter": f}
        while url:
            j = httpx.get(url, params=params, timeout=60).json()
            items += j.get("Items", []); url, params = j.get("NextPageLink"), None
        return items
    ca = query(f"serviceName eq 'Azure Container Apps' and armRegionName eq '{REGION}' and priceType eq 'Consumption'")
    acr = query(f"serviceName eq 'Container Registry' and armRegionName eq '{REGION}' and priceType eq 'Consumption'")

    def find(items, *words, unit=None):
        for it in items:
            name = (it["meterName"] + " " + it.get("skuName", "")).lower()
            if all(w.lower() in name for w in words) and it["retailPrice"] > 0 and (unit is None or unit in it["unitOfMeasure"]):
                return it
        return None
    return {"vcpu_active": find(ca, "standard", "vcpu", "active") or find(ca, "vcpu", "active"),
            "vcpu_idle": find(ca, "standard", "vcpu", "idle") or find(ca, "vcpu", "idle"),
            "mem_active": find(ca, "standard", "memory", "active") or find(ca, "memory", "active"),
            "mem_idle": find(ca, "standard", "memory", "idle") or find(ca, "memory", "idle"),
            "requests": find(ca, "request"),
            "acr_basic": find(acr, "basic", "registry", unit="Day") or find(acr, "basic", unit="Day")}


def per_unit(item):
    """Price per single unit (meters are quoted per 1, 100, 1M ... units)."""
    if not item:
        return None
    u = item["unitOfMeasure"].split()[0].replace(",", "")
    mult = {"1": 1, "10": 10, "100": 100, "1K": 1e3, "10K": 1e4, "1M": 1e6}.get(u, float(u) if u.replace(".", "").isdigit() else 1)
    return item["retailPrice"] / mult


def cost(p, active_s_per_month, requests_per_month, always_on=False):
    va, vi, ma, mi = (per_unit(p[k]) for k in ("vcpu_active", "vcpu_idle", "mem_active", "mem_idle"))
    req = per_unit(p["requests"]) or 0
    vcpu_active_s, gib_active_s = VCPU * active_s_per_month, MEM_GIB * active_s_per_month
    idle_s = (MONTH_S - active_s_per_month) if always_on else 0
    compute = max(vcpu_active_s - FREE_VCPU_S, 0) * va + max(gib_active_s - FREE_GIB_S, 0) * ma
    compute += VCPU * idle_s * (vi or va) + MEM_GIB * idle_s * (mi or ma)
    requests = max(requests_per_month - FREE_REQ, 0) * req
    registry = 30 * per_unit(p["acr_basic"]) if p["acr_basic"] else None
    return {"compute_usd": round(compute, 2), "requests_usd": round(requests, 2), "registry_usd": round(registry, 2) if registry else None,
            "total_usd_per_month": round(compute + requests + (registry or 0), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL); ap.add_argument("--n", type=int, default=50); ap.add_argument("--cold", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"url": a.url, "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "container": {"vcpu": VCPU, "memory_gib": MEM_GIB, "min_replicas": 0, "max_replicas": 1, "region": REGION}}
    with httpx.Client(base_url=a.url, timeout=120) as c:
        if a.cold:
            t0 = time.perf_counter(); r = c.get("/health")
            res["cold_start_first_request_s"] = round(time.perf_counter() - t0, 2)
            res["cold_start_model_load_ms"] = r.json().get("model_load_ms")
        c.get("/health")                                                  # make sure it is warm
        res["latency"] = [bench(c, "GET", "/health", a.n), bench(c, "GET", "/forecast/latest/SS0302", a.n),
                          bench(c, "GET", "/forecast/latest", a.n), bench(c, "POST", "/predict", a.n, payload(1)),
                          bench(c, "POST", "/predict", a.n, payload(79))]
    try:
        p = prices()
        res["prices_usd"] = {k: ({"meter": v["meterName"], "price": v["retailPrice"], "unit": v["unitOfMeasure"]} if v else None) for k, v in p.items()}
        res["cost_estimates_per_month"] = {
            "scale_to_zero_demo_use (40 visits/day, each keeps 1 replica up ~5 min)": cost(p, 40 * 300 * 30, 40 * 20 * 30),
            "scale_to_zero_heavy_use (8 h/day active, 100k requests/day)": cost(p, 8 * 3600 * 30, 100_000 * 30),
            "always_on_one_replica (min-replicas 1)": cost(p, 2 * 3600 * 30, 40 * 20 * 30, always_on=True)}
        res["cost_notes"] = ("Pay-as-you-go list prices; Azure for Students credit ($100) pays them. Free monthly grant applied: "
                             "180,000 vCPU-s, 360,000 GiB-s, 2M requests. Log Analytics ingestion (a few MB/month) is within "
                             "its free 5 GB. Check actual spend in Azure portal > Cost Management.")
    except Exception as e:                                                # pricing API unavailable: keep latency results
        res["cost_error"] = str(e)
    (OUT / "latency_cost.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
