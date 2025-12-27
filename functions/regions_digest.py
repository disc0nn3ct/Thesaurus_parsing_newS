# functions/regions_digest.py
# Требуемые зависимости:
#   pip install plotly kaleido geopandas shapely pyproj numpy pyarrow
#
# Файлы, которые нужно положить в:
#   src/wordstat/geo/
#     - russia_regions.geojson           (из статьи)
#     - (опционально) russia_regions.parquet  (будет создан автоматически при первом запуске)
#
# ВАЖНО: если у тебя geo-файлы лежат в functions/src/wordstat/geo/,
# то либо перенеси их в src/wordstat/geo/, либо измени WORDSTAT_SAVE_DIR ниже.

import os, re, json, time, hashlib
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

# 1) поправь этот импорт на реальный файл, где лежит _wordstat_post
from functions.wordstat_api import _wordstat_post

try:
    from loguru import logger
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# --- PATHS ---

# КУДА СОХРАНЯЕМ РЕЗУЛЬТАТЫ (картинки, raw и т.п.) — КАК РАНЬШЕ
WORDSTAT_SAVE_DIR = os.path.join("src", "wordstat")
WORDSTAT_MAPS_DIR = os.path.join(WORDSTAT_SAVE_DIR, "maps")

# ОТКУДА БЕРЁМ ГЕО-ДАННЫЕ (только чтение)
WORDSTAT_GEO_DIR = os.path.join("functions", "src", "wordstat", "geo")

# создаём директории при необходимости
os.makedirs(WORDSTAT_MAPS_DIR, exist_ok=True)
os.makedirs(WORDSTAT_GEO_DIR, exist_ok=True)

# --- GEO FILES ---

REGIONS_GEOJSON = os.path.join(WORDSTAT_GEO_DIR, "russia_regions.geojson")
REGIONS_PARQUET = os.path.join(WORDSTAT_GEO_DIR, "russia_regions.parquet")

WORDSTAT_TREE_PATH = os.path.join(WORDSTAT_GEO_DIR, "wordstat_regions_tree.json")
REGIONID_TO_MAPREGION_PATH = os.path.join(WORDSTAT_GEO_DIR, "regionid_to_mapregion.json")
POPULATION_PATH = os.path.join(WORDSTAT_GEO_DIR, "population_by_regionid.json")


# ----------------------------
# helpers
# ----------------------------
def slugify_phrase(s: str, max_len: int = 60) -> str:
    s = (s or "").strip().lower()
    base = re.sub(r"[^\w.-]+", "_", s, flags=re.UNICODE).strip("_")
    if not base:
        base = "phrase"
    if len(base) > max_len:
        base = base[:max_len].rstrip("_")
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{h}"


def _daily_bounds_last_60_days():
    """
    period=daily: доступны данные только за последние 60 дней.
    toDate по умолчанию = вчера, но зададим явно.
    """
    today = datetime.today().date()
    to_date = today - timedelta(days=1)          # вчера
    from_date = to_date - timedelta(days=59)     # 60 дней включая to_date
    return from_date, to_date


def _normalize_region_name(s: str) -> str:
    """
    Нормализация названий для сопоставления:
    - lower
    - ё->е
    - убираем кавычки/скобки
    - схлопываем пробелы
    """
    s = (s or "").strip().lower()
    s = s.replace("ё", "е")
    s = re.sub(r"[\"'«»()]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ----------------------------
# Geo: geojson -> (x,y) shapes
# ----------------------------
def geom2shape(g):
    """Polygon/MultiPolygon -> x[], y[] с None-разделителями (как в статье)."""
    if isinstance(g, MultiPolygon):
        x = np.array([], dtype=object)
        y = np.array([], dtype=object)
        for poly in g.geoms:
            xx, yy = poly.exterior.coords.xy
            x = np.append(x, np.array(xx, dtype=object))
            y = np.append(y, np.array(yy, dtype=object))
            x = np.append(x, None)
            y = np.append(y, None)
        return pd.Series([x[:-1], y[:-1]])
    if isinstance(g, Polygon):
        xx, yy = g.exterior.coords.xy
        return pd.Series([np.array(xx, dtype=object), np.array(yy, dtype=object)])
    return pd.Series([np.array([], dtype=object), np.array([], dtype=object)])


def load_regions_shapes(simplify_tol: int = 500, target_crs: str = "EPSG:32646") -> pd.DataFrame:
    """
    Возвращает DataFrame с колонками: region, population, x, y
    Логика:
      1) если есть russia_regions.parquet (подготовленный) — читаем его
      2) иначе читаем russia_regions.geojson, делаем to_crs + simplify + geom2shape,
         и сохраняем parquet как кэш.
    """
    if os.path.exists(REGIONS_PARQUET):
        shapes = pd.read_parquet(REGIONS_PARQUET)
        # если старый parquet без центров — пересоберём из geojson
        if ("cx" not in shapes.columns) or ("cy" not in shapes.columns):
            logger.info("REGIONS_PARQUET без cx/cy — пересобираю из geojson...")
            os.remove(REGIONS_PARQUET)
            return load_regions_shapes(simplify_tol=simplify_tol, target_crs=target_crs)

        need = {"region", "population", "x", "y", "cx", "cy"}
        miss = need - set(shapes.columns)
        if miss:
            raise ValueError(f"В {REGIONS_PARQUET} нет колонок: {miss}")
        return shapes[["region", "population", "x", "y", "cx", "cy"]].copy()

    if not os.path.exists(REGIONS_GEOJSON):
        raise FileNotFoundError(
            f"Нет {REGIONS_GEOJSON}. Положи файл russia_regions.geojson в {WORDSTAT_GEO_DIR}"
        )

    gdf = gpd.read_file(REGIONS_GEOJSON)
    need = {"region", "population", "geometry"}
    miss = need - set(gdf.columns)
    if miss:
        raise ValueError(f"В {REGIONS_GEOJSON} нет колонок {miss}. Есть: {list(gdf.columns)}")

    # geojson обычно в EPSG:4326
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs(target_crs)

    # центры для подписей (берём representative_point чтобы точка гарантированно была внутри полигона)
    rp = gdf.geometry.representative_point()
    gdf["cx"] = rp.x
    gdf["cy"] = rp.y

    # упрощаем геометрию чтобы plotly не лагал
    gdf["geometry"] = gdf["geometry"].simplify(simplify_tol)

    # делаем x/y
    gdf[["x", "y"]] = gdf["geometry"].apply(geom2shape)

    shapes = gdf[["region", "population", "x", "y", "cx", "cy"]].copy()


    # сохраняем подготовленный parquet (ускоряет последующие запуски)
    try:
        shapes.to_parquet(REGIONS_PARQUET, index=False)
        logger.info(f"💾 Сохранён подготовленный слой регионов: {REGIONS_PARQUET}")
    except Exception as e:
        logger.warning(f"Не смог сохранить {REGIONS_PARQUET}: {e}")

    return shapes


# ----------------------------
# Wordstat regions tree -> mapping regionId -> region_name
# ----------------------------
def fetch_wordstat_regions_tree(force_refresh: bool = False) -> dict:
    """
    /v1/getRegionsTree: дерево регионов с regionId и именами.
    Кэш: WORDSTAT_TREE_PATH
    """
    if (not force_refresh) and os.path.exists(WORDSTAT_TREE_PATH):
        try:
            with open(WORDSTAT_TREE_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
            # если в кэше вдруг лежит ошибка — игнорим кэш
            if isinstance(cached, dict) and ("error" in cached or "errors" in cached):
                logger.warning("Wordstat regions tree cache contains error; refetching with force_refresh=True")
            else:
                return cached
        except Exception as e:
            logger.warning(f"Не смог прочитать кэш дерева регионов: {e}. Перезапрошу.")

    data = _wordstat_post("/v1/getRegionsTree", {})  # без параметров

    try:
        logger.info(f"Wordstat getRegionsTree: type={type(data)}, top_keys={list(data.keys())[:20] if isinstance(data, dict) else 'n/a'}")
    except Exception:
        pass

    # если сервис вернул ошибку — логируем и всё равно сохраняем (чтобы видеть что пришло)
    if isinstance(data, dict) and ("error" in data or "errors" in data):
        logger.error(f"Wordstat getRegionsTree returned error: {data}")

    try:
        with open(WORDSTAT_TREE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Сохранено дерево регионов Wordstat: {WORDSTAT_TREE_PATH}")
    except Exception as e:
        logger.exception(f"Не смог сохранить дерево регионов: {e}")

    return data


def flatten_wordstat_regions(tree_json) -> pd.DataFrame:
    """
    Парсит реальный формат Wordstat:
    - regionId  <- value
    - name      <- label
    """
    rows = []

    def walk(node, parent_id=None, level=0):
        if isinstance(node, dict):
            rid = node.get("value")
            name = node.get("label")

            if rid is not None and name:
                try:
                    rid_int = int(rid)
                except Exception:
                    rid_int = None

                if rid_int is not None:
                    rows.append({
                        "regionId": rid_int,
                        "name": str(name),
                        "parentId": int(parent_id) if parent_id is not None else None,
                        "level": int(level),
                    })
                    parent_for_children = rid_int
                    next_level = level + 1
                else:
                    parent_for_children = parent_id
                    next_level = level
            else:
                parent_for_children = parent_id
                next_level = level

            children = node.get("children")
            if isinstance(children, list):
                for c in children:
                    walk(c, parent_for_children, next_level)

        elif isinstance(node, list):
            for item in node:
                walk(item, parent_id, level)

    walk(tree_json)

    df = pd.DataFrame(rows).drop_duplicates(subset=["regionId"]).reset_index(drop=True)

    if df.empty:
        logger.error("flatten_wordstat_regions: дерево распарсилось в пустоту")
        return pd.DataFrame(columns=["regionId", "name", "parentId", "level", "name_norm"])

    df["name_norm"] = df["name"].map(_normalize_region_name)
    return df

def get_regionid_to_name(force_refresh_tree: bool = False) -> dict[int, str]:
    """
    Возвращает mapping regionId -> name (как в Wordstat)
    Например: 225 -> "Россия"
    """
    tree = fetch_wordstat_regions_tree(force_refresh=force_refresh_tree)
    wdf = flatten_wordstat_regions(tree)
    if wdf.empty:
        logger.warning("get_regionid_to_name: не смог распарсить дерево, подписи будут только regionId")
        return {}
    return {int(r["regionId"]): str(r["name"]) for _, r in wdf.iterrows()}



def build_regionid_to_mapregion(force_refresh_tree: bool = False) -> dict[int, str]:
    """
    Строим mapping regionId -> region (как в geojson/parquet shapes)
    1) грузим shapes (parquet если есть, иначе geojson -> parquet)
    2) берём дерево Wordstat
    3) мачим по нормализованным именам
    4) сохраняем JSON в src/wordstat/geo/regionid_to_mapregion.json
    """
    if os.path.exists(REGIONID_TO_MAPREGION_PATH) and not force_refresh_tree:
        with open(REGIONID_TO_MAPREGION_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): str(v) for k, v in raw.items()}

    shapes = load_regions_shapes()
    shapes["region_norm"] = shapes["region"].map(_normalize_region_name)

    tree = fetch_wordstat_regions_tree(force_refresh=force_refresh_tree)
    wdf = flatten_wordstat_regions(tree)
    if wdf.empty:
        raise RuntimeError("Не удалось распарсить дерево регионов Wordstat.")

    merged = wdf.merge(
        shapes[["region", "region_norm"]],
        left_on="name_norm",
        right_on="region_norm",
        how="left"
    )

    mapping: dict[int, str] = {}
    for _, r in merged.dropna(subset=["region"]).iterrows():
        mapping[int(r["regionId"])] = str(r["region"])

    # если будут несовпадения — сюда можно добавить ручные фиксы:
    # mapping[123] = "Москва"

    with open(REGIONID_TO_MAPREGION_PATH, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in mapping.items()}, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 Сохранён mapping regionId->mapregion: {REGIONID_TO_MAPREGION_PATH} ({len(mapping)} записей)")
    return mapping


# ----------------------------
# Map plot: choropleth per capita
# ----------------------------
def plot_russia_choropleth_per_capita(
    df_daily: pd.DataFrame,
    phrase: str,
    out_png: str,
    window_days: int = 7,
    top_n_regions: int | None = None,
):
    """
    Карта РФ по регионам:
    value = (сумма запросов за window_days) / население * 100000
    df_daily: date, count, regionId, phrase
    """
    shapes = load_regions_shapes()
    rid2region = build_regionid_to_mapregion(force_refresh_tree=True)


    sdf = df_daily[df_daily["phrase"] == phrase].copy()
    if sdf.empty:
        logger.warning(f"plot_russia_choropleth_per_capita: пусто для '{phrase}'")
        return None

    sdf["date"] = pd.to_datetime(sdf["date"])
    last_day = sdf["date"].max()
    start = last_day - pd.Timedelta(days=window_days - 1)

    sdf = sdf[(sdf["date"] >= start) & (sdf["date"] <= last_day)]
    agg = sdf.groupby("regionId", as_index=False)["count"].sum().rename(columns={"count": "count_sum"})

    if top_n_regions is not None:
        top_ids = agg.sort_values("count_sum", ascending=False)["regionId"].head(top_n_regions).tolist()
        agg = agg[agg["regionId"].isin(top_ids)]

    agg["region"] = agg["regionId"].map(rid2region)
    agg = agg.dropna(subset=["region"])

    m = shapes.merge(agg, on="region", how="left")
    m["count_sum"] = m["count_sum"].fillna(0)

    m["value"] = m.apply(
        lambda r: (float(r["count_sum"]) / float(r["population"]) * 100000.0) if r["population"] else 0.0,
        axis=1
    )

    vmin, vmax = float(m["value"].min()), float(m["value"].max())
    denom = (vmax - vmin) if (vmax - vmin) > 1e-9 else 1.0

    colors = [
        (0.0,  "rgb(247,251,255)"),
        (0.25, "rgb(200,221,240)"),
        (0.5,  "rgb(115,179,216)"),
        (0.75, "rgb(49,130,189)"),
        (1.0,  "rgb(8,81,156)"),
    ]

    def pick_color(x01: float) -> str:
        xs = [c[0] for c in colors]
        cs = [c[1] for c in colors]
        i = min(range(len(xs)), key=lambda i: abs(xs[i] - x01))
        return cs[i]

    fig = go.Figure()

    for _, r in m.iterrows():
        val = float(r["value"])
        x01 = (val - vmin) / denom
        fill = pick_color(x01)

        hover = (
            f"<b>{r['region']}</b><br>"
            f"Сумма {window_days}д: {int(round(r['count_sum']))}<br>"
            f"Население: {int(r['population']):,}".replace(",", " ") + "<br>"
            f"На 100k: {val:.2f}"
        )

        fig.add_trace(go.Scatter(
            x=r["x"], y=r["y"],
            name=r["region"],
            text=hover,
            hoverinfo="text",
            mode="lines",
            line=dict(color="grey", width=0.6),
            fill="toself",
            fillcolor=fill,
            showlegend=False,
        ))


        # --- подписи регионов ---
    # чтобы не было каши: подписи только для регионов с ненулевым значением (или топ-N)
    labels_df = m.copy()

    if top_n_regions is not None:
        labels_df = labels_df.sort_values("value", ascending=False).head(top_n_regions)
    else:
        labels_df = labels_df[labels_df["value"] > 0]

    # если всё равно много — ограничим (иначе мелкий текст будет мешать)
    if len(labels_df) > 35:
        labels_df = labels_df.sort_values("value", ascending=False).head(35)

    # короткие названия (можешь расширить правила)
    def short_name(name: str) -> str:
        n = str(name)
        n = n.replace("область", "обл.").replace("Республика", "Респ.")
        n = n.replace("автономный округ", "АО").replace("край", "кр.")
        return n

    fig.add_trace(go.Scatter(
        x=labels_df["cx"],
        y=labels_df["cy"],
        text=labels_df["region"].map(short_name),
        mode="text",
        textposition="middle center",
        hoverinfo="skip",
        showlegend=False,
        textfont=dict(size=10),
    ))


    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=f"Wordstat: '{phrase}' — на 100k жителей (сумма {window_days} дней)",
        margin=dict(l=10, r=10, t=40, b=10),
        width=1100,
        height=650,
    )
    import plotly.io as pio
    # для snap chromium обычно так:
    CHROMIUM = "/snap/bin/chromium"
    if os.path.exists(CHROMIUM):
        try:
            # у разных версий plotly/kaleido имя поля может отличаться
            pio.kaleido.scope.chromium_executable = CHROMIUM
        except Exception:
            pass
        try:
            pio.kaleido.scope.executable = CHROMIUM
        except Exception:
            pass
        
    print([x for x in dir(pio.kaleido.scope) if "exec" in x.lower() or "chrome" in x.lower()])

    # НУЖЕН kaleido: pip install kaleido
    try:
        fig.write_image(out_png, scale=2)
        return out_png
    except Exception as e:
        logger.exception(f"PNG export failed, fallback to HTML: {e}")
        out_html = out_png.replace(".png", ".html")
        fig.write_html(out_html, include_plotlyjs="embed")
        return out_html


# ----------------------------
# Existing plots: total + heatmap
# ----------------------------
def plot_daily_total_compare(df: pd.DataFrame, phrases: list[str], out_path: str):
    """
    Суммарная дневная динамика по выбранным регионам (сумма по regionId) для каждой фразы.
    """
    sdf = df[df["phrase"].isin(phrases)].copy()
    if sdf.empty:
        logger.warning("plot_daily_total_compare: пустой df")
        return None

    g = (sdf.groupby(["date", "phrase"])["count"].sum()
         .reset_index()
         .sort_values("date"))

    plt.figure(figsize=(12, 5))
    for ph in phrases:
        gg = g[g["phrase"] == ph]
        plt.plot(gg["date"], gg["count"], linewidth=2, label=ph)

    plt.title("Wordstat daily: суммарная динамика по топ-регионам")
    plt.xlabel("Дата")
    plt.ylabel("Число запросов (sum count)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info(f"💾 daily total compare сохранён: {out_path}")
    return out_path


def plot_daily_heatmap(df: pd.DataFrame, phrase: str, out_path: str, top_regions: int = 25):
    """
    Heatmap: строки=regionId, столбцы=даты, значение=count.
    Подписи по оси Y: "Название (regionId)".
    """
    sdf = df[df["phrase"] == phrase].copy()
    if sdf.empty:
        logger.warning(f"plot_daily_heatmap: пусто для '{phrase}'")
        return None

    order = (sdf.groupby("regionId")["count"].sum()
             .sort_values(ascending=False)
             .head(top_regions)
             .index.tolist())
    sdf = sdf[sdf["regionId"].isin(order)]

    pivot = (sdf.pivot_table(index="regionId", columns="date", values="count", aggfunc="sum")
               .fillna(0))

    # гарантируем нормальные даты в колонках
    pivot.columns = pd.to_datetime(pivot.columns)

    plt.figure(figsize=(14, max(6, 0.28 * len(pivot.index))))
    plt.imshow(pivot.values, aspect="auto")
    plt.title(f"Wordstat daily heatmap (top {top_regions} regions): {phrase}")
    plt.xlabel("Дата")
    plt.ylabel("Регион (regionId)")

    cols = list(pivot.columns)
    step = max(1, len(cols) // 10)
    xticks = list(range(0, len(cols), step))
    plt.xticks(xticks, [cols[i].strftime("%m-%d") for i in xticks], rotation=45)

    # ✅ подписи регионов
    rid2name = get_regionid_to_name(force_refresh_tree=False)

    def label_rid(rid: int) -> str:
        name = rid2name.get(int(rid))
        return f"{name} ({int(rid)})" if name else str(int(rid))

    plt.yticks(range(len(pivot.index)), [label_rid(rid) for rid in pivot.index])

    plt.colorbar(label="count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    logger.info(f"💾 heatmap сохранён: {out_path}")
    return out_path


# ----------------------------
# Wordstat API fetchers
# ----------------------------
def fetch_wordstat_regions_distribution(phrase: str, region_type: str = "regions", devices=None) -> pd.DataFrame:
    """
    /v1/regions: распределение по регионам за последние 30 дней.
    Возвращает: regionId, count, share, affinityIndex
    region_type: 'cities' | 'regions' | 'all'
    """
    if devices is None:
        devices = ["all"]

    payload = {
        "phrase": phrase,
        "regionType": region_type,
        "devices": devices,
    }
    data = _wordstat_post("/v1/regions", payload)
    rows = data.get("regions", []) or []
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["regionId", "count", "share", "affinityIndex"])
    df["regionId"] = df["regionId"].astype(int)
    return df.sort_values("count", ascending=False).reset_index(drop=True)


def fetch_wordstat_daily_dynamics(phrase: str, region_id: int, devices=None) -> pd.DataFrame:
    """
    /v1/dynamics daily по конкретному региону (последние 60 дней).
    """
    if devices is None:
        devices = ["all"]

    from_date, to_date = _daily_bounds_last_60_days()

    payload = {
        "phrase": phrase,
        "period": "daily",
        "fromDate": from_date.strftime("%Y-%m-%d"),
        "toDate": to_date.strftime("%Y-%m-%d"),
        "regions": [int(region_id)],
        "devices": devices,
    }
    data = _wordstat_post("/v1/dynamics", payload)
    dyn = data.get("dynamics", []) or []
    df = pd.DataFrame(dyn)
    if df.empty:
        df = pd.DataFrame(columns=["date", "count", "share"])
    else:
        df["date"] = pd.to_datetime(df["date"])
        df["count"] = df["count"].astype(float)
        df["share"] = df["share"].astype(float)
        df = df.sort_values("date").reset_index(drop=True)

    df["regionId"] = int(region_id)
    df["phrase"] = phrase
    return df


def build_daily_region_dataset(
    phrases: list[str],
    top_n_regions: int = 25,
    region_type: str = "regions",
    devices=None,
    sleep_s: float = 0.25,
) -> pd.DataFrame:
    """
    1) /v1/regions -> берём топ-N регионов за 30 дней
    2) для каждого региона: /v1/dynamics daily (60 дней)
    """
    if devices is None:
        devices = ["all"]

    frames = []
    for phrase in phrases:
        dist = fetch_wordstat_regions_distribution(phrase, region_type=region_type, devices=devices)
        if dist.empty:
            logger.warning(f"/v1/regions пусто для '{phrase}'")
            continue

        top_regions = dist["regionId"].head(top_n_regions).tolist()

        for rid in top_regions:
            try:
                df = fetch_wordstat_daily_dynamics(phrase, rid, devices=devices)
                if not df.empty:
                    frames.append(df)
                time.sleep(sleep_s)
            except Exception as e:
                logger.exception(f"Ошибка daily dynamics '{phrase}' region={rid}: {e}")

    if not frames:
        return pd.DataFrame(columns=["date", "count", "share", "regionId", "phrase"])

    return pd.concat(frames, ignore_index=True)


# ----------------------------
# Main sender
# ----------------------------
def send_incidents_daily_regions_digest(tg_client, recipients, phrases=None, top_n_regions: int = 25):
    """
    Дневные графики + карта РФ (на 100k) + heatmap по регионам для инцидентных слов.
    По умолчанию: ["пожар", "взрыв"]
    """
    if phrases is None:
        phrases = ["пожар", "взрыв"]

    df = build_daily_region_dataset(
        phrases=phrases,
        top_n_regions=top_n_regions,
        region_type="regions",
        devices=["all"],
        sleep_s=0.25,
    )

    if df.empty:
        msg = "⚠️ Не удалось собрать daily динамику по регионам (df пустой)."
        for chat_id in recipients:
            try:
                tg_client.send_message(chat_id, msg)
            except Exception as e:
                logger.exception(f"Ошибка отправки в {chat_id}: {e}")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_total = os.path.join(WORDSTAT_MAPS_DIR, f"daily_total_incidents_{ts}.png")
    total_path = plot_daily_total_compare(df, phrases, out_total)

    map_paths = []
    for ph in phrases:
        out_map = os.path.join(WORDSTAT_MAPS_DIR, f"daily_map_{slugify_phrase(ph)}_{ts}.png")
        mp = plot_russia_choropleth_per_capita(
            df_daily=df,
            phrase=ph,
            out_png=out_map,
            window_days=7,
            top_n_regions=None
        )
        if mp:
            map_paths.append((ph, mp))

    heat_paths = []
    for ph in phrases:
        out_hm = os.path.join(WORDSTAT_MAPS_DIR, f"daily_heatmap_{slugify_phrase(ph)}_{ts}.png")
        hp = plot_daily_heatmap(df, ph, out_hm, top_regions=top_n_regions)
        if hp:
            heat_paths.append((ph, hp))

    from_date, to_date = _daily_bounds_last_60_days()
    header = (
        "🗺️ Wordstat: дневная динамика по регионам (топ регионов)\n"
        f"Фразы: {', '.join(phrases)}\n"
        f"Период: {from_date.strftime('%Y-%m-%d')} .. {to_date.strftime('%Y-%m-%d')}\n"
        "Источник: /v1/regions (топ за 30 дней) + /v1/dynamics daily (60 дней)\n"
        "Формат: суммарный график + карта РФ (на 100k) + heatmap (regionId × дата)."
    )

    for chat_id in recipients:
        try:
            tg_client.send_message(chat_id, header)

            if total_path:
                tg_client.send_photo(chat_id, photo=total_path)

            for ph, path in map_paths:
                tg_client.send_photo(chat_id, photo=path, caption=f"Карта РФ: {ph}")

            for ph, path in heat_paths:
                tg_client.send_photo(chat_id, photo=path, caption=f"Heatmap: {ph}")

            logger.info(f"✅ incidents daily+regions дайджест отправлен в {chat_id}")
        except Exception as e:
            logger.exception(f"⚠️ Ошибка отправки incidents дайджеста в {chat_id}: {e}")
