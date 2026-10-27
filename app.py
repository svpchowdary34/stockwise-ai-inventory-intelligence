# STOCKWISE AI — Retail Inventory Intelligence & Optimization | ONE Google Colab cell
# The right stock. At the right time.
import subprocess, sys, os, re, zipfile, html, math, warnings
from statistics import NormalDist
warnings.filterwarnings("ignore")
def _deps():
    try:
        from importlib.metadata import version
        assert int(version("gradio").split(".")[0]) in (4, 5)
        for p in ("plotly", "scikit-learn", "pandas", "numpy"): version(p)
    except Exception:
        print("Installing dependencies (about a minute)...")
        subprocess.call([sys.executable, "-m", "pip", "install", "-q", "gradio>=4.44,<6", "plotly", "scikit-learn>=1.2", "pandas", "numpy"])
_deps()
import numpy as np, pandas as pd, plotly.graph_objects as go, gradio as gr
from sklearn.ensemble import HistGradientBoostingRegressor

C = dict(fg="#173C35", sg="#BCE8B0", am="#E8B96A", iv="#F5F2E9", ch="#263B37", li="#D6C8F0", co="#D98276", gg="#A5B5AD")
G = {}; TESTS = []; _DEMO = {}
ST = ["Critical stockout risk", "Reorder now", "Needs data", "Potentially obsolete", "Slow-moving", "Overstocked", "Healthy"]
STC = dict(zip(ST, [C["co"], C["am"], C["gg"], C["li"], C["li"], C["am"], C["sg"]])); GL = dict(zip(ST, ["▲", "●", "?", "◆", "◇", "■", "✓"]))
S0 = dict(sl=.95, review=14, excess=120, slow=1.0, obs=180, hold=22.0, ocost=500.0, budget=0.0, sym="₹", fxr=1.0)
esc = lambda x: html.escape(str(x)); nul = lambda v: v is None or (not isinstance(v, str) and bool(pd.isna(v)))
nf = lambda v: "—" if nul(v) else ("∞" if np.isinf(v) else format(v, ",.0f"))
fx = lambda v, d=1: "—" if nul(v) else ("∞" if np.isinf(v) else format(v, "." + str(d) + "f"))
pc = lambda v, d=0: "—" if nul(v) else format(v, "." + str(d) + "%")
dts = lambda v: "—" if nul(v) else pd.Timestamp(v).strftime("%d %b %Y")
def money(v):
    if nul(v): return "—"
    S = G.get("S", S0); v = v * S["fxr"]; s = format(int(round(abs(v))), "d")
    if S["sym"] == "₹" and len(s) > 3: s = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", s[:-3]) + "," + s[-3:]
    elif S["sym"] != "₹": s = format(int(round(abs(v))), ",d")
    return ("-" if v < 0 else "") + S["sym"] + s
pill = lambda s: "<span class='pill' style='background:%s'>%s %s</span>" % (STC.get(s, C["gg"]), GL.get(s, ""), esc(s))
def note(t, k="info"): return "<div class='note' style='border-left:4px solid %s'>%s</div>" % ({"info": C["gg"], "ok": C["fg"], "warn": C["am"], "err": C["co"], "ai": C["li"]}[k], t)
def card(l, v, d, c=C["fg"]): return "<div class='kc'><div class='kl'>%s</div><div class='kv' style='color:%s'>%s</div><div class='kd'>%s</div></div>" % (l, c, v, d)
H = lambda t: "<div class='sec'>%s</div>" % t
def tbl(d, cols, fmt=None):
    fmt = fmt or {}; h = "<div class='tw'><table><tr>" + "".join("<th>%s</th>" % n for _, n in cols) + "</tr>"
    for _, r in d.iterrows(): h += "<tr>" + "".join("<td>%s</td>" % (fmt[k](r[k]) if k in fmt else ("—" if nul(r[k]) else esc(r[k]))) for k, _n in cols) + "</tr>"
    return h + "</table></div>"
def sty(f, h=360, t=""):
    f.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=C["ch"]), height=h, title=dict(text=t, x=.02), margin=dict(l=55, r=20, t=50, b=45), legend=dict(orientation="h", y=-.25), colorway=[C["fg"], C["am"], C["li"], C["co"], C["gg"]]); return f
def empty(m):
    f = go.Figure(); f.add_annotation(text=m, showarrow=False, font=dict(color=C["gg"], size=14)); f.update_xaxes(visible=False); f.update_yaxes(visible=False); return sty(f, 280)
def banner(): return note("<b>Illustrative synthetic data.</b> Fictional products, suppliers and sales; not a real retailer's performance.", "warn") if G.get("demo") else ""
def safe(n):
    def deco(fn):
        def w(*a):
            if "X" not in G: return (note("Load data first."),) + (None,) * (n - 1)
            try: return fn(*a)
            except Exception as e: return (note("Could not render this page: " + esc(e), "err"),) + (None,) * (n - 1)
        return w
    return deco
BLANK = ["", "nan", "none", "n/a", "na", "-", "null", "nat"]
def clean_str(s): t = s.astype(str).str.strip(); return t.where(~(s.isna() | t.str.lower().isin(BLANK)), np.nan)
def to_num(s): return s.astype(float) if pd.api.types.is_numeric_dtype(s) else pd.to_numeric(s.astype(str).str.replace(r"[₹$€£,%\s]", "", regex=True), errors="coerce")
def to_dt(s):
    try: return pd.to_datetime(s, errors="coerce", format="mixed")
    except Exception: return pd.to_datetime(s, errors="coerce")
def read_csv_any(f):
    p = f if isinstance(f, str) else getattr(f, "name", f)
    try: d = pd.read_csv(p)
    except Exception: d = pd.read_csv(p, sep=None, engine="python", encoding_errors="replace")
    if d.empty: raise ValueError("A file contains no rows.")
    return d

# ------------------------------ FICTIONAL synthetic retail data (fixed seed) ------------------------------
CAT_ITEMS = {"Staples": ["Basmati Rice 5kg", "Sona Masoori Rice 10kg", "Atta 10kg", "Toor Dal 1kg", "Moong Dal 1kg", "Chana Dal 1kg", "Sugar 1kg", "Salt 1kg", "Poha 500g", "Besan 500g", "Rava 1kg", "Jaggery 500g", "Idli Rice 5kg"],
 "Beverages": ["Tea 500g", "Filter Coffee 250g", "Instant Coffee 100g", "Green Tea 25 bags", "Mango Drink 1L", "Cola 2L", "Lemon Soda 600ml", "Energy Drink 250ml", "Coconut Water 200ml", "Buttermilk 200ml", "Mineral Water 1L", "Cold Coffee 200ml", "Malt Drink 500g"],
 "Snacks": ["Namkeen Mix 400g", "Potato Chips 150g", "Banana Chips 200g", "Glucose Biscuits", "Cream Biscuits", "Cookies 300g", "Roasted Peanuts 200g", "Murukku 250g", "Popcorn 100g", "Chocolate Bar 50g", "Wafers 75g", "Dry Fruit Mix 250g", "Rusk 300g"],
 "Personal Care": ["Shampoo 340ml", "Soap 4-pack", "Toothpaste 200g", "Face Wash 100ml", "Body Lotion 400ml", "Hair Oil 200ml", "Deodorant 150ml", "Sanitary Pads 20s", "Handwash 250ml", "Razor 3-pack", "Talcum Powder 300g", "Sunscreen 50g", "Lip Balm"],
 "Household": ["Detergent 2kg", "Dish Wash Liquid 750ml", "Floor Cleaner 1L", "Toilet Cleaner 500ml", "Garbage Bags 30s", "Tissue Roll 6s", "Agarbatti 100s", "Mosquito Coil 10s", "Phenyl 1L", "Scrub Pad 3s", "Matchbox 10s", "LED Bulb 9W", "Battery AA 4s"],
 "Dairy & Packaged": ["Paneer 200g", "Butter 500g", "Ghee 1L", "Cheese Slices 200g", "Curd 400g", "UHT Milk 1L", "Instant Noodles 5s", "Pasta 500g", "Pasta Sauce 400g", "Ketchup 1kg", "Pickle 400g", "Jam 500g", "Peanut Butter 340g"]}
SUPS = [("SUP1", "Sahyadri Foods", 5, .5, .05), ("SUP2", "Deccan Distributors", 7, 1, .1), ("SUP3", "Bharat Agro", 10, 2, .15), ("SUP4", "Konkan Beverages", 14, 3, .2), ("SUP5", "Gangetic FMCG", 21, 6, .3), ("SUP6", "Malabar Traders", 9, 1.5, .1), ("SUP7", "Nilgiri Care", 12, 2, .15), ("SUP8", "Metro Wholesale", 28, 7, .35)]
ASOF = pd.Timestamp("2026-10-26"); NDAYS = 730
def make_data(seed=42):
    r = np.random.default_rng(seed); dates = pd.date_range(ASOF - pd.Timedelta(days=NDAYS - 1), ASOF); T = len(dates); dow = dates.dayofweek.values; doy = dates.dayofyear.values; cats = list(CAT_ITEMS)
    typ = np.array(["fast"] * 12 + ["stable"] * 25 + ["seasonal"] * 14 + ["promo"] * 10 + ["slow"] * 17); r.shuffle(typ); skus, sm, inv, po, units_all, promo_all = [], [], [], [], [], []; nfast = nslow = npo = 0
    for k in range(78):
        t = typ[k]; cat = cats[k % 6]; items = CAT_ITEMS[cat]; nm = items[(k // 6) % len(items)]; sid, sname, blt, bsd, blate = SUPS[(k * 5 + k // 6) % 8]; sku = "SKU%03d" % (k + 1); skus.append(sku)
        base = {"fast": r.uniform(45, 120), "stable": r.uniform(10, 35), "seasonal": r.uniform(8, 30), "promo": r.uniform(15, 40), "slow": r.uniform(.15, 1.2)}[t]
        mu = base * (1 + r.uniform(-.25, .25) * np.linspace(-1, 1, T)) * (1 + .25 * (dow >= 5)); pr = np.zeros(T)
        if t == "seasonal": mu = mu * (1 + .5 * np.cos(2 * np.pi * (doy - int(r.choice([300, 100, 200]))) / 365))
        if t == "promo":
            for s0 in range(int(r.integers(10, 40)), T - 5, 45): pr[s0:s0 + 5] = 1
            mu = mu * (1 + .8 * pr)
        u = r.poisson(mu * r.lognormal(0, .25, T)).astype(float); obs = False
        if t == "slow":
            nslow += 1
            if nslow <= 2: u[-220:] = 0; obs = True
        units_all.append(u); promo_all.append(pr); rate = max(u[-90:].mean(), .05); cost = float(np.clip(r.lognormal(4.2, .7), 15, 900)); price = cost * float(r.uniform(1.15, 1.45)); bo = 0; lr = int(r.integers(3, 40)); fast_stock = False
        if t == "fast":
            nfast += 1
            if nfast <= 5: oh = int(rate * blt * r.uniform(.2, .45)); bo = int(rate * 1.5) if nfast <= 3 else 0; fast_stock = True
            else: oh = int(rate * (blt + r.uniform(10, 50)))
        elif obs: oh = int(r.uniform(40, 120)); lr = int(r.integers(300, 400))
        elif t == "slow": oh = int(max(rate * r.uniform(250, 500), 40)); lr = int(r.integers(150, 350))
        else: oh = int(rate * (blt + r.uniform(10, 50)))
        inv.append(dict(sku=sku, snapshot_date=ASOF, on_hand=oh, reserved=int(oh * .05), backorder=bo, last_receipt_date=ASOF - pd.Timedelta(days=lr)))
        if fast_stock and nfast <= 4: npo += 1; po.append(dict(po_id="PO%04d" % npo, sku=sku, qty=int(rate * blt * .5), order_date=ASOF - pd.Timedelta(days=3), expected_date=ASOF + pd.Timedelta(days=int(r.integers(2, 6)))))
        elif fast_stock: npo += 1; po.append(dict(po_id="PO%04d" % npo, sku=sku, qty=int(rate * blt), order_date=ASOF - pd.Timedelta(days=blt + 4), expected_date=ASOF - pd.Timedelta(days=4)))
        elif t == "stable" and r.random() < .3: npo += 1; po.append(dict(po_id="PO%04d" % npo, sku=sku, qty=int(rate * blt * .8), order_date=ASOF - pd.Timedelta(days=4), expected_date=ASOF + pd.Timedelta(days=int(r.integers(3, blt + 3)))))
        sm.append(dict(sku=sku, product_name=nm, category=cat, supplier_id=sid, unit_cost=round(cost, 2), unit_price=round(price, 2), moq=int(r.choice([12, 24, 48, 100])), pack_size=int(r.choice([1, 6, 12, 24])), lead_time_days=blt - (3 if sid in ("SUP5", "SUP8") else 0), shelf_life_days=90 if cat == "Dairy & Packaged" else np.nan))
    sales = pd.DataFrame(dict(date=np.tile(dates.values, 78), sku=np.repeat(skus, T), units=np.concatenate(units_all).astype(int), promo=np.concatenate(promo_all).astype(int))); hist = []
    for sid, sname, blt, bsd, blate in SUPS:
        for j in range(60):
            od = ASOF - pd.Timedelta(days=int(r.integers(20, NDAYS))); ltv = max(1, int(round(r.normal(blt + (4 if blate > .25 else 0), bsd))))
            if od + pd.Timedelta(days=ltv) <= ASOF: hist.append(dict(supplier_id=sid, supplier_name=sname, order_date=od, promised_date=od + pd.Timedelta(days=int(blt)), received_date=od + pd.Timedelta(days=ltv)))
    return dict(sales=sales, inv=pd.DataFrame(inv), sku=pd.DataFrame(sm), sup=pd.DataFrame(hist), po=pd.DataFrame(po))
def get_demo_raw(seed=42):
    if ("raw", seed) not in _DEMO: _DEMO[("raw", seed)] = make_data(seed)
    return _DEMO[("raw", seed)]

# ------------------------------ ingestion and validation ------------------------------
def _sp(s):
    o = {}
    for it in s.split(","):
        p = it.strip().split(":"); o[p[0]] = (p[1], p[2] == "1", p[3].split("|") if len(p) > 3 and p[3] else [])
    return o
SPEC = dict(sku=_sp("sku:s:1:sku_id|item_id,product_name:s:0:name,category:s:0,supplier_id:s:0:supplier,unit_cost:n:1:cost,unit_price:n:0:price,moq:n:0:min_order_qty,pack_size:n:0:pack,lead_time_days:n:0:lead_time,shelf_life_days:n:0"),
 sales=_sp("date:d:1,sku:s:1:sku_id,units:n:1:units_sold|quantity,promo:n:0:promotion"), inv=_sp("sku:s:1,snapshot_date:d:0:date,on_hand:n:1:on_hand_qty|stock,reserved:n:0,backorder:n:0,last_receipt_date:d:0"),
 sup=_sp("supplier_id:s:1,supplier_name:s:0,order_date:d:1,promised_date:d:0,received_date:d:1:delivery_date"), po=_sp("po_id:s:0,sku:s:1,qty:n:1:quantity|order_qty,order_date:d:0,expected_date:d:1:eta"))
TN = dict(sales="sales_history", inv="inventory_snapshot", sku="sku_master", sup="supplier_lead_times", po="open_purchase_orders"); KEY = dict(sku=["sku"], sales=["date", "sku"], inv=["sku", "snapshot_date"], po=["po_id", "sku"])
XR = dict(sup=[("received_date before order_date", lambda d: d.received_date < d.order_date)], po=[("expected_date before order_date", lambda d: d.expected_date < d.order_date)])
def parse_map(text):
    ov = {}
    for p in re.split(r"[,;\n]", text or ""):
        if "=" in p: k, v = p.split("=", 1); ov[re.sub(r"\W+", "_", k.strip().lower())] = re.sub(r"\W+", "_", v.strip().lower())
    return ov
nm_ = lambda s: re.sub(r"\W+", "_", str(s).strip().lower()).strip("_")
def map_cols(raw, t, ov, iss):
    d = raw.copy(); d.columns = [nm_(c) for c in d.columns]; ren = {}
    for f, (k, req, al) in SPEC[t].items():
        c = ov.get(t + "_" + f) or ov.get(f); c = c if c in d.columns else next((a for a in [f] + [nm_(x) for x in al] if a in d.columns and a not in ren), None)
        if c is None or c in ren:
            if req: iss.append(("error", "%s: required field '%s' not found. Columns found: %s. Use the mapping box, e.g. %s=your_column." % (TN[t], f, ", ".join(map(str, raw.columns[:12])), f)))
            continue
        if c != f and f in d.columns: d = d.drop(columns=[f])
        ren[c] = f
    d = d.rename(columns=ren); return d.loc[:, ~d.columns.duplicated()]
def ctab(raw, t, ov, iss, ex):
    d = map_cols(raw, t, ov, iss)
    if any(l == "error" for l, _ in iss): return d
    d = d.copy(); d["_row"] = d.index + 2; bad = pd.Series(False, index=d.index)
    for f, (k, req, al) in SPEC[t].items():
        if f not in d.columns: d[f] = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]") if k == "d" else np.nan
        elif k == "n": d[f] = to_num(d[f])
        elif k == "d": d[f] = to_dt(d[f])
        else: d[f] = clean_str(d[f])
    if "sku" in d.columns:
        o = d["sku"].copy(); d["sku"] = d["sku"].str.upper().str.replace(r"\s+", "", regex=True); n = int(((o != d["sku"]) & o.notna()).sum())
        if n: iss.append(("info", "%s: %d SKU identifiers were normalised (trimmed, upper-cased, spaces removed) to reconcile with the SKU master." % (TN[t], n)))
    def flag(m, why):
        nonlocal bad
        m = m.fillna(False) & ~bad
        if m.any(): ex.append((TN[t], why, int(m.sum()))); iss.append(("warn", "%s: %d rows excluded: %s (rows %s)" % (TN[t], m.sum(), why, ", ".join(str(int(i)) for i in d["_row"][m].values[:6])))); bad = bad | m
    for f, (k, req, al) in SPEC[t].items():
        if req: flag(d[f].isna(), "missing or invalid " + f)
        if k == "n" and f != "on_hand": flag(d[f] < 0, "negative " + f)
    for why, fn in XR.get(t, []): flag(fn(d), why)
    if t in KEY and not (t == "po" and d.po_id.isna().all()): flag(d.duplicated(KEY[t], keep="first"), "duplicate " + "+".join(KEY[t]))
    return d[~bad].drop(columns="_row").reset_index(drop=True)
def load_all(raw, mapping=""):
    iss, ex, ov, D = [], [], parse_map(mapping), {}; rows_in = {t: len(raw[t]) for t in SPEC if raw.get(t) is not None}
    for t in ("sales", "inv", "sku"):
        if raw.get(t) is None: iss.append(("error", "Missing required table: %s." % TN[t]))
    if any(l == "error" for l, _ in iss): raise ValueError(" | ".join(m for l, m in iss if l == "error"))
    for t in rows_in: D[t] = ctab(raw[t], t, ov, iss, ex)
    errs = [m for l, m in iss if l == "error"]
    if errs: raise ValueError(" | ".join(errs))
    for t in ("sup", "po"):
        if t not in D: D[t] = pd.DataFrame({f: pd.Series(dtype="datetime64[ns]" if k == "d" else float if k == "n" else object) for f, (k, _, _) in SPEC[t].items()}); iss.append(("info", "No %s table: analyses that need it are downgraded (%s)." % (TN[t], "supplier reliability cannot be evaluated; stated lead times are used" if t == "sup" else "no open orders assumed")))
    if D["sku"].empty or D["sales"].empty or D["inv"].empty: raise ValueError("No valid rows remain in the SKU master, sales or inventory table after validation.")
    ids = set(D["sku"].sku)
    for t in ("sales", "inv", "po"):
        m = ~D[t].sku.isin(ids)
        if m.any(): ex.append((TN[t], "SKU not in SKU master", int(m.sum()))); iss.append(("warn", "%s: %d rows excluded: SKU not found in the SKU master (%s)." % (TN[t], m.sum(), ", ".join(D[t].sku[m].unique()[:5])))); D[t] = D[t][~m].reset_index(drop=True)
    inv = D["inv"]; n = int((inv.on_hand < 0).sum())
    if n: iss.append(("warn", "%d inventory rows have negative on-hand (kept as oversold/backorder balances; available stock is clipped at 0, inventory position uses the raw value)." % n))
    D["inv"] = inv.sort_values("snapshot_date", na_position="first").drop_duplicates("sku", keep="last").reset_index(drop=True)
    if D["sku"].moq.isna().any() or D["sku"].pack_size.isna().any(): iss.append(("info", "Missing MOQ / pack size are treated as no constraint (MOQ 0, pack 1): an explicit assumption."))
    D["sales"]["promo"] = D["sales"]["promo"].fillna(0); T_ = D["sales"].date.nunique(); cov = len(D["sales"]) / max(T_ * D["sales"].sku.nunique(), 1)
    if cov < .5: iss.append(("info", "Sales rows cover only %.0f%% of SKU-days: days with no row are treated as zero sales (assumption; verify your extract is not missing days)." % (100 * cov)))
    if T_ < 180: iss.append(("warn", "Only %d days of sales history: forecasting falls back to a documented baseline and results are unreliable." % T_))
    D.update(issues=iss, excluded=ex, rows_in=rows_in, asof=D["sales"].date.max()); return D

# ------------------------------ forecasting (time-aware) ------------------------------
FEATS = ["m7", "m28", "m91", "sd28", "pr28", "yoy", "mo", "wk", "cat"]
def build_rows(Mv, Pv, dates, catc, h, origins):
    T, N = Mv.shape; z = np.zeros((1, N)); cs = np.vstack([z, np.cumsum(Mv, 0)]); cs2 = np.vstack([z, np.cumsum(Mv ** 2, 0)]); cp = np.vstack([z, np.cumsum(Pv, 0)]); out = []
    for i in origins:
        m7 = (cs[i + 1] - cs[i - 6]) / 7; m28 = (cs[i + 1] - cs[i - 27]) / 28; m91 = (cs[i + 1] - cs[i - 90]) / 91; sd28 = np.sqrt(np.maximum((cs2[i + 1] - cs2[i - 27]) / 28 - m28 ** 2, 0)); pr28 = (cp[i + 1] - cp[i - 27]) / 28; a = i + 1 - 364
        yoy = (cs[i + h - 364 + 1] - cs[a]) / h if a >= 0 else np.full(N, np.nan); mo = (dates[i] + pd.Timedelta(days=1 + h // 2)).month; wk = float((((dates[i].dayofweek + 1 + np.arange(h)) % 7) >= 5).mean()); y = (cs[i + h + 1] - cs[i + 1]) / h if i + h < T else np.full(N, np.nan)
        out.append(pd.DataFrame(dict(i=i, sku_i=np.arange(N), m7=m7, m28=m28, m91=m91, sd28=sd28, pr28=pr28, yoy=yoy, mo=mo, wk=wk, cat=catc, y=y)))
    return pd.concat(out, ignore_index=True)
def metr(y, p, h):
    e = (p - y) * h; den = (y * h).sum(); return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())), WAPE=float(np.abs(e).sum() / den) if den > 0 else np.nan)
def run_forecast(Mv, Pv, dates, catc):
    T, N = Mv.shape; res = dict(perf=[], tot={}, lo={}, hi={}, pred=None, msg=[], model={}, wape={}, split={}, daily={})
    for h in (7, 30, 90):
        if T < 91 + h + 14:
            r0 = Mv[-min(28, T):].mean(0); res["tot"][h] = r0 * h; res["lo"][h] = res["hi"][h] = res["tot"][h]; res["model"][h] = "Moving average (history too short)"; res["daily"][h] = r0; res["msg"].append("h=%d: history too short (%d days): moving-average baseline, no evaluation or interval." % (h, T)); continue
        R = build_rows(Mv, Pv, dates, catc, h, range(90, T - h, 7)); F = build_rows(Mv, Pv, dates, catc, h, [T - 1]); cut = int(T * .75); tr = R[R.i + h <= cut]; te = R[R.i >= cut]; res["split"][h] = (int(tr.i.max() + h) if len(tr) else -1, cut, int(te.i.min()) if len(te) else 10 ** 9)
        cand = {"Moving average (28d)": (te.m28.values, F.m28.values)}
        if len(te) and te.yoy.notna().mean() >= .9: cand["Seasonal naive (last year)"] = (te.yoy.fillna(te.m28).values, F.yoy.fillna(F.m28).values)
        ml = len(tr) >= 300 and len(te) >= 50 and tr.y.sum() > 0; mdl = None
        if ml:
            mdl = HistGradientBoostingRegressor(max_iter=150, learning_rate=.08, max_depth=4, random_state=0).fit(tr[FEATS], tr.y); cand["HistGradientBoosting (ML)"] = (np.maximum(mdl.predict(te[FEATS]), 0), None)
        else: res["msg"].append("h=%d: ML disabled (needs 300+ training and 50+ held-out rows; found %d / %d): baseline used." % (h, len(tr), len(te)))
        if not len(te): res["tot"][h] = F.m28.values * h; res["lo"][h] = res["hi"][h] = res["tot"][h]; res["model"][h] = "Moving average (no holdout)"; res["daily"][h] = F.m28.values; continue
        sc = {k: metr(te.y.values, v[0], h) for k, v in cand.items()}; best = min(sc, key=lambda k: (np.inf if np.isnan(sc[k]["WAPE"]) else sc[k]["WAPE"], sc[k]["MAE"]))
        for k in sc: res["perf"].append(dict(horizon=h, model=k, **sc[k], selected=(k == best)))
        if best.startswith("Hist"):
            rr = R[R.y.notna()]; mdl = HistGradientBoostingRegressor(max_iter=150, learning_rate=.08, max_depth=4, random_state=0).fit(rr[FEATS], rr.y); fp = np.maximum(mdl.predict(F[FEATS]), 0)
        else: fp = cand[best][1]
        p_te = cand[best][0]; rel = (te.y.values - p_te) / (p_te + .5); q1, q2 = np.nanquantile(rel, .1), np.nanquantile(rel, .9); res["tot"][h] = fp * h; res["lo"][h] = np.maximum(fp + q1 * (fp + .5), 0) * h; res["hi"][h] = (fp + q2 * (fp + .5)) * h; res["model"][h] = best; res["wape"][h] = sc[best]["WAPE"]; res["daily"][h] = fp
        if h == 30: res["pred"] = te.assign(p=p_te)[["i", "sku_i", "y", "p"]]
    return res

# ------------------------------ replenishment engine ------------------------------
zval = lambda sl: NormalDist().inv_cdf(min(max(sl, .5), .9999))
def ssf(z, rate, sd, lt, sdlt): return z * np.sqrt(np.maximum(lt * sd ** 2 + rate ** 2 * sdlt ** 2, 0))
def rnd(q, moq, pack):
    q = np.asarray(q, float); pk = np.maximum(np.nan_to_num(np.asarray(pack, float), nan=1.0), 1.0); mq = np.maximum(np.nan_to_num(np.asarray(moq, float), nan=0.0), 0.0); r = np.where(q > 0, np.ceil(q / pk) * pk, 0.0); return np.where((r > 0) & (r < mq), np.ceil(mq / pk) * pk, r)
SC0 = dict(dm=1.0, seas=0.0, ltm=1.0, sl=.95, pol="Statistical (z-score)", ssd=7.0)
def plan(B, S, sc=None):
    sc = dict(SC0, **(sc or {})); P = B.copy(); f = sc["dm"] * (1 + sc["seas"] / 100); rate = P["rate"].values * f; sd = P["sd"].values * f; ltd = P["ltd"].values * sc["ltm"]; sdlt = P["sdlt"].values * sc["ltm"]; z = zval(sc["sl"])
    ss = ssf(z, rate, sd, ltd, sdlt) if sc["pol"].startswith("Stat") else (rate * sc["ssd"] if sc["pol"].startswith("Fixed") else np.zeros(len(P))); rop = rate * ltd + ss; tgt = rop + rate * S["review"]
    oh = P["on_hand"].values.astype(float); rs = P["reserved"].fillna(0).values; bo = P["backorder"].fillna(0).values; poq = P["po_qty"].fillna(0).values; avail = np.maximum(oh - rs, 0); eff = np.maximum(oh - rs - bo, 0); pos = oh - rs - bo + poq
    with np.errstate(divide="ignore", invalid="ignore"): cover = np.where(rate > 0, eff / rate, np.inf); dos = np.where(rate > 0, np.maximum(pos, 0) / rate, np.inf)
    nxt = P["po_next"].values.astype(float); e = np.where(np.isfinite(nxt), nxt, ltd); valid = np.isfinite(oh) & np.isfinite(ltd) & np.isfinite(rate); raw = np.where(pos <= rop, np.maximum(tgt - pos, 0), 0.0); qty = rnd(np.nan_to_num(raw), P["moq"].values, P["pack"].values)
    crit = valid & (rate > 0) & (cover < e); reord = valid & ~crit & (pos <= rop) & (qty > 0); obs = valid & ~crit & ~reord & (oh > 0) & (P["last_sale"].values >= S["obs"]) & (np.nan_to_num(P["rcpt_age"].values, nan=-1) >= S["obs"]); slow = valid & ~(crit | reord | obs) & (rate < S["slow"]); over = valid & ~(crit | reord | obs | slow) & (dos > S["excess"])
    P["state"] = np.select([~valid, crit, reord, obs, slow, over], [ST[2], ST[0], ST[1], ST[3], ST[4], ST[5]], default=ST[6]); P["urg"] = P["state"].map({s: i for i, s in enumerate(ST)}).values
    P["rate"], P["sd"], P["ltd"], P["ss"], P["rop"], P["tgt"], P["pos"], P["avail"], P["cover"], P["dos"], P["e"] = rate, sd, ltd, ss, rop, tgt, pos, avail, cover, dos, e; P["qty"] = np.where(valid, qty, np.nan); P["order_value"] = P["qty"] * P["cost"]
    exu = np.where(rate > 0, np.maximum(avail - rate * S["excess"], 0), avail); P["excess_units"] = np.where(valid, exu, 0.0); P["excess_value"] = P["excess_units"] * P["cost"]; P["inv_value"] = np.maximum(np.nan_to_num(oh), 0) * P["cost"]
    P["arrival"] = [asof_ + pd.Timedelta(days=int(math.ceil(l))) if (q > 0 and np.isfinite(l)) else pd.NaT for asof_, l, q in zip([B.attrs.get("asof", ASOF)] * len(P), ltd, np.nan_to_num(P["qty"].values))]
    short = np.where(crit, rate * P["price"].values * np.maximum(e - cover, 0), 0); P["impact"] = np.where(crit, short, np.where(reord, P["order_value"], P["excess_value"])); P["impact"] = np.nan_to_num(P["impact"])
    fq, fs = fund(P, sc.get("budget", S["budget"])); P["fq"], P["fstat"] = fq, fs; P["funded_value"] = P["fq"] * P["cost"]
    hd, nz = P["hdays"].values, P["nz"].values; P["conf"] = np.select([~valid | (hd < 180) | (nz < 20), (hd >= 365) & (P["lt_src"].values == "observed history") & (P["cv"].fillna(9).values < 1)], ["Low", "High"], default="Medium"); P["_sl"] = sc["sl"]; P["_hold"] = S["excess"]
    P["action"] = [action(r) for r in P.to_dict("records")]; P["expl"] = [explain(r, S) for r in P.to_dict("records")]; return P
def fund(P, budget):
    q = np.nan_to_num(P["qty"].values).copy(); cost = P["cost"].values; fq = q.copy(); st = np.where(q > 0, "Funded", "—").astype(object)
    if budget and budget > 0:
        order = np.lexsort((-np.nan_to_num(P["impact"].values), P["urg"].values)); rem = float(budget); fq = np.zeros(len(P))
        for j in order:
            if q[j] <= 0: continue
            c = q[j] * cost[j]
            if c <= rem: fq[j] = q[j]; rem -= c; st[j] = "Funded"
            else:
                pk = P["pack"].iloc[j]; pk = 1 if nul(pk) else max(pk, 1); mq = P["moq"].iloc[j]; mq = 0 if nul(mq) else mq; pq = math.floor(rem / cost[j] / pk) * pk
                if pq > 0 and pq >= mq: fq[j] = pq; rem -= pq * cost[j]; st[j] = "Partly funded (budget)"
                else: st[j] = "Unfunded (budget)"
    return fq, st
def action(r):
    s = r["state"]; q = r["qty"]
    if s == ST[2]: return "Supply the missing lead-time or stock data"
    if s == ST[0]: return ("Order %s units now and expedite any open PO" % nf(q)) if q > 0 else "Expedite the open purchase order"
    if s == ST[1]: return "Order %s units (arrive about %s)" % (nf(q), dts(r["arrival"]))
    if s == ST[3]: return "Stop replenishment; review transfer or markdown after checking business reasons"
    if s == ST[4]: return "Keep order quantities minimal; review range and assortment"
    if s == ST[5]: return "Pause new orders; consider transfer or markdown of %s excess units" % nf(r["excess_units"])
    return "No action: within policy"
def explain(r, S):
    s = r["state"]; head = "%s (%s)" % (r["sku"], r["pname"])
    if s == ST[2]: return "%s: needs data. Lead time or stock level is missing, so no recommendation is made (no value is invented)." % head
    sup = "Supports: available %s units, forecast demand %s/day, lead time %s d (%s), service level %s, reorder point %s, inventory position %s." % (nf(r["avail"]), fx(r["rate"], 1), fx(r["ltd"], 0), r["lt_src"], pc(r["_sl"]), nf(r["rop"]), nf(r["pos"]))
    if s == ST[0]: w = "projected to run out in %s days, before the next delivery (%s days)" % (fx(r["cover"]), fx(r["e"], 0)); y = "about %s of revenue is at risk" % money(r["impact"]); t = "Expediting costs more; review the forecast if a promotion is expected."
    elif s == ST[1]: w = "inventory position %s is at or below the reorder point %s" % (nf(r["pos"]), nf(r["rop"])); y = "waiting risks a stock-out before the next delivery"; t = "Quantity is rounded to pack size and MOQ and may be limited by budget (%s)." % r["fstat"]
    elif s == ST[3]: w = "no sales for %s days and last receipt %s days ago while holding %s units" % (nf(r["last_sale"]), nf(r["rcpt_age"]), nf(r["on_hand"])); y = "capital is tied up (%s)" % money(r["inv_value"]); t = "Low sales alone is not obsolescence: confirm discontinuation, seasonality or listing status first."
    elif s == ST[4]: w = "velocity %s/day is below the %s/day threshold" % (fx(r["rate"], 2), fx(S["slow"], 1)); y = "slow turnover ties up cash"; t = "Reducing stock may raise stock-out risk on rare orders."
    elif s == ST[5]: w = "stock covers %s days, above the %s-day limit" % (fx(r["dos"], 0), S["excess"]); y = "%s of stock is excess and incurs carrying cost" % money(r["excess_value"]); t = "Markdowns reduce margin; transfers need another location with demand."
    else: w = "stock is within policy (cover %s days)" % fx(r["dos"], 0); y = "no problem detected"; t = "Re-check after forecast or lead-time changes."
    return "%s: %s. Why it matters: %s. Next: %s. %s Trade-off: %s Confidence: %s." % (head, w, y, r["action"], sup, t, r["conf"])

# ------------------------------ inventory simulation ------------------------------
def simulate(B, S, pol, world, H=90, budget=0.0, hold=22.0, ocost=500.0, seed=7):
    Pp = plan(B, dict(S, budget=0.0), pol); N = len(B); rng = np.random.default_rng(seed); Zd = rng.standard_normal((H, N)); Zl = rng.standard_normal((H, N)); f = world["dm"] * (1 + world["seas"] / 100)
    rw = np.nan_to_num(B["rate"].values * f); sdw = np.nan_to_num(B["sd"].values * f); lw = np.nan_to_num(B["ltd"].values * world["ltm"], nan=7.0); slw = np.nan_to_num(B["sdlt"].values * world["ltm"]); rel = np.where(rw > 0, np.clip(sdw / np.where(rw > 0, rw, 1), 0, 3), 0.0)
    valid = (Pp["state"].values != ST[2]) & np.isfinite(Pp["rop"].values); rop = np.nan_to_num(Pp["rop"].values); tgt = np.nan_to_num(Pp["tgt"].values); cost = B["cost"].values; price = B["price"].values; moq = B["moq"].values; pack = B["pack"].values
    A = np.zeros((H + 130, N)); poq = B["po_qty"].fillna(0).values; nx = B["po_next"].values.astype(float)
    for j in np.nonzero(poq > 0)[0]: A[int(np.clip(max(1, math.ceil(nx[j] * world["ltm"])) if np.isfinite(nx[j]) else max(1, round(lw[j])), 1, H + 120)), j] += poq[j]
    bal = np.nan_to_num((B["on_hand"].fillna(0) - B["reserved"].fillna(0) - B["backorder"].fillna(0)).values.astype(float)); bal0 = bal.copy(); pipe = A.sum(0); po_total = pipe.copy(); prio = np.lexsort((-np.nan_to_num(Pp["impact"].values), Pp["urg"].values))
    rec_t = sold_t = lost_t = ord_t = spend = inv_acc = 0.0; lost_sku = np.zeros(N); dem_t = 0.0; norders = 0; rec_v = np.zeros(N); sold_v = np.zeros(N); ord_v = np.zeros(N)
    for t in range(H):
        rec = A[t]; bal = bal + rec; pipe = pipe - rec; rec_v += rec; pos = bal + pipe; q = rnd(np.where(valid & (pos <= rop), np.maximum(tgt - pos, 0), 0), moq, pack)
        if budget > 0 and q.any():
            for j in prio:
                if q[j] <= 0: continue
                c = q[j] * cost[j]
                if spend + c > budget:
                    pk = 1 if nul(pack[j]) else max(pack[j], 1); pq = math.floor((budget - spend) / cost[j] / pk) * pk; q[j] = pq if (pq > 0 and pq >= (0 if nul(moq[j]) else moq[j])) else 0
                spend += q[j] * cost[j]
        idx = np.nonzero(q > 0)[0]
        if len(idx): L = np.clip(np.rint(lw + slw * Zl[t]), 1, 120).astype(int); np.add.at(A, (t + L[idx], idx), q[idx]); pipe = pipe + q; ord_v += q; norders += len(idx)
        dem = np.maximum(np.rint(rw * (1 + rel * Zd[t])), 0); sold = np.minimum(np.maximum(bal, 0), dem); lost = dem - sold; bal = bal - sold; sold_v += sold; lost_sku += lost; dem_t += dem.sum(); inv_acc += float(np.maximum(bal, 0) @ cost)
    lost_t = lost_sku.sum(); end_inv = float(np.maximum(bal, 0) @ cost); excess_end = float((np.maximum(np.maximum(bal, 0) - rw * S["excess"], 0) * cost).sum()); avg_inv = inv_acc / H; hc = avg_inv * hold / 100 * H / 365; oc = norders * ocost; lm = float((lost_sku * (price - cost)).sum())
    cons = bool(np.allclose(bal0 + rec_v - sold_v, bal, atol=1e-6)) and bool(np.allclose(po_total + ord_v, rec_v + A[H:].sum(0) + 0.0 + (A[:H].sum(0) - rec_v), atol=1e-6))
    return dict(fill=1 - lost_t / dem_t if dem_t > 0 else np.nan, so_skus=int((lost_sku > 0).sum()), lost_rev=float((lost_sku * price).sum()), avg_inv=avg_inv, end_inv=end_inv, excess_end=excess_end, spend=spend if budget > 0 else float((ord_v * cost).sum()), norders=norders, hold=hc, order_cost=oc, lost_margin=lm, total=hc + oc + lm, cons=cons)

# ------------------------------ analysis pipeline ------------------------------
def sup_stats(D):
    h = D["sup"]
    if h is None or h.empty: return pd.DataFrame(columns=["supplier_id", "n", "mean", "sd", "p90", "late"])
    h = h.assign(l=(h.received_date - h.order_date).dt.days); h["late"] = ((h.received_date - h.promised_date).dt.days > 0).astype(float).where(h.promised_date.notna())
    return h.groupby("supplier_id").agg(n=("l", "size"), mean=("l", "mean"), sd=("l", "std"), p90=("l", lambda s: s.quantile(.9)), late=("late", "mean")).reset_index()
def analyze(D, S):
    sk = D["sku"].set_index("sku"); skus = list(sk.index); N = len(skus); sales = D["sales"]; asof = sales.date.max(); dates = pd.date_range(sales.date.min(), asof); T = len(dates)
    M = sales.pivot_table(index="date", columns="sku", values="units", aggfunc="sum").reindex(index=dates, columns=skus).fillna(0.0); Pm = sales.pivot_table(index="date", columns="sku", values="promo", aggfunc="max").reindex(index=dates, columns=skus).fillna(0.0)
    Mv, Pv = M.values.astype(float), Pm.values.astype(float); catc = pd.factorize(sk.category.fillna("NA"))[0]; fc = run_forecast(Mv, Pv, dates, catc); rate = fc["daily"].get(30, Mv[-28:].mean(0)); sd = Mv[-90:].std(0, ddof=1) if T > 2 else np.zeros(N)
    nzm = Mv > 0; last_idx = np.where(nzm.any(0), T - 1 - np.argmax(nzm[::-1], 0), -1); last_sale = np.where(last_idx >= 0, T - 1 - last_idx, T); first_idx = np.where(nzm.any(0), np.argmax(nzm, 0), T); hdays = T - first_idx
    wk_ = M.resample("W").sum(); w52 = wk_.tail(52); cv = (w52.std() / w52.mean().replace(0, np.nan)).reindex(skus).values if len(w52) >= 26 else np.full(N, np.nan); tv = Mv[-28:].mean(0) / np.where(Mv[-91:].mean(0) > 0, Mv[-91:].mean(0), np.nan) - 1
    ann = pd.Series(Mv[-365:].sum(0) * sk.unit_cost.values, index=skus).sort_values(ascending=False); cum = ann.cumsum() / max(ann.sum(), 1e-9); abc = pd.Series(np.where(cum - ann / max(ann.sum(), 1e-9) < .8, "A", np.where(cum - ann / max(ann.sum(), 1e-9) < .95, "B", "C")), index=ann.index).reindex(skus).values
    xyz = np.where(np.isnan(cv), "n/a", np.where(cv <= .5, "X", np.where(cv <= 1, "Y", "Z"))); ss_ = sup_stats(D); ssi = ss_.set_index("supplier_id") if len(ss_) else pd.DataFrame(); inv = D["inv"].set_index("sku").reindex(skus)
    pod = D["po"]; pg = pod.groupby("sku").qty.sum().reindex(skus).fillna(0); fut = pod[pod.expected_date >= asof].assign(d=lambda x: (x.expected_date - asof).dt.days).groupby("sku").d.min().reindex(skus).fillna(np.inf); od = (pod[pod.expected_date < asof].groupby("sku").qty.sum().reindex(skus).fillna(0) > 0)
    ltd, sdlt, src = [], [], []
    for s in skus:
        sid = sk.supplier_id[s]; st_ = sk.lead_time_days[s]
        if len(ssi) and sid in ssi.index and ssi.loc[sid, "n"] >= 10: ltd.append(float(ssi.loc[sid, "mean"])); sdlt.append(float(np.nan_to_num(ssi.loc[sid, "sd"]))); src.append("observed history")
        elif not nul(st_): ltd.append(float(st_)); sdlt.append(0.0); src.append("stated, variability unknown")
        else: ltd.append(np.nan); sdlt.append(0.0); src.append("missing")
    B = pd.DataFrame(dict(sku=skus, pname=sk.product_name.fillna(pd.Series(skus, index=skus)).values, cat=sk.category.fillna("NA").values, sup=sk.supplier_id.fillna("NA").values, cost=sk.unit_cost.values, price=sk.unit_price.fillna(sk.unit_cost * 1.3).values, moq=sk.moq.values, pack=sk.pack_size.values, ltd=ltd, sdlt=sdlt, lt_src=src, lt_stated=sk.lead_time_days.values,
                          on_hand=inv.on_hand.values, reserved=inv.reserved.values, backorder=inv.backorder.values, last_receipt=inv.last_receipt_date.values, po_qty=pg.values, po_next=fut.values, po_overdue=od.values, rate=rate, sd=sd, hdays=hdays, nz=nzm.sum(0), last_sale=last_sale, cv=cv, trend=tv, abc=abc, xyz=xyz))
    B["rcpt_age"] = (asof - pd.to_datetime(B["last_receipt"])).dt.days; B.attrs["asof"] = asof; X = dict(D=D, S=S, B=B, fc=fc, asof=asof, dates=dates, M=M, Mw=wk_, Pv=Pv, skus=skus, ss=ss_, supname=(D["sup"].drop_duplicates("supplier_id").set_index("supplier_id").supplier_name.to_dict() if "supplier_name" in D["sup"] and len(D["sup"]) else {}))
    X["P"] = plan(B, S); iss = list(D["issues"]); spk = int((Mv > Mv.mean(0) + 6 * Mv.std(0) + 1).sum())
    if spk: iss.append(("info", "%d daily sales values are more than 6 standard deviations above the SKU mean (potential anomalies or promotions; kept)." % spk))
    X["issues"] = iss; return X

# ------------------------------ pages ------------------------------
def flt(P, c, s, st):
    if c: P = P[P["cat"].isin(c)]
    if s: P = P[P["sup"].isin(s)]
    if st: P = P[P["state"].isin(st)]
    return P
@safe(4)
def render_cc(c, s, st):
    X = G["X"]; S = X["S"]; P = flt(X["P"], c, s, st); fc = X["fc"]; inv_val = P["inv_value"].sum(); below = int((P["pos"] <= P["rop"]).sum()); crit = int((P["state"] == ST[0]).sum()); exc = P["excess_value"].sum(); slowv = P[P["state"].isin([ST[3], ST[4]])]["inv_value"].sum(); dosm = np.minimum(P["dos"], 365).mean(); acc = fc["wape"].get(30)
    h = banner() + "<div class='kg'>" + card("Inventory value", money(inv_val), "on-hand stock at cost") + card("Below reorder point", "%d (%s)" % (below, pc(below / max(len(P), 1))), "inventory position ≤ reorder point", C["am"]) + card("Stockout-risk SKUs", nf(crit), "▲ will run out before next delivery", C["co"]) + card("Excess stock value", money(exc), "slow-moving stock value: " + money(slowv), C["am"]) + \
        card("Avg days of supply", fx(dosm, 0), "capped at 365 days") + card("Forecast accuracy", pc(1 - acc) if acc == acc and acc is not None else "n/a", "1 − WAPE, 30-day, held-out" if acc else "not measurable") + card("Carrying-cost exposure", money(inv_val * S["hold"] / 100), "per year at %.0f%% holding cost (assumption); excess part %s" % (S["hold"], money(exc * S["hold"] / 100)), C["am"]) + "</div>"
    h += note("Position = on hand − reserved − backorders + open purchase orders. Forecasts are model estimates; classifications and actions are proposals for review by buyers.", "info"); q = P[P["urg"] <= 5].sort_values(["urg", "impact"], ascending=[True, False]).head(10)
    h += H("Priority action queue") + (tbl(q, [("sku", "SKU"), ("pname", "Product"), ("state", "Status"), ("action", "Recommended action"), ("impact", "Impact"), ("expl", "Why and evidence")], dict(state=pill, impact=money)) if len(q) else note("No priority actions in this view.", "ok"))
    a = P[P["fq"] > 0]; h += H("Recommended actions summary") + "<div class='note'>%d replenishment orders worth %s (%s units); %d SKUs overstocked or slow with %s excess; %d SKUs need data.</div>" % (len(a), money(a["funded_value"].sum()), nf(a["fq"].sum()), int(P["state"].isin([ST[3], ST[4], ST[5]]).sum()), money(exc), int((P["state"] == ST[2]).sum()))
    w = X["Mw"][list(P["sku"])].sum(1); f1 = go.Figure([go.Scatter(x=w.index, y=w.values, name="Weekly units", line=dict(color=C["gg"], width=1)), go.Scatter(x=w.index, y=w.rolling(4, min_periods=1).mean().values, name="4-week mean", line=dict(color=C["fg"], width=3))]); f1 = sty(f1, 340, "Demand trend (units per week)")
    cnt = P["state"].value_counts(); f2 = sty(go.Figure(go.Bar(x=[GL[x] + " " + x for x in ST], y=[int(cnt.get(x, 0)) for x in ST], marker_color=[STC[x] for x in ST])), 340, "SKUs by inventory state"); f2.update_xaxes(tickangle=-25); g = P.groupby("cat")["inv_value"].sum().sort_values(); f3 = sty(go.Figure(go.Bar(y=g.index, x=g.values * S["fxr"], orientation="h", marker_color=C["fg"])), 320, "Inventory value by category (%s)" % S["sym"])
    return h, f1, f2, f3
@safe(4)
def render_dem(c, s, st, hz, sku):
    X = G["X"]; S = X["S"]; fc = X["fc"]; P = flt(X["P"], c, s, st); hz = int(hz or 30); idx = {k: i for i, k in enumerate(X["skus"])}; sel = [idx[k] for k in P["sku"]] if not sku or sku == "All SKUs" else [idx[sku]]
    h = banner() + note("<b>Method.</b> Baselines: 28-day moving average and seasonal naive (same window last year). ML: HistGradientBoosting on past-only features (7/28/91-day means, 28-day volatility, promo share, last year's window, calendar of the target window, category). Chronological split: models train on origins whose whole target window ends before the cut-off (75% of history) and are tested on origins after it, so no future data enters training. Selection uses held-out WAPE.", "info")
    if fc["msg"]: h += note(esc(" | ".join(fc["msg"])), "warn")
    if fc["perf"]: pf = pd.DataFrame(fc["perf"]); h += H("Held-out model comparison (WAPE = Σ|error| ÷ Σ actual; zero-demand safe)") + tbl(pf, [("horizon", "Horizon (d)"), ("model", "Model"), ("MAE", "MAE (units)"), ("RMSE", "RMSE (units)"), ("WAPE", "WAPE"), ("selected", "Selected")], dict(MAE=fx, RMSE=fx, WAPE=lambda v: pc(v, 1), selected=lambda v: "✓ selected" if v else ""))
    tab = pd.DataFrame(dict(sku=X["skus"], pname=X["B"]["pname"], cat=X["B"]["cat"], f7=fc["tot"][7], f30=fc["tot"][30], f90=fc["tot"][90], lo30=fc["lo"][30], hi30=fc["hi"][30], model=fc["model"][30], trend=X["B"]["trend"])); X["fctab"] = tab; t2 = tab[tab["sku"].isin(P["sku"])].sort_values("f30", ascending=False).head(40)
    h += H("SKU forecasts (units, next 7 / 30 / 90 days; 30-day range = empirical 10th-90th percentile of held-out error)") + tbl(t2, [("sku", "SKU"), ("pname", "Product"), ("f7", "7 d"), ("f30", "30 d"), ("lo30", "30 d low"), ("hi30", "30 d high"), ("f90", "90 d"), ("model", "Model"), ("trend", "28d vs 91d trend")], dict(f7=nf, f30=nf, lo30=nf, hi30=nf, f90=nf, trend=lambda v: pc(v)))
    Mv = X["M"].values; tot = Mv.sum(1); dts_ = X["dates"]; wd = pd.Series(tot).groupby(dts_.dayofweek).mean(); wd = wd / wd.mean(); mo = pd.Series(tot).groupby(dts_.month).mean(); mo = mo / mo.mean(); pv = X["Pv"]; ups = []
    for j in range(Mv.shape[1]):
        if pv[:, j].sum() >= 5 and (pv[:, j] == 0).sum() > 5 and Mv[pv[:, j] == 0, j].mean() > 0: ups.append(Mv[pv[:, j] == 1, j].mean() / Mv[pv[:, j] == 0, j].mean())
    h += note("<b>Patterns (all SKUs):</b> weekend index %s, strongest month %s (index %s), weakest month %s (index %s); promo days sell %s× non-promo days for %d SKUs with promo history. Trend and seasonality are only reliable with 12+ months of history." % (fx(wd.loc[[5, 6]].mean(), 2), mo.idxmax(), fx(mo.max(), 2), mo.idxmin(), fx(mo.min(), 2), fx(np.median(ups), 2) if ups else "—", len(ups)), "ai")
    f1 = empty("Held-out evaluation unavailable (history too short)")
    if fc["pred"] is not None: pr = fc["pred"][fc["pred"]["sku_i"].isin(sel)].groupby("i")[["y", "p"]].sum() * 30; f1 = go.Figure([go.Scatter(x=[X["dates"][i] for i in pr.index], y=pr["y"], name="Actual 30-day demand", line=dict(color=C["fg"], width=3)), go.Scatter(x=[X["dates"][i] for i in pr.index], y=pr["p"], name="Predicted (%s)" % fc["model"][30], line=dict(color=C["am"], width=3, dash="dot"))]); f1 = sty(f1, 340, "Actual vs predicted 30-day demand on the held-out period (origin date)")
    ff = [fc["tot"][k][sel].sum() for k in (7, 30, 90)]; lo = [fc["lo"][k][sel].sum() for k in (7, 30, 90)]; hi = [fc["hi"][k][sel].sum() for k in (7, 30, 90)]; f2 = go.Figure(go.Bar(x=["7 days", "30 days", "90 days"], y=ff, marker_color=C["fg"], error_y=dict(type="data", symmetric=False, array=[b - a for a, b in zip(ff, hi)], arrayminus=[a - b for a, b in zip(ff, lo)]))); f2 = sty(f2, 320, "Forecast demand by horizon (selection; bars = approximate interval)")
    f3 = sty(go.Figure(go.Bar(x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], y=wd.values, marker_color=C["fg"])), 300, "Weekday demand index (1 = average)"); return h, f1, f2, f3
@safe(3)
def render_health(c, s, st):
    X = G["X"]; P = flt(X["P"], c, s, st).sort_values(["urg", "impact"], ascending=[True, False]); h = banner() + note("<b>Definitions.</b> Available = on hand − reserved (≥ 0). Inventory position = on hand − reserved − backorders + open POs (overdue POs count but are flagged). Days of supply = available ÷ forecast daily demand. <b>Low sales</b> means low velocity; <b>low turnover</b> means stock is large relative to sales; <b>excess</b> means stock beyond the cover limit; <b>obsolescence</b> needs no sales and no recent receipts for the configured period plus business confirmation.", "info")
    h += tbl(P.head(120), [("sku", "SKU"), ("pname", "Product"), ("state", "State"), ("on_hand", "On hand"), ("reserved", "Reserved"), ("avail", "Available"), ("po_qty", "Incoming POs"), ("pos", "Position"), ("rate", "Demand/day"), ("ltd", "Lead time d"), ("cover", "Days supply"), ("inv_value", "Value"), ("trend", "28d trend"), ("abc", "ABC"), ("xyz", "XYZ"), ("expl", "Explanation")],
                dict(state=pill, on_hand=nf, reserved=nf, avail=nf, po_qty=nf, pos=nf, rate=fx, ltd=lambda v: fx(v, 0), cover=lambda v: fx(v, 0), inv_value=money, trend=pc))
    cnt = P["state"].value_counts(); f1 = sty(go.Figure(go.Bar(x=[GL[x] + " " + x for x in ST], y=[int(cnt.get(x, 0)) for x in ST], marker_color=[STC[x] for x in ST])), 320, "SKUs by state"); f1.update_xaxes(tickangle=-25)
    mx = pd.crosstab(P["abc"], P["xyz"]).reindex(index=["A", "B", "C"], columns=["X", "Y", "Z", "n/a"]).fillna(0); f2 = go.Figure(go.Heatmap(z=mx.values, x=list(mx.columns), y=list(mx.index), colorscale=[[0, C["iv"]], [1, C["fg"]]], text=mx.values.astype(int), texttemplate="%{text}")); f2 = sty(f2, 320, "ABC-XYZ matrix (count of SKUs; A = top 80% of annual cost-value, X = stable demand)"); return h, f1, f2
@safe(3)
def render_rep(c, s, st):
    X = G["X"]; S = X["S"]; P = flt(X["P"], c, s, st); sc = P[(P["qty"] > 0) | (P["urg"] <= 1)].sort_values(["urg", "impact"], ascending=[True, False]); z = zval(S["sl"])
    h = banner() + note("<b>Policy.</b> Safety stock = z × √(L·σd² + d²·σL²) with z = %.2f (service level %s), σd = daily demand std (last 90 days), σL = supplier lead-time std from delivery history (0 if unknown, shown in the lead-time source). Reorder point = demand during lead time + safety stock. Target = reorder point + %d review days of demand. Order when position ≤ reorder point: quantity = target − position, rounded up to pack size and MOQ, never negative, and open POs are already inside the position (no duplicate orders). Forecast rate = 30-day model forecast. Budget (if set) is allocated by urgency then impact (greedy heuristic, not an optimiser)." % (z, pc(S["sl"]), S["review"]), "info")
    h += "<div class='kg'>" + card("Orders recommended", nf((P["qty"] > 0).sum()), "SKUs with a quantity") + card("Order value (unconstrained)", money(P["order_value"].sum()), "at unit cost") + card("Funded value", money(P["funded_value"].sum()), "after budget" if S["budget"] > 0 else "no budget limit set") + card("Unfunded SKUs", nf(P["fstat"].astype(str).str.startswith("Unfunded").sum()), "budget-limited", C["co"]) + "</div>"
    h += tbl(sc.head(100), [("sku", "SKU"), ("pname", "Product"), ("state", "State"), ("rate", "Demand/day"), ("ltd", "Lead time d"), ("ss", "Safety stock"), ("rop", "Reorder pt"), ("tgt", "Target"), ("pos", "Position"), ("qty", "Order qty"), ("fq", "Funded qty"), ("fstat", "Budget"), ("order_value", "Order value"), ("arrival", "Expected arrival"), ("conf", "Confidence"), ("lt_src", "Lead-time source"), ("expl", "Reason")],
                dict(state=pill, rate=fx, ltd=lambda v: fx(v, 0), ss=nf, rop=nf, tgt=nf, pos=nf, qty=nf, fq=nf, order_value=money, arrival=dts))
    g = P[P["qty"] > 0].groupby("sup")["order_value"].sum(); f1 = sty(go.Figure(go.Bar(x=[X["supname"].get(i, i) for i in g.index], y=g.values * S["fxr"], marker_color=C["fg"])), 320, "Recommended order value by supplier (%s)" % S["sym"]); f1.update_xaxes(tickangle=-20)
    d = P[(P["rate"] > 0) & np.isfinite(P["cover"])]; f2 = go.Figure(go.Scatter(x=d["ltd"], y=d["cover"], mode="markers+text", text=d["sku"].where(d["urg"] <= 1, ""), textposition="top center", marker=dict(size=9, color=[STC[x] for x in d["state"]], line=dict(width=1, color=C["ch"])))); f2.add_shape(type="line", x0=0, y0=0, x1=max(d["ltd"].max(), 1), y1=max(d["ltd"].max(), 1), line=dict(color=C["gg"], dash="dash")); f2 = sty(f2, 340, "Days of supply vs lead time (below the dashed line = stock shorter than lead time)"); return h, f1, f2
@safe(3)
def render_exc(c, s, st):
    X = G["X"]; S = X["S"]; P = flt(X["P"], c, s, st); e = P[(P["excess_value"] > 0) | P["state"].isin([ST[3], ST[4], ST[5]])].sort_values("excess_value", ascending=False)
    h = banner() + note("<b>Thresholds (editable in Settings):</b> excess if stock covers more than %d days; slow-moving if forecast velocity is below %s units/day; potentially obsolete only if there are no sales for %d days <b>and</b> no receipt for %d days. <b>Low sales</b> = low velocity. <b>Low turnover</b> = stock large vs sales. <b>Excess</b> = stock above the cover limit. <b>Obsolescence</b> = likely no future demand; it cannot be proven from sales alone." % (S["excess"], S["slow"], S["obs"], S["obs"]), "info")
    h += "<div class='kg'>" + card("Excess value", money(e["excess_value"].sum()), "%s units" % nf(e["excess_units"].sum()), C["am"]) + card("Annual carrying cost of excess", money(e["excess_value"].sum() * S["hold"] / 100), "at %.0f%% (assumption)" % S["hold"], C["am"]) + card("SKUs flagged", nf(len(e)), "slow, overstocked or obsolete") + "</div>"
    def act(r):
        a = []
        if r["po_qty"] > 0: a.append("review and reduce or cancel the open PO of %s units" % nf(r["po_qty"]))
        if r["state"] == ST[3]: a.append("stop reordering; verify discontinuation; evaluate transfer or markdown")
        else: a.append("lower the replenishment target; consider transfer to a store with demand, or a markdown if cover stays above the limit")
        if not nul(r.get("shelf")) and r["dos"] > r["shelf"]: a.append("stock may outlast its shelf life: prioritise clearance")
        return "; ".join(a)
    e = e.assign(shelf=X["D"]["sku"].set_index("sku")["shelf_life_days"].reindex(e["sku"]).values); e = e.assign(sug=[act(r) for r in e.to_dict("records")])
    h += tbl(e.head(80), [("sku", "SKU"), ("pname", "Product"), ("state", "State"), ("avail", "Available"), ("rate", "Demand/day"), ("dos", "Days of supply"), ("last_sale", "Days since sale"), ("trend", "28d trend"), ("excess_units", "Excess units"), ("excess_value", "Excess value"), ("sug", "Potential actions")], dict(state=pill, avail=nf, rate=lambda v: fx(v, 2), dos=lambda v: fx(v, 0), last_sale=nf, trend=pc, excess_units=nf, excess_value=money))
    g = e.groupby("cat")["excess_value"].sum().sort_values(); f1 = sty(go.Figure(go.Bar(y=g.index, x=g.values * S["fxr"], orientation="h", marker_color=C["am"])), 320, "Excess value by category (%s)" % S["sym"]); d = P[(P["rate"] > 0)]; f2 = go.Figure(go.Scatter(x=d["rate"], y=np.minimum(d["dos"], 730), mode="markers", text=d["sku"], marker=dict(size=9, color=[STC[x] for x in d["state"]], line=dict(width=1, color=C["ch"])))); f2.add_hline(y=S["excess"], line_dash="dash", line_color=C["gg"]); f2 = sty(f2, 340, "Demand velocity vs days of supply (log x; dashed = excess limit)"); f2.update_xaxes(type="log", title="Units per day"); f2.update_yaxes(title="Days of supply (capped 730)"); return h, f1, f2
@safe(3)
def render_sup(c, s, st):
    X = G["X"]; S = X["S"]; P = flt(X["P"], c, s, st); ss = X["ss"]; tot = max(P["inv_value"].sum(), 1); g = P.groupby("sup").agg(skus=("sku", "size"), inv=("inv_value", "sum"), crit=("urg", lambda v: int((v <= 1).sum())), lt_stated=("lt_stated", "mean")).reset_index(); g["share"] = g["inv"] / tot; g["name"] = g["sup"].map(X["supname"]).fillna(g["sup"])
    g = g.merge(ss.rename(columns={"supplier_id": "sup"}), on="sup", how="left") if len(ss) else g.assign(n=np.nan, mean=np.nan, sd=np.nan, p90=np.nan, late=np.nan); g["cvlt"] = g["sd"] / g["mean"]; g["exposed"] = (g["mean"].fillna(g["lt_stated"]) >= 21) | (g["late"].fillna(0) >= .3) | (g["cvlt"].fillna(0) >= .3); g["bn"] = g["crit"] * g["mean"].fillna(g["lt_stated"])
    h = banner() + (note("Supplier history is available: reliability metrics below come from received purchase orders. Alternative-supplier comparison is not available (no alternate supplier data).", "info") if len(ss) else note("No supplier delivery history: stated lead times are used and supplier reliability cannot yet be evaluated.", "warn"))
    h += tbl(g.sort_values("bn", ascending=False), [("name", "Supplier"), ("skus", "SKUs"), ("share", "Share of inventory value"), ("n", "Deliveries"), ("mean", "Mean lead time d"), ("sd", "Lead-time SD"), ("p90", "P90 d"), ("late", "Late deliveries"), ("lt_stated", "Mean stated lead d"), ("crit", "SKUs critical/reorder"), ("exposed", "Exposed")], dict(skus=nf, share=pc, n=nf, mean=fx, sd=fx, p90=fx, late=pc, lt_stated=fx, crit=nf, exposed=lambda v: "▲ yes" if v else "no"))
    h += note("<b>Exposed</b> = mean lead time ≥ 21 days, or late deliveries ≥ 30%, or lead-time variability (SD ÷ mean) ≥ 0.3. <b>Procurement bottleneck</b> ranking = number of critical or reorder-now SKUs × mean lead time.", "info")
    f1 = empty("No delivery history")
    if len(ss): d = g.dropna(subset=["mean"]); f1 = go.Figure(go.Bar(x=d["name"], y=d["mean"], marker_color=C["fg"], error_y=dict(type="data", array=d["sd"].fillna(0)))); f1 = sty(f1, 340, "Mean supplier lead time ± 1 SD (days)"); f1.update_xaxes(tickangle=-20)
    f2 = sty(go.Figure(go.Bar(x=g["name"], y=g["share"] * 100, marker_color=C["li"])), 320, "Supplier concentration: % of inventory value"); f2.update_xaxes(tickangle=-20); return h, f1, f2
@safe(3)
def render_scn(c, s, st, dm, seas, ltc, sl, pol, ssd, bud, hold, oc, H_):
    X = G["X"]; S = X["S"]; B = X["B"]; ids = set(flt(X["P"], c, s, st)["sku"]); Bs = B[B["sku"].isin(ids)].reset_index(drop=True)
    if Bs.empty: raise ValueError("No SKUs match the filters.")
    base = dict(sl=S["sl"]); scn = dict(dm=1 + dm / 100, seas=seas, ltm=1 + ltc / 100, sl=sl / 100, pol=pol, ssd=ssd); world0 = dict(dm=1.0, seas=0.0, ltm=1.0); worlds = dict(dm=scn["dm"], seas=seas, ltm=scn["ltm"]); H_ = int(H_); b0 = S["budget"]
    A = simulate(Bs, S, base, world0, H_, b0, hold, oc); Bb = simulate(Bs, S, base, worlds, H_, b0, hold, oc); Cc = simulate(Bs, S, scn, worlds, H_, bud, hold, oc); P0 = plan(Bs, dict(S, budget=b0), base); P1 = plan(Bs, S, dict(scn, budget=bud))
    G["scn"] = pd.DataFrame([dict(case=n, **{k: v for k, v in r.items() if k != "cons"}) for n, r in (("Baseline policy, baseline world", A), ("Baseline policy, scenario world (no adaptation)", Bb), ("Scenario policy, scenario world", Cc))])
    rows = [("Stockout-risk SKUs today (plan)", nf((P0["state"] == ST[0]).sum()), nf((P1["state"] == ST[0]).sum()), ""), ("Excess inventory value today", money(P0["excess_value"].sum()), money(P1["excess_value"].sum()), ""), ("Order investment today (funded)", money(P0["funded_value"].sum()), money(P1["funded_value"].sum()), "")]
    t = pd.DataFrame(rows, columns=["m", "a", "b", "c"]); h = banner() + note("<b>Hypothetical simulation, not a forecast or a guaranteed saving.</b> %d-day day-by-day simulation: stock depletes with demand, purchase orders arrive after the simulated lead time, and orders follow the (reorder point, order-up-to target) policy. All three cases use the <b>same random demand and lead-time draws</b>, so differences come only from policy and assumptions. Lost sales are not back-ordered. Open POs arrive in one lump (overdue POs after one lead time)." % H_, "warn")
    h += H("Plan before vs after (today's recommendations)") + tbl(t, [("m", "Measure"), ("a", "Baseline settings"), ("b", "Scenario")], dict(a=str, b=str))
    rr = G["scn"]; h += H("Simulated outcomes over the horizon") + tbl(rr, [("case", "Case"), ("fill", "Projected service level (fill rate)"), ("so_skus", "SKUs with stock-outs"), ("lost_rev", "Lost revenue"), ("avg_inv", "Avg inventory value"), ("end_inv", "Ending inventory"), ("excess_end", "Excess at end"), ("spend", "Procurement spend"), ("norders", "Orders"), ("hold", "Holding cost"), ("order_cost", "Ordering cost"), ("lost_margin", "Lost margin"), ("total", "Total cost (hold + order + lost margin)")], dict(fill=lambda v: pc(v, 1), so_skus=nf, lost_rev=money, avg_inv=money, end_inv=money, excess_end=money, spend=money, norders=nf, hold=money, order_cost=money, lost_margin=money, total=money))
    h += note("Reading it: row 2 shows the exposure if you do nothing about the scenario; row 3 shows the effect of adapting the policy (service level, safety stock, budget). Costs use your holding (%.0f%%/yr) and ordering (%s/order) assumptions. Simulation consistency check: %s." % (hold, money(oc), "stock and order flows balance" if all((A["cons"], Bb["cons"], Cc["cons"])) else "FAILED"), "info")
    nm_ = ["Baseline / baseline", "Baseline / scenario", "Scenario / scenario"]; f1 = go.Figure(go.Bar(x=nm_, y=[A["fill"] * 100, Bb["fill"] * 100, Cc["fill"] * 100], marker_color=[C["gg"], C["co"], C["fg"]])); f1 = sty(f1, 320, "Projected service level (fill rate %)"); f1.update_yaxes(range=[0, 100])
    f2 = go.Figure([go.Bar(x=nm_, y=[r[k] * S["fxr"] for r in (A, Bb, Cc)], name=n, marker_color=cl) for k, n, cl in (("hold", "Holding", C["fg"]), ("order_cost", "Ordering", C["li"]), ("lost_margin", "Lost margin", C["co"]))]); f2.update_layout(barmode="stack"); f2 = sty(f2, 340, "Cost components (%s)" % S["sym"]); return h, f1, f2
@safe(2)
def render_dq():
    X = G["X"]; D = X["D"]; h = banner() + note("Rows read: " + ", ".join("%s %s" % (TN[k], nf(v)) for k, v in D["rows_in"].items()) + ". Raw uploads stay unchanged; a validated copy is analysed. Excluded rows: <b>%d</b> (listed; nothing is dropped silently)." % sum(c for _, _, c in D["excluded"]), "ok")
    h += "".join(note(esc(m), "err" if l == "error" else "warn" if l == "warn" else "info") for l, m in X["issues"]) or note("No validation issues.", "ok")
    if D["excluded"]: h += H("Excluded rows by reason") + tbl(pd.DataFrame(D["excluded"], columns=["t", "w", "c"]), [("t", "Table"), ("w", "Reason"), ("c", "Rows")], dict(c=nf))
    return h, H("Self-test results (run at start-up)") + "".join(note(("PASS " if ok else "FAIL ") + esc(g + ": " + n) + " — " + esc(d), "ok" if ok else "err") for g, n, ok, d in TESTS)

# ------------------------------ strategist, exports ------------------------------
def ask(q, _=None):
    if "X" not in G: return "Load data first."
    X = G["X"]; P = X["P"]; S = X["S"]; q = (q or "").strip() or "What should we do first?"; ql = q.lower(); parts = []; ids = re.findall(r"\bSKU\d{3}\b", q, re.I)
    if ids and ids[0].upper() in set(P["sku"]): parts.append(("sku_explanation", P[P["sku"] == ids[0].upper()]["expl"].iloc[0]))
    if re.search(r"first|urgent|stockout|stock-out|run out|reorder|priority", ql): t = P[P["urg"] <= 1].sort_values(["urg", "impact"], ascending=[True, False]).head(5); parts.append(("urgent_actions", " | ".join("%s: %s (%s)" % (r.sku, r.action, r.state) for r in t.itertuples()) or "no SKU is at stockout or reorder risk"))
    if re.search(r"excess|slow|overstock|obsolete|markdown", ql): t = P.sort_values("excess_value", ascending=False).head(5); parts.append(("excess_stock", "Excess value %s. Largest: " % money(P["excess_value"].sum()) + "; ".join("%s %s (%s days cover)" % (r.sku, money(r.excess_value), fx(r.dos, 0)) for r in t.itertuples())))
    if re.search(r"supplier|lead", ql): ss = X["ss"]; parts.append(("suppliers", ("Slowest by observed mean lead time: " + "; ".join("%s %s d (late %s)" % (X["supname"].get(r.supplier_id, r.supplier_id), fx(r.mean), pc(r.late)) for r in ss.sort_values("mean", ascending=False).head(3).itertuples())) if len(ss) else "no delivery history: reliability cannot be evaluated"))
    if re.search(r"forecast|accuracy|model|demand", ql): parts.append(("forecast", "30-day model: %s; held-out WAPE %s. %s" % (X["fc"]["model"][30], pc(X["fc"]["wape"].get(30), 1), " ".join(X["fc"]["msg"]) or "Forecasts are estimates, not guarantees.")))
    if re.search(r"budget|spend|invest|cost", ql): parts.append(("investment", "Recommended orders: %s funded of %s unconstrained (budget %s). Inventory carrying-cost exposure %s per year at %.0f%% (assumption)." % (money(P["funded_value"].sum()), money(P["order_value"].sum()), money(S["budget"]) if S["budget"] > 0 else "none", money(P["inv_value"].sum() * S["hold"] / 100), S["hold"])))
    if re.search(r"limit|caveat|trust", ql): parts.append(("limitations", "Forecasts and simulations are estimates; synthetic demo data; lost sales not back-ordered; demand is not corrected for past stock-outs; supplier reliability needs delivery history; decisions need buyer review."))
    if not parts: c = P["state"].value_counts(); parts = [("summary", "%d SKUs: %s. Inventory %s; excess %s; recommended orders %s." % (len(P), ", ".join("%s %d" % (k, v) for k, v in c.items()), money(P["inv_value"].sum()), money(P["excess_value"].sum()), money(P["funded_value"].sum())))]
    return "### Answer (deterministic, computed from the loaded data)\n" + "\n".join("- **%s**: %s" % (n, t) for n, t in parts) + "\n\n*Every figure is calculated from the active dataset; scenarios are hypothetical; buyers must review before ordering.*"
def export_all():
    if "X" not in G: return None, note("Load data first.", "info")
    X = G["X"]; P = X["P"]; out = []
    def w(df, fn):
        df.to_csv(fn, index=False)
        if os.path.exists(fn): out.append(fn)
    fc = X["fc"]; tab = pd.DataFrame(dict(sku=X["skus"], product=X["B"]["pname"], f7=fc["tot"][7], f30=fc["tot"][30], f30_low=fc["lo"][30], f30_high=fc["hi"][30], f90=fc["tot"][90], model_30d=fc["model"][30])); w(tab, "stockwise_forecast.csv")
    cols = ["sku", "pname", "cat", "sup", "state", "rate", "ltd", "ss", "rop", "tgt", "pos", "qty", "fq", "fstat", "order_value", "arrival", "conf", "lt_src", "action", "expl"]; w(P[cols], "stockwise_replenishment.csv"); w(P[(P["excess_value"] > 0) | P["state"].isin([ST[3], ST[4], ST[5]])][["sku", "pname", "cat", "state", "avail", "rate", "dos", "excess_units", "excess_value", "action"]], "stockwise_excess_stock.csv")
    w(P[["sku", "pname", "state", "on_hand", "reserved", "avail", "po_qty", "pos", "cover", "inv_value", "abc", "xyz", "expl"]], "stockwise_inventory_health.csv")
    if "scn" in G: w(G["scn"], "stockwise_scenario_comparison.csv")
    w(pd.DataFrame(X["issues"], columns=["level", "message"]), "stockwise_data_quality.csv"); c = P["state"].value_counts(); top = P[P["urg"] <= 1].sort_values(["urg", "impact"], ascending=[True, False]).head(8)
    md = ["# STOCKWISE AI — Executive summary", "", "_Synthetic demo data._" if G.get("demo") else "", "", "- %d SKUs; inventory value %s; states: %s." % (len(P), money(P["inv_value"].sum()), ", ".join("%s %d" % (k, v) for k, v in c.items())), "- Excess stock value %s; carrying cost of excess about %s per year at %.0f%% (assumption)." % (money(P["excess_value"].sum()), money(P["excess_value"].sum() * X["S"]["hold"] / 100), X["S"]["hold"]), "- Recommended orders: %d SKUs, %s funded (budget %s)." % (int((P["fq"] > 0).sum()), money(P["funded_value"].sum()), money(X["S"]["budget"]) if X["S"]["budget"] > 0 else "none"), "- 30-day forecast: %s, held-out WAPE %s." % (X["fc"]["model"][30], pc(X["fc"]["wape"].get(30), 1)), "", "## Priority actions"] + ["- " + r.expl for r in top.itertuples()] + ["", "## Assumptions and limitations", "Forecasts and simulations are estimates. Lost sales are not back-ordered. Review by buyers is required."]
    open("stockwise_executive_summary.md", "w", encoding="utf-8").write("\n".join(md)); out.append("stockwise_executive_summary.md"); return [f for f in out if os.path.exists(f)], note("Generated %d files from the active dataset and settings." % len(out), "ok")

# ------------------------------ self-tests ------------------------------
def miniB(**kw):
    d = dict(sku=["A"], pname=["a"], cat=["c"], sup=["s"], cost=[10.0], price=[15.0], moq=[1.0], pack=[1.0], ltd=[9.0], sdlt=[0.0], lt_src=["observed history"], lt_stated=[9.0], on_hand=[50.0], reserved=[0.0], backorder=[0.0], last_receipt=[pd.NaT], po_qty=[0.0], po_next=[np.inf], po_overdue=[False], rate=[10.0], sd=[2.0], hdays=[700], nz=[600], last_sale=[0], cv=[.3], trend=[0.0], abc=["A"], xyz=["X"], rcpt_age=[10.0]); d.update(kw); return pd.DataFrame(d)
def miniraw():
    dt = pd.date_range("2026-01-01", periods=5); return dict(sku=pd.DataFrame(dict(sku=["A", "B"], unit_cost=[10.0, 20.0])), sales=pd.DataFrame(dict(date=list(dt) * 2, sku=["A"] * 5 + ["B"] * 5, units=[1, 2, 3, 4, 5] * 2)), inv=pd.DataFrame(dict(sku=["A", "B"], on_hand=[10, 20])))
def run_selftests(verbose=True):
    if TESTS: return TESTS
    T = []; saved = dict(G)
    def chk(g, n, fn):
        try: ok, d = fn(); T.append((g, n, bool(ok), d))
        except Exception as e: T.append((g, n, False, type(e).__name__ + ": " + str(e)[:90]))
    def t1(): P = plan(miniB(), S0); return abs(P["ss"][0] - 1.64485 * 6) < .02 and abs(P["rop"][0] - (90 + P["ss"][0])) < 1e-6, "SS = z·σ·√L = %.2f; ROP = 90 + SS = %.2f" % (P["ss"][0], P["rop"][0])
    def t1b(): P = plan(miniB(sdlt=[1.0]), S0); return abs(P["ss"][0] - 1.64485 * math.sqrt(9 * 4 + 100)) < .02, "variable lead time uses √(L·σd² + d²·σL²) = %.2f" % P["ss"][0]
    def t2(): P = plan(miniB(rate=[0.0], sd=[0.0]), S0); return P["qty"][0] == 0 and np.isfinite(P["rop"][0]), "zero demand: ROP %.0f, order 0, no NaN" % P["rop"][0]
    def t3(): P = plan(miniB(ltd=[0.0]), S0); return P["rop"][0] == 0 and P["qty"][0] == 0, "zero lead time: ROP 0, no order, no error"
    def t4(): a = plan(miniB(on_hand=[10.0]), S0); b = plan(miniB(on_hand=[10.0], po_qty=[300.0]), S0); return a["qty"][0] > 0 and b["qty"][0] == 0, "open PO of 300 covers the need: order %s -> %s (no duplicate replenishment)" % (nf(a["qty"][0]), nf(b["qty"][0]))
    def t5(): return rnd(13, 20, 6)[()] == 24 and rnd(25, 20, 6)[()] == 30 and rnd(0, 20, 6)[()] == 0 and rnd(-4, 20, 6)[()] == 0, "13→24 (MOQ 20, pack 6), 25→30, 0→0, negative→0"
    def t6(): P = plan(miniB(on_hand=[-5.0]), S0); return P["avail"][0] == 0 and P["qty"][0] > 0 and P["state"][0] == ST[0], "negative on-hand: available 0, position %s, order %s" % (nf(P["pos"][0]), nf(P["qty"][0]))
    def t7():
        r = miniraw(); r["sku"] = r["sku"].drop(columns=["unit_cost"])
        try: load_all(r); return False, "no error"
        except ValueError as e: return "unit_cost" in str(e), "missing required column reported with guidance"
    def t8():
        r = miniraw(); s = r["sales"]; r["sales"] = pd.concat([s, s.iloc[[0]], pd.DataFrame(dict(date=[pd.Timestamp("2026-01-02")], sku=["ZZZ"], units=[5])), pd.DataFrame(dict(date=[pd.Timestamp("2026-01-09")], sku=[" a "], units=["abc"]))], ignore_index=True); D = load_all(r); w = " | ".join(m for _, m in D["issues"])
        return "duplicate" in w and "not found in the SKU master" in w and "missing or invalid units" in w and len(D["sales"]) == 10, "duplicate row, unknown SKU and malformed units excluded and reported"
    def t9(): r = np.random.default_rng(1); M = r.poisson(5, (60, 3)).astype(float); res = run_forecast(M, np.zeros_like(M), pd.date_range("2026-01-01", periods=60), np.zeros(3, int)); return all(res["model"][h].startswith("Moving average") for h in (7, 30, 90)) and np.isfinite(res["tot"][30]).all() and any("too short" in m for m in res["msg"]), "60 days of history: documented baseline fallback with a disclosure"
    def t10(): r = np.random.default_rng(2); M = r.poisson(8, (800, 6)).astype(float); res = run_forecast(M, np.zeros_like(M), pd.date_range("2024-01-01", periods=800), np.arange(6) % 2); ok = all(res["split"][h][0] <= res["split"][h][1] < res["split"][h][2] for h in (30, 90) if h in res["split"]); return ok and bool(res["perf"]), "latest training target day %s ≤ cut-off %s < first test origin %s (h=30); no random split" % res["split"][30]
    def t11(): a = plan(miniB(), S0); b = plan(miniB(), S0, dict(dm=1.2)); return b["rop"][0] > a["rop"][0] * 1.15, "+20%% demand lifts the reorder point %.0f -> %.0f" % (a["rop"][0], b["rop"][0])
    def t12(): B = miniB(sku=["A", "B", "C"], pname=["a", "b", "c"], cat=["c"] * 3, sup=["s"] * 3, cost=[10.0, 5.0, 8.0], price=[15.0, 8.0, 12.0], moq=[10.0] * 3, pack=[6.0] * 3, ltd=[9.0, 5.0, 12.0], sdlt=[1.0] * 3, lt_src=["observed history"] * 3, lt_stated=[9.0] * 3, on_hand=[50.0, 5.0, 300.0], reserved=[0.0] * 3, backorder=[0.0, 3.0, 0.0], last_receipt=[pd.NaT] * 3, po_qty=[0.0, 40.0, 0.0], po_next=[np.inf, 4.0, np.inf], po_overdue=[False] * 3, rate=[10.0, 6.0, 2.0], sd=[2.0, 1.0, 1.0], hdays=[700] * 3, nz=[600] * 3, last_sale=[0] * 3, cv=[.3] * 3, trend=[0.0] * 3, abc=["A"] * 3, xyz=["X"] * 3, rcpt_age=[10.0] * 3); r = simulate(B, S0, dict(sl=.95), dict(dm=1.2, seas=0.0, ltm=1.2), 60); r2 = simulate(B, S0, dict(sl=.95), dict(dm=1.2, seas=0.0, ltm=1.2), 60); return r["cons"] and r["total"] == r2["total"] and 0 <= r["fill"] <= 1, "stock and order flows balance; same seed gives identical results; fill rate %.1f%%" % (100 * r["fill"])
    def t13():
        X = demo_X(); G.clear(); G.update(X=X, D=X["D"], demo=True, S=dict(S0)); f, _ = export_all(); n = len(pd.read_csv("stockwise_replenishment.csv")); G.clear(); G.update(saved); return n == len(X["P"]) and "stockwise_executive_summary.md" in f, "%d files written; replenishment export has %d rows matching the plan" % (len(f), n)
    def t14(): X = demo_X(); P = X["P"]; c = P["state"].value_counts(); return (c.get(ST[0], 0) >= 1) and (c.get(ST[5], 0) + c.get(ST[4], 0) >= 1) and (P["qty"].fillna(0) >= 0).all() and ((P["qty"].fillna(0) == 0) | (P["qty"].fillna(0) >= P["moq"].fillna(0))).all(), "demo shows both problems: %d stockout-risk SKUs and %d overstocked/slow; all quantities ≥ 0 and respect MOQ" % (c.get(ST[0], 0), c.get(ST[5], 0) + c.get(ST[4], 0))
    def t15(): s2 = dict(G); X = demo_X(); G.clear(); G.update(X=X, D=X["D"], demo=True, S=dict(S0)); a = ask("What should we do first?"); G.clear(); G.update(s2); return "deterministic" in a, "strategist answers from calculated results without any API"
    for g, n, f in [("metrics", "safety stock and reorder point", t1), ("metrics", "variable lead-time safety stock", t1b), ("metrics", "zero demand", t2), ("metrics", "zero lead time", t3), ("replenishment", "open PO prevents duplicate order", t4), ("replenishment", "pack size and MOQ", t5), ("replenishment", "negative inventory", t6), ("validation", "missing column", t7), ("validation", "duplicates, SKU mismatch, malformed values", t8),
                    ("forecasting", "insufficient history fallback", t9), ("forecasting", "time-based split, no leakage", t10), ("scenario", "recalculation", t11), ("simulation", "consistency and reproducibility", t12), ("reporting", "CSV export correctness", t13), ("demo", "both sides of the problem", t14), ("strategist", "works without an API key", t15)]: chk(g, n, f)
    G.clear(); G.update(saved); TESTS[:] = T
    if verbose:
        for g, n, ok, d in T: print(("PASS " if ok else "FAIL ") + g + ": " + n + " :: " + d)
        print("Result: %d of %d self-tests passed." % (sum(1 for t in T if t[2]), len(T)))
    return TESTS
def demo_X(seed=42, S=None):
    S = dict(S or S0); key = ("X", seed, tuple(S.values()))
    if key not in _DEMO: _DEMO[key] = analyze(load_all(get_demo_raw(seed)), S)
    return _DEMO[key]

# ------------------------------ project export ------------------------------
METH = """### Methodology
- **Inventory position** = on hand − reserved − backorders + open purchase orders; **available** = on hand − reserved (≥ 0). Overdue POs count but are flagged.
- **Safety stock** = z × √(L·σd² + d²·σL²); **reorder point** = d·L + SS; **target** = ROP + review days × d; order qty = target − position when position ≤ ROP, rounded to pack size and MOQ. Rate d = model forecast.
- **Forecasting:** 28-day moving average, seasonal naive, HistGradientBoosting (past-only features), chronological split, WAPE-based selection; intervals are empirical held-out error quantiles. ML is disabled when history is too short.
- **States:** critical (cover < next delivery), reorder now, needs data, obsolete (no sales and no receipts for N days), slow-moving, overstocked, healthy. ABC by annual cost-value; XYZ by weekly demand CV.
- **Budget:** greedy allocation by urgency then impact (heuristic, not an LP solver).
- **Simulation:** day-by-day with common random numbers; lost sales are not back-ordered; hypothetical.
- **Not included:** LP optimisation, PDF export, perishable-expiry modelling, transfers between locations as optimised actions.
"""
LICENSE = "MIT License\n\nCopyright (c) 2026 STOCKWISE AI contributors\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n"
README = """# STOCKWISE AI — Retail Inventory Intelligence
The right stock. At the right time.

Connects demand forecasts to explainable replenishment decisions, excess-stock actions and scenario comparisons for a retailer with stock-outs on fast movers and excess on slow movers. Bundled data is **fictional synthetic data** (78 SKUs, 6 categories, 8 suppliers, 24 months of daily sales).

## Architecture
@@@mermaid
flowchart LR
  A[CSV or synthetic data] --> B[Validation + SKU reconciliation]
  B --> C[Demand features + time-aware forecast]
  C --> D[Safety stock + reorder point + order quantity]
  D --> E[State classification + ABC-XYZ]
  D --> F[Day-by-day simulation]
  E --> G[Strategist, exports, executive summary]
  F --> G
@@@

## Run
@@@
pip install -r requirements.txt
python app.py
@@@
In Google Colab paste `app.py` into one cell. The `src/` modules in the brief are consolidated into `app.py`.

## Tests and limits
`pip install pytest && pytest tests` runs the built-in self-tests. See `docs/`. Not included: LP optimiser, PDF, expiry modelling. Estimates are not guarantees; buyers must review. Screenshots: add yours to `screenshots/` (none included).
""".replace("@@@", "`" * 3)
TFILE = "from app import run_selftests\n\n\ndef test_%s():\n    res = [r for r in run_selftests(verbose=False) if r[0] in %s]\n    assert res and all(r[2] for r in res), res\n"
def _src():
    try: return open(__file__, encoding="utf-8").read()
    except Exception: pass
    try:
        for s in list(get_ipython().user_ns.get("In") or [])[::-1]:
            if "STOCKWISE AI — Retail Inventory" in s and "def main" in s: return s
    except Exception: pass
    return None
def write_samples():
    raw = get_demo_raw(42); os.makedirs("sample_data", exist_ok=True); out = []
    for k, n in TN.items(): p = "sample_data/%s.csv" % n; raw[k].to_csv(p, index=False); out.append(p)
    return out
def build_project(files):
    root = "stockwise-ai-inventory-intelligence"
    for d in ["sample_data", "tests", "docs", "outputs", "screenshots"]: os.makedirs(os.path.join(root, d), exist_ok=True)
    src = _src()
    if src: open(root + "/app.py", "w", encoding="utf-8").write(src)
    for f in files: open(os.path.join(root, f), "wb").write(open(f, "rb").read())
    W = {"requirements.txt": "gradio>=4.44,<6\nplotly\nscikit-learn\npandas\nnumpy\n", "README.md": README, "LICENSE": LICENSE, ".gitignore": "__pycache__/\n.ipynb_checkpoints/\n*.zip\nstockwise_*.csv\nstockwise_*.md\n.env\n", "outputs/README.md": "# Outputs\nGenerated exports (stockwise_*.csv, executive summary) are written here when you run the app locally.\n",
         "docs/methodology.md": METH, "docs/data_dictionary.md": "# Data dictionary\nsku_master (sku, product_name, category, supplier_id, unit_cost, unit_price, moq, pack_size, lead_time_days, shelf_life_days); sales_history (date, sku, units, promo); inventory_snapshot (sku, snapshot_date, on_hand, reserved, backorder, last_receipt_date); supplier_lead_times (supplier_id, order_date, promised_date, received_date); open_purchase_orders (po_id, sku, qty, order_date, expected_date). Samples are synthetic; demo values are in INR.\n",
         "docs/assumptions_and_limitations.md": "# Assumptions and limitations\nSynthetic data with planted patterns. Sales are not corrected for past stock-outs. Lost sales are not back-ordered in simulation. Budget allocation is a greedy heuristic. Forecasts, intervals and simulations are estimates, not guarantees. Human review is required before ordering.\n", "screenshots/README.md": "# Screenshots\nAdd your own screenshots here. None are included.\n"}
    for nm, g in [("inventory_metrics", "metrics"), ("forecasting", "forecasting"), ("replenishment", "replenishment"), ("simulation", "simulation"), ("data_validation", "validation")]: W["tests/test_%s.py" % nm] = TFILE % (nm, "('%s',)" % g)
    for fn, tx in W.items(): open(os.path.join(root, fn), "w", encoding="utf-8").write(tx)
    z = "stockwise_ai_project.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for dp, _, fs in os.walk(root):
            for f in fs: zf.write(os.path.join(dp, f))
    return z, bool(src)

# ------------------------------ UI ------------------------------
CSS = """
.gradio-container{background:#F5F2E9!important;color:#263B37!important;font-family:Inter,system-ui,-apple-system,sans-serif!important;max-width:1380px!important}
.hdr{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;padding:24px 30px;border-radius:14px;background:#173C35;color:#F5F2E9;margin-bottom:12px}
.brand{display:flex;gap:16px;align-items:center}.logo{width:52px;height:52px;border-radius:12px;background:#BCE8B0;color:#173C35;font-weight:800;font-size:22px;display:flex;align-items:center;justify-content:center}
.hdr h1{margin:0;font-size:28px;letter-spacing:2px;color:#F5F2E9}.hdr h1 span{color:#BCE8B0}.hdr p{margin:4px 0 0;color:#D6C8F0;font-style:italic}.badges span{display:inline-block;margin:3px;padding:4px 11px;border-radius:6px;background:#ffffff1a;color:#F5F2E9;font-size:12px}
button[role=tab]{background:#173C35!important;color:#F5F2E9!important;border-radius:8px 8px 0 0!important;margin-right:2px}button[role=tab][aria-selected=true]{background:#E8B96A!important;color:#173C35!important;font-weight:700}
.kg{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:10px 0}.kc{background:#fff;border:1px solid #173C3526;border-radius:10px;padding:14px 16px}.kl{color:#5d726c;font-size:11px;text-transform:uppercase;letter-spacing:1.1px}.kv{font-size:24px;font-weight:700;margin:6px 0 4px}.kd{color:#5d726c;font-size:12px}
.note{background:#fff;padding:10px 14px;border-radius:8px;margin:8px 0;color:#263B37;font-size:14px;line-height:1.5;border:1px solid #173C3518}.sec{font-size:15px;font-weight:700;color:#173C35;margin:18px 0 6px;text-transform:uppercase;letter-spacing:1px}
.pill{display:inline-block;border-radius:99px;padding:2px 10px;font-size:11px;font-weight:700;color:#173C35;white-space:nowrap}.tw{overflow-x:auto;border-radius:8px;border:1px solid #173C3526;margin:8px 0;background:#fff}table{border-collapse:collapse;width:100%;font-size:13px}
th{background:#173C35;color:#F5F2E9;text-align:left;padding:9px 10px;font-weight:600}td{padding:8px 10px;border-top:1px solid #173C3512;color:#263B37;vertical-align:top}tr:hover td{background:#BCE8B022}
button.primary{background:#173C35!important;color:#F5F2E9!important;border:none!important;font-weight:700!important}@media(max-width:640px){.hdr h1{font-size:21px}.kv{font-size:20px}}
"""
HEADER = "<div class='hdr'><div class='brand'><div class='logo'>SW</div><div><h1>STOCKWISE <span>AI</span></h1><p>The right stock. At the right time.</p></div></div><div class='badges'><span>Forecasts are estimates</span><span>Scenarios are hypothetical</span><span>Buyer review required</span><span>Works offline</span></div></div>"
SYM = {"₹ INR": "₹", "$ USD": "$", "€ EUR": "€"}
def mkS(sl, rv, ex, sw, ob, hd, oc, bd, sy, fxr):
    return dict(sl=float(sl) / 100, review=int(rv), excess=int(ex), slow=float(sw), obs=int(ob), hold=float(hd), ocost=float(oc), budget=float(bd or 0), sym=SYM.get(sy, "₹"), fxr=float(fxr or 1.0))
def run_all(*a):
    try:
        S = mkS(*a[:10]); files = a[10:15]; mapping = a[15]; seed = int(a[16] or 42); demo = files[0] is None
        X = demo_X(seed, S) if demo and not mapping else analyze(load_all(get_demo_raw(seed) if demo else dict(zip(TN.keys(), [read_csv_any(f) if f is not None else None for f in files])), mapping), S); keep = G.get("scn"); G.clear(); G.update(X=X, D=X["D"], S=S, demo=demo)
        if keep is not None: G["scn"] = keep
        return note(("Fictional demo data analysed. " if demo else "Your data analysed. ") + "%d SKUs, %s sales rows, %d recommended orders, %d stockout-risk SKUs." % (len(X["P"]), nf(len(X["D"]["sales"])), int((X["P"]["qty"].fillna(0) > 0).sum()), int((X["P"]["state"] == ST[0]).sum())), "ok")
    except Exception as e: return note("Could not complete the analysis: " + esc(e), "err")
def refresh():
    if "X" not in G: return tuple(gr.update() for _ in range(4))
    P = G["X"]["P"]; return gr.update(choices=sorted(P["cat"].unique()), value=[]), gr.update(choices=[(G["X"]["supname"].get(x, x), x) for x in sorted(P["sup"].unique())], value=[]), gr.update(choices=ST, value=[]), gr.update(choices=["All SKUs"] + list(P["sku"]), value="All SKUs")
def build_ui(zip_path, sfiles):
    with gr.Blocks(css=CSS, title="STOCKWISE AI", theme=gr.themes.Base(primary_hue="green", neutral_hue="stone")) as demo:
        gr.HTML(HEADER)
        with gr.Accordion("Settings: service level, thresholds, costs, currency (press Recalculate after changing)", open=False):
            with gr.Row():
                s1 = gr.Slider(80, 99.9, 95, step=.1, label="Service-level target (%)"); s2 = gr.Slider(1, 60, 14, step=1, label="Review period (days of cover in target)"); s3 = gr.Slider(30, 365, 120, step=10, label="Excess if cover exceeds (days)"); s4 = gr.Number(1.0, label="Slow-moving if demand below (units/day)")
            with gr.Row():
                s5 = gr.Slider(60, 365, 180, step=10, label="Obsolescence review: no sales and no receipts for (days)"); s6 = gr.Number(22, label="Annual holding cost (% of value)"); s7 = gr.Number(500, label="Ordering cost per order"); s8 = gr.Number(0, label="Procurement budget for today's orders (0 = none)")
            with gr.Row():
                s9 = gr.Dropdown(list(SYM), value="₹ INR", label="Display currency"); s10 = gr.Number(1.0, label="Display rate (display units per ₹; data are in ₹)"); seed = gr.Number(42, label="Demo seed"); run_btn = gr.Button("Recalculate", variant="primary"); rg_btn = gr.Button("Regenerate demo dataset")
        with gr.Accordion("Filters (apply to every page)", open=True):
            with gr.Row():
                f_c = gr.Dropdown([], multiselect=True, label="Category"); f_s = gr.Dropdown([], multiselect=True, label="Supplier"); f_st = gr.Dropdown([], multiselect=True, label="Inventory state"); ap_btn = gr.Button("Apply filters", variant="primary")
        status = gr.HTML(note("Loading the fictional retail dataset, training forecasts and running self-tests (about 1-2 minutes)...", "info")); FIN = [f_c, f_s, f_st]
        with gr.Tabs():
            with gr.Tab("Command Centre"):
                cc_h = gr.HTML()
                with gr.Row():
                    cc1 = gr.Plot(); cc2 = gr.Plot()
                cc3 = gr.Plot()
            with gr.Tab("Demand Intelligence"):
                with gr.Row():
                    d_h = gr.Dropdown(["7", "30", "90"], value="30", label="Horizon (days)"); d_sku = gr.Dropdown(["All SKUs"], value="All SKUs", label="SKU for charts"); d_btn = gr.Button("Update", variant="primary")
                dm_h = gr.HTML()
                with gr.Row():
                    dm1 = gr.Plot(); dm2 = gr.Plot()
                dm3 = gr.Plot()
            with gr.Tab("Inventory Health"):
                hl_h = gr.HTML()
                with gr.Row():
                    hl1 = gr.Plot(); hl2 = gr.Plot()
            with gr.Tab("Replenishment Planner"):
                rp_h = gr.HTML()
                with gr.Row():
                    rp1 = gr.Plot(); rp2 = gr.Plot()
            with gr.Tab("Excess & Slow-Moving"):
                ex_h = gr.HTML()
                with gr.Row():
                    ex1 = gr.Plot(); ex2 = gr.Plot()
            with gr.Tab("Supplier Intelligence"):
                su_h = gr.HTML()
                with gr.Row():
                    su1 = gr.Plot(); su2 = gr.Plot()
            with gr.Tab("Scenario Lab"):
                gr.Markdown("Change assumptions and run. Results are **hypothetical simulations** (same random draws for every policy).")
                with gr.Row():
                    c1 = gr.Slider(-50, 100, 0, step=5, label="Demand change (%)"); c2 = gr.Slider(-30, 100, 0, step=5, label="Seasonal uplift (%)"); c3 = gr.Slider(-30, 100, 0, step=5, label="Supplier lead-time change (%)"); c4 = gr.Slider(80, 99.9, 95, step=.1, label="Service-level target (%)")
                with gr.Row():
                    c5 = gr.Dropdown(["Statistical (z-score)", "Fixed days of cover", "No safety stock"], value="Statistical (z-score)", label="Safety-stock policy"); c6 = gr.Number(7, label="Fixed safety days (if fixed policy)"); c7 = gr.Number(0, label="Budget / procurement limit (0 = none)"); c8 = gr.Number(22, label="Holding cost %/yr"); c9 = gr.Number(500, label="Ordering cost per order"); c10 = gr.Slider(30, 180, 90, step=10, label="Simulation horizon (days)")
                sc_btn = gr.Button("Run scenario", variant="primary"); sc_h = gr.HTML()
                with gr.Row():
                    sc1 = gr.Plot(); sc2 = gr.Plot()
            with gr.Tab("Reports & Data"):
                gr.Markdown("Upload CSVs (SKU master, sales history and inventory snapshot are required; supplier delivery history and open purchase orders are optional). Leave empty for the fictional demo. Raw uploads are never modified.")
                up = []
                with gr.Row():
                    for k in ("sku", "sales", "inv", "sup", "po"): up.append(gr.File(label=TN[k] + ".csv" + ("" if k in ("sku", "sales", "inv") else " (optional)"), file_types=[".csv"]))
                mp = gr.Textbox(label="Column mapping (only if auto-detect fails)", placeholder="units=qty_sold, on_hand=stock_qty", lines=2); ld_btn = gr.Button("Validate and load uploaded data", variant="primary")
                gr.File(value=sfiles, label="Sample CSV datasets (synthetic)", file_count="multiple", interactive=False); dq_h = gr.HTML(); t_h = gr.HTML()
                gr.Markdown("### Exports"); xp_btn = gr.Button("Generate exports and executive summary", variant="primary"); xp_msg = gr.HTML(); xp_files = gr.File(label="Exports", file_count="multiple", interactive=False); gr.File(value=zip_path, label="Download GitHub-ready project (ZIP)", interactive=False)
            with gr.Tab("Methodology"):
                gr.Markdown(METH)
            with gr.Tab("Strategist"):
                gr.Markdown("Answers are computed from the loaded data. Try: *What should we do first?* · *Why is SKU001 a problem?* · *Which stock is excess?* · *Which suppliers are slow?* · *How accurate is the forecast?* · *How much budget do we need?*")
                ag_q = gr.Textbox(label="Ask a question"); ag_btn = gr.Button("Ask", variant="primary"); ag_out = gr.Markdown()
        RUN = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] + up + [mp, seed]; SCI = FIN + [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]
        def pages(e):
            e = e.then(render_cc, FIN, [cc_h, cc1, cc2, cc3]).then(render_dem, FIN + [d_h, d_sku], [dm_h, dm1, dm2, dm3]).then(render_health, FIN, [hl_h, hl1, hl2]).then(render_rep, FIN, [rp_h, rp1, rp2]).then(render_exc, FIN, [ex_h, ex1, ex2])
            return e.then(render_sup, FIN, [su_h, su1, su2]).then(render_scn, SCI, [sc_h, sc1, sc2]).then(render_dq, None, [dq_h, t_h])
        def full(ev): return pages(ev.then(refresh, None, [f_c, f_s, f_st, d_sku]))
        full(run_btn.click(run_all, RUN, [status], show_progress="full")); full(ld_btn.click(run_all, RUN, [status], show_progress="full")); full(demo.load(run_all, RUN, [status])); full(rg_btn.click(lambda s: (s or 42) + 1, seed, seed).then(run_all, RUN, [status], show_progress="full"))
        ap_btn.click(render_cc, FIN, [cc_h, cc1, cc2, cc3]).then(render_dem, FIN + [d_h, d_sku], [dm_h, dm1, dm2, dm3]).then(render_health, FIN, [hl_h, hl1, hl2]).then(render_rep, FIN, [rp_h, rp1, rp2]).then(render_exc, FIN, [ex_h, ex1, ex2]).then(render_sup, FIN, [su_h, su1, su2])
        d_btn.click(render_dem, FIN + [d_h, d_sku], [dm_h, dm1, dm2, dm3]); sc_btn.click(render_scn, SCI, [sc_h, sc1, sc2], show_progress="full"); xp_btn.click(export_all, None, [xp_files, xp_msg], show_progress="full"); ag_btn.click(ask, ag_q, ag_out); ag_q.submit(ask, ag_q, ag_out)
    return demo
def main():
    sf = write_samples(); zp, saved = build_project(sf)
    print("=" * 78); print("STOCKWISE AI starting. Demo data is FICTIONAL. No API key needed."); print("Building the demo analysis and running self-tests (about 1-2 minutes):")
    demo_X(); run_selftests(verbose=True); demo = build_ui(zp, sf)
    print("Project ZIP:", zp, "| standalone app.py included:", "yes" if saved else "NO (create app.py yourself by pasting this cell into it)")
    print("Not included / not tested: LP optimiser (greedy budget heuristic used), PDF export, perishable-expiry modelling, uploads of your own CSVs. The public Gradio link is open to anyone: do not load confidential data.")
    print("=" * 78); demo.queue().launch(debug=False, show_error=True)
if __name__ == "__main__" or "google.colab" in sys.modules:
    main()