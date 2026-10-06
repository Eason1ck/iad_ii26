import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV", "ALTV",
            "MLTV", "Width", "Min", "Max", "Nmax", "Nzeros", "Mode", "Mean",
            "Median", "Variance", "Tendency"]
TARGET = "NSP"
CLASS_NAMES = ["Normal (1)", "Suspect (2)", "Pathologic (3)"]


def load_ctg(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".xlsx"):
        try:
            raw = pd.read_excel(path, sheet_name="Data", header=None)
        except ImportError as e:
            need = "xlrd" if ext == ".xls" else "openpyxl"
            raise SystemExit(f"Для чтения {ext} нужен пакет {need}. Установите: "
                             f"pip install {need}   (или откройте файл в Excel, "
                             f"сохраните как .xlsx и запустите с --data CTG.xlsx)") from e
        hdr = None
        for i in range(min(len(raw), 10)):
            if TARGET in [str(v).strip() for v in raw.iloc[i].values]:
                hdr = i
                break
        if hdr is None:
            raise ValueError("В листе 'Data' не найдена строка заголовка с колонкой NSP")
        df = raw.iloc[hdr + 1:].copy()
        df.columns = [str(c).strip() for c in raw.iloc[hdr].values]
    else:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]

    dup = df.columns[df.columns.duplicated()].tolist()
    if dup:
        print(f"Повторяющиеся колонки {sorted(set(dup))}: оставлено первое вхождение")
        df = df.loc[:, ~df.columns.duplicated()]

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"В файле нет колонок: {missing}. Найдены: {list(df.columns)}")

    df = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df.dropna(subset=FEATURES)
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=int) - 1
    assert set(np.unique(y)) <= {0, 1, 2}, "NSP должен принимать значения 1/2/3"
    return X, y


ACTS = {
    "relu": (lambda z: np.maximum(z, 0.0), lambda a: (a > 0).astype(a.dtype)),
    "tanh": (np.tanh, lambda a: 1.0 - a * a),
    "sigmoid": (lambda z: 1.0 / (1.0 + np.exp(-z)), lambda a: a * (1.0 - a)),
}


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, sizes, act, task, rng):
        self.sizes, self.act, self.task = list(sizes), act, task
        self.f, self.df = ACTS[act]
        self.W, self.b = [], []
        last = len(sizes) - 2
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            if act == "relu" and i < last:
                W = rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_out))
            else:
                lim = np.sqrt(6.0 / (n_in + n_out))
                W = rng.uniform(-lim, lim, (n_in, n_out))
            self.W.append(W)
            self.b.append(np.zeros(n_out))

    def forward(self, X):
        A = [X]
        L = len(self.W)
        for i in range(L):
            Z = A[-1] @ self.W[i] + self.b[i]
            if i < L - 1:
                A.append(self.f(Z))
            else:
                A.append(softmax(Z) if self.task == "clf" else Z)
        return A

    def loss(self, X, Y):
        out = self.forward(X)[-1]
        if self.task == "clf":
            return -np.mean(np.log(out[np.arange(len(X)), Y] + 1e-12))
        return np.mean((out - Y) ** 2)

    def grads(self, X, Y, l2):
        A = self.forward(X)
        n, out = len(X), A[-1]
        if self.task == "clf":
            delta = out.copy()
            delta[np.arange(n), Y] -= 1.0
            delta /= n
        else:
            delta = 2.0 * (out - Y) / (n * out.shape[1])
        L = len(self.W)
        gW, gb = [None] * L, [None] * L
        for i in reversed(range(L)):
            gW[i] = A[i].T @ delta + l2 * self.W[i]
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * self.df(A[i])
        return gW, gb

    def predict(self, X):
        return self.forward(X)[-1].argmax(axis=1)


def fit(net, X, Y, epochs, lr, batch, l2, rng, eval_fn=None):
    params = net.W + net.b
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    n, hist = len(X), []
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            bi = idx[s:s + batch]
            gW, gb = net.grads(X[bi], Y[bi], l2)
            t += 1
            for p, g, mm, vv in zip(params, gW + gb, m, v):
                mm *= b1
                mm += (1 - b1) * g
                vv *= b2
                vv += (1 - b2) * g * g
                p -= lr * (mm / (1 - b1 ** t)) / (np.sqrt(vv / (1 - b2 ** t)) + eps)
        if eval_fn is not None:
            hist.append(eval_fn(net))
    return hist


def pretrain_autoencoders(Xtr, Xva, hidden, act, epochs, lr, batch, l2, noise, seed):
    f = ACTS[act][0]
    Htr, Hva = Xtr, Xva
    enc, curves = [], []
    for k, h in enumerate(hidden):
        rng = np.random.default_rng(seed * 100 + k)
        ae = MLP([Htr.shape[1], h, Htr.shape[1]], act, "ae", rng)
        sd = Htr.std()

        def ev(net, Htr=Htr, Hva=Hva):
            return {"train": net.loss(Htr, Htr), "val": net.loss(Hva, Hva)}

        if noise > 0:
            hist = []
            for _ in range(epochs):
                Xn = Htr + rng.normal(0.0, noise * sd, Htr.shape)
                fit(ae, Xn, Htr, 1, lr, batch, l2, rng)
                hist.append(ev(ae))
        else:
            hist = fit(ae, Htr, Htr, epochs, lr, batch, l2, rng, ev)
        enc.append((ae.W[0].copy(), ae.b[0].copy()))
        curves.append(hist)
        Htr = f(Htr @ ae.W[0] + ae.b[0])
        Hva = f(Hva @ ae.W[0] + ae.b[0])
    return enc, curves


def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_per_class": f.tolist(),
        "precision_per_class": p.tolist(),
        "recall_per_class": r.tolist(),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def run_seed(X, y, seed, a):
    Xtmp, Xte, ytmp, yte = train_test_split(X, y, test_size=0.20, stratify=y,
                                            random_state=seed)
    Xtr, Xva, ytr, yva = train_test_split(Xtmp, ytmp, test_size=0.15,
                                          stratify=ytmp, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    sizes = [X.shape[1], *a.hidden, 3]

    def make_eval(Xv, yv):
        def ev(net):
            pred = net.predict(Xv)
            return {"val_loss": net.loss(Xv, yv),
                    "val_f1": f1_score(yv, pred, average="macro", zero_division=0),
                    "train_loss": net.loss(Xtr, ytr)}
        return ev

    out = {}
    netA = MLP(sizes, a.act, "clf", np.random.default_rng(seed + 1))
    histA = fit(netA, Xtr, ytr, a.epochs, a.lr, a.batch, a.l2,
                np.random.default_rng(seed + 2), make_eval(Xva, yva))
    out["scratch"] = {"metrics": metrics(yte, netA.predict(Xte)), "hist": histA}

    t0 = time.time()
    enc, ae_curves = pretrain_autoencoders(Xtr, Xva, a.hidden, a.act, a.ae_epochs,
                                           a.ae_lr, a.batch, a.l2, a.noise, seed)
    netB = MLP(sizes, a.act, "clf", np.random.default_rng(seed + 1))
    for i, (W, b) in enumerate(enc):
        netB.W[i], netB.b[i] = W, b
    loss0 = netB.loss(Xva, yva)
    histB = fit(netB, Xtr, ytr, a.epochs, a.lr, a.batch, a.l2,
                np.random.default_rng(seed + 2), make_eval(Xva, yva))
    out["pretrained"] = {"metrics": metrics(yte, netB.predict(Xte)), "hist": histB,
                         "ae_curves": ae_curves, "val_loss_after_transfer": loss0,
                         "pretrain_seconds": time.time() - t0}
    out["split"] = {"train": len(ytr), "val": len(yva), "test": len(yte)}
    return out


C_A, C_B = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
LBL_A, LBL_B = "Без предобучения", "С предобучением (автоэнкодеры)"


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_plots(res, a, outdir):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "text.color": INK})
    r0 = res[0]

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, ttl in zip(axs, ["train_loss", "val_loss", "val_f1"],
                            ["Loss на обучающей выборке", "Loss на валидации",
                             "Macro-F1 на валидации"]):
        for name, col, lbl in (("scratch", C_A, LBL_A), ("pretrained", C_B, LBL_B)):
            ax.plot([h[key] for h in r0[name]["hist"]], color=col, lw=2, label=lbl)
        ax.set_title(ttl, loc="left", fontsize=11)
        ax.set_xlabel("Эпоха дообучения")
        style_ax(ax)
    axs[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_learning_curves.png"), dpi=150)
    plt.close(fig)

    curves = r0["pretrained"]["ae_curves"]
    fig, axs = plt.subplots(1, len(curves), figsize=(3.6 * len(curves), 3.6))
    axs = np.atleast_1d(axs)
    for k, (ax, c) in enumerate(zip(axs, curves)):
        ax.plot([h["train"] for h in c], color=C_B, lw=2, label="train")
        ax.plot([h["val"] for h in c], color=C_B, lw=2, ls="--", label="val")
        ax.set_title(f"Автоэнкодер слоя {k + 1} ({a.hidden[k]} нейр.)",
                     loc="left", fontsize=10)
        ax.set_xlabel("Эпоха предобучения")
        if k == 0:
            ax.set_ylabel("MSE реконструкции")
            ax.legend(frameon=False, fontsize=9)
        style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_autoencoder_reconstruction.png"), dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, name, ttl in zip(axs, ["scratch", "pretrained"], [LBL_A, LBL_B]):
        cm = sum(np.array(r[name]["metrics"]["confusion"]) for r in res)
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{pct[i, j]:.1f}%\n({cm[i, j]})", ha="center",
                        va="center", fontsize=9,
                        color="white" if pct[i, j] > 55 else INK)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES, fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlabel("Предсказано")
        ax.set_ylabel("Истинный класс")
        ax.set_title(ttl, loc="left", fontsize=11)
    fig.suptitle(f"Confusion matrix, сумма по {len(res)} разбиениям (тест)", x=0.01,
                 ha="left", fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_confusion_matrices.png"), dpi=150)
    plt.close(fig)

    fA = [r["scratch"]["metrics"]["macro_f1"] for r in res]
    fB = [r["pretrained"]["metrics"]["macro_f1"] for r in res]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bp = ax.boxplot([fA, fB], widths=0.45, patch_artist=True, showfliers=False)
    for patch, col in zip(bp["boxes"], (C_A, C_B)):
        patch.set(facecolor=col, alpha=0.35, edgecolor=col, lw=1.5)
    for k, c in enumerate((C_A, C_B)):
        bp["medians"][k].set(color=c, lw=2)
    for w in bp["whiskers"] + bp["caps"]:
        w.set(color=MUTED)
    rng = np.random.default_rng(0)
    for k, (v, col) in enumerate(((fA, C_A), (fB, C_B)), start=1):
        ax.scatter(k + rng.uniform(-0.1, 0.1, len(v)), v, s=28, color=col,
                   edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Без предобучения", "С предобучением"])
    ax.set_ylabel("Macro-F1 (тест)")
    ax.set_title("Macro-F1 по разбиениям", loc="left", fontsize=11)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig4_macro_f1_boxplot.png"), dpi=150)
    plt.close(fig)


def summarize(res, a, outdir, n_total, class_counts, n_feat):
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    rows, lines = {}, []
    for name in ("scratch", "pretrained"):
        ms = [r[name]["metrics"] for r in res]
        rows[name] = {k: (np.mean([m[k] for m in ms]), np.std([m[k] for m in ms]))
                      for k in keys}
        f1c = np.array([m["f1_per_class"] for m in ms])
        rows[name]["f1_per_class"] = (f1c.mean(0), f1c.std(0))

    names = {"scratch": "Без предобучения", "pretrained": "С предобучением"}
    lines.append(f"# Результаты: Cardiotocography (NSP), {len(res)} разбиений\n")
    lines.append(f"Объектов: {n_total}; распределение классов (Normal/Suspect/Pathologic): "
                 f"{class_counts}")
    lines.append(f"Архитектура: {[n_feat, *a.hidden, 3]}, активация: {a.act}; "
                 f"fine-tuning: {a.epochs} эпох, Adam lr={a.lr}, batch={a.batch}, L2={a.l2}; "
                 f"предобучение: {a.ae_epochs} эпох/слой, lr={a.ae_lr}, "
                 f"шум (denoising) = {a.noise}\n")
    lines.append("| Метрика | " + " | ".join(names.values()) + " | Δ (с − без) |")
    lines.append("|---|---|---|---|")
    for k in keys:
        mA, sA = rows["scratch"][k]
        mB, sB = rows["pretrained"][k]
        lines.append(f"| {k} | {mA:.4f} ± {sA:.4f} | {mB:.4f} ± {sB:.4f} | {mB - mA:+.4f} |")
    for c, cn in enumerate(CLASS_NAMES):
        mA, sA = rows["scratch"]["f1_per_class"][0][c], rows["scratch"]["f1_per_class"][1][c]
        mB, sB = rows["pretrained"]["f1_per_class"][0][c], rows["pretrained"]["f1_per_class"][1][c]
        lines.append(f"| F1: {cn} | {mA:.4f} ± {sA:.4f} | {mB:.4f} ± {sB:.4f} | {mB - mA:+.4f} |")

    fA = np.array([r["scratch"]["metrics"]["macro_f1"] for r in res])
    fB = np.array([r["pretrained"]["metrics"]["macro_f1"] for r in res])
    wins = int((fB > fA).sum())
    lines.append(f"\nПредобучение лучше по macro-F1 в {wins} из {len(res)} разбиений "
                 f"(ничья: {int((fB == fA).sum())}).")
    p_txt = "н/д"
    try:
        from scipy.stats import wilcoxon
        if np.any(fA != fB):
            p = wilcoxon(fB, fA).pvalue
            p_txt = f"{p:.4f}"
    except Exception:
        pass
    lines.append(f"Критерий Уилкоксона (парный, macro-F1): p = {p_txt}.")
    cmA = sum(np.array(r["scratch"]["metrics"]["confusion"]) for r in res)
    cmB = sum(np.array(r["pretrained"]["metrics"]["confusion"]) for r in res)
    lines.append("\nСуммарные confusion matrix (строки - истинный класс):\n")
    lines.append("Без предобучения:\n```\n" + str(cmA) + "\n```")
    lines.append("С предобучением:\n```\n" + str(cmB) + "\n```")
    ae_last = [np.mean([r["pretrained"]["ae_curves"][k][-1]["val"] for r in res])
               for k in range(len(a.hidden))]
    lines.append("\nСредняя ошибка реконструкции (MSE, val) после предобучения слоёв: "
                 + ", ".join(f"слой {k + 1}: {v:.4f}" for k, v in enumerate(ae_last)))
    text = "\n".join(lines)
    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    slim = [{"scratch": r["scratch"]["metrics"], "pretrained": r["pretrained"]["metrics"]}
            for r in res]
    with open(os.path.join(outdir, "metrics_per_seed.json"), "w", encoding="utf-8") as fh:
        json.dump(slim, fh, ensure_ascii=False, indent=1)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="CTG.xls", help="путь к CTG.xls / .xlsx / .csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--act", default="relu", choices=list(ACTS))
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 48, 32, 16])
    ap.add_argument("--epochs", type=int, default=100, help="эпох дообучения")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--ae-epochs", dest="ae_epochs", type=int, default=60,
                    help="эпох предобучения одного автоэнкодера")
    ap.add_argument("--ae-lr", dest="ae_lr", type=float, default=1e-3)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="доля std входа как шум (denoising AE); 0 - обычный AE")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    X, y = load_ctg(a.data)
    counts = np.bincount(y, minlength=3).tolist()
    print(f"Загружено: {X.shape[0]} объектов, {X.shape[1]} признаков, классы: {counts}")

    res = []
    for s in range(a.seeds):
        t0 = time.time()
        r = run_seed(X, y, s, a)
        res.append(r)
        print(f"seed {s}: macro-F1  без = {r['scratch']['metrics']['macro_f1']:.4f}  "
              f"с = {r['pretrained']['metrics']['macro_f1']:.4f}  ({time.time() - t0:.0f} c)")

    make_plots(res, a, a.out)
    print("\n" + summarize(res, a, a.out, len(y), counts, X.shape[1]))
    print(f"\nФайлы сохранены в: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()