# -*- coding: utf-8 -*-
"""
Лабораторна робота № 11. Двовимірне перетворення Фур'є та детектування контурів зображень.
Виконав: Яцентюк Євгеній, група КІ-24-1. Варіант 2 (№ 22 у списку: 22 - 20 = 2).

Параметри варіанта 2: D0 = 50 px (ILPF), sigma_f = 35.0 (GLPF), ядро Собеля k = 3,
пороги Кенні [T_low, T_high] = [60, 180], цільовий контур - границі мідних провідників.

Блоки: 1 - тестова сцена; 2 - 2D-ДПФ та спектри; 3 - частотна фільтрація ILPF/GLPF/GHPF;
4 - градієнти Собеля та Шарра; 5 - детектор Кенні з трьома режимами порогів (та режимом варіанта 2).
Запуск: python lab11_fft2d_edges.py
"""
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

N, D0, SIGF, K_SOBEL = 512, 50, 35.0, 3
T_LOW, T_HIGH = 60, 180
MODES = [("Варіант 2: [60, 180]", 60, 180), ("Режим [30, 80]", 30, 80), ("Режим [50, 150]", 50, 150), ("Режим [120, 220]", 120, 220)]
SEED = 11


def make_scene(n=N, seed=SEED):
    """Блок 1. Тестова сцена n x n: концентричні кола, повернутий прямокутник, штрихова решітка, промені, текст
    та одне слабоконтрастне кільце (щоб пороги Кенні давали різні результати)."""
    rng = np.random.default_rng(seed)
    m = np.zeros((n, n), np.uint8)
    for r in range(20, 130, 20):
        cv2.circle(m, (140, 140), r, 255, 3)                                    # концентричні кола
    box = cv2.boxPoints(((380, 140), (150, 80), 25)).astype(np.int32)
    cv2.fillPoly(m, [box], 255)                                                 # повернутий прямокутник
    for x in range(40, 250, 14):
        cv2.rectangle(m, (x, 300), (x + 5, 420), 255, -1)                       # штрихова решітка
    for a in range(0, 91, 15):
        t = np.deg2rad(a)
        cv2.line(m, (300, 480), (int(300 + 110 * np.cos(t)), int(480 - 110 * np.sin(t))), 255, 2)   # промені
    cv2.putText(m, "FFT CANNY", (30, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_8)        # текст
    faint = np.zeros((n, n), np.uint8)
    cv2.circle(faint, (430, 300), 50, 255, 3)                                   # слабоконтрастне кільце (+60 рівнів)
    base = 30.0 + 170.0 * (m > 0) + 60.0 * ((faint > 0) & (m == 0))
    m = np.maximum(m, faint)
    clean = cv2.GaussianBlur(base, (0, 0), 1.2)                                 # розмиття оптики
    noisy = clean + rng.normal(0.0, 4.0, (n, n))                                # гаусів шум
    gray = np.clip(np.round(noisy), 0, 255).astype(np.uint8)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    gt = (m > 0) & ~(cv2.erode(m, cross) > 0)                                   # еталонний контур: внутрішня межа маски
    return gray, clean, m, gt


def freq_grid(n=N):
    u = np.arange(n) - n // 2
    V, U = np.meshgrid(u, u)
    return np.hypot(U, V)


def apply_filter(a, Hc):
    return np.real(np.fft.ifft2(np.fft.ifftshift(Hc) * np.fft.fft2(a)))


def psnr(a, b, peak=255.0):
    return float(10 * np.log10(peak ** 2 / np.mean((a - b) ** 2)))


def evaluate(edge, gt, tol=2.0):
    """Точність/повнота контуру: піксель вірний, якщо лежить не далі tol px від еталонного контуру."""
    e = edge > 0
    dt_gt = cv2.distanceTransform((~gt).astype(np.uint8), cv2.DIST_L2, 5)
    dt_e = cv2.distanceTransform((~e).astype(np.uint8), cv2.DIST_L2, 5)
    tp = int((e & (dt_gt <= tol)).sum())
    prec = tp / max(int(e.sum()), 1)
    rec = float((gt & (dt_e <= tol)).sum() / gt.sum())
    return dict(n=int(e.sum()), tp=tp, fp=int(e.sum()) - tp, prec=prec, rec=rec, f1=2 * prec * rec / max(prec + rec, 1e-12))


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    plt.rcParams['font.family'] = 'DejaVu Sans'

    res = {}
    # ---------------- Блок 1: тестова сцена
    gray, clean, mask, gt = make_scene()
    img = gray.astype(np.float64)
    cv2.imwrite("data/test_edge_scene.png", gray)
    print(f"Блок 1. Сцена {N}x{N}: яскравість [{gray.min()}, {gray.max()}], еталонний контур {int(gt.sum())} px")

    # ---------------- Блок 2: 2D-ДПФ, амплітудний та фазовий спектри
    F = np.fft.fft2(img)
    Fs = np.fft.fftshift(F)
    S = 20 * np.log10(np.abs(Fs) + 1)
    phase = np.angle(Fs)
    parseval = abs((img ** 2).sum() - (np.abs(F) ** 2).sum() / N ** 2) / (img ** 2).sum()
    inv_err = np.abs(np.real(np.fft.ifft2(F)) - img).max()
    dc = np.unravel_index(np.argmax(np.abs(Fs)), Fs.shape)
    print(f"Блок 2. Спектр {S.shape[0]}x{S.shape[1]}, максимум {S.max():.1f} дБ у позиції {tuple(int(i) for i in dc)}; Парсеваль: {parseval:.1e}; обернене ДПФ: {inv_err:.1e}")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    axes[0].imshow(gray, cmap='gray'); axes[0].set_title("Тестове зображення")
    im1 = axes[1].imshow(S, cmap='magma'); axes[1].set_title("Логарифмічний амплітудний спектр, дБ"); plt.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi); axes[2].set_title("Фазовий спектр, рад"); plt.colorbar(im2, ax=axes[2], fraction=0.046)
    for a in axes: a.axis('off')
    plt.tight_layout(); plt.savefig("figures/fft2d_spectrum_analysis.png", dpi=200); plt.show()

    res['spectrum'] = dict(max_db=float(S.max()), max_pos=[int(i) for i in dc], parseval_rel=float(parseval), inverse_err=float(inv_err))

    # ---------------- Блок 3: ILPF, GLPF, GHPF
    D = freq_grid()
    H = {"ILPF": (D <= D0).astype(float), "GLPF": np.exp(-D ** 2 / (2 * SIGF ** 2))}
    H["GHPF"] = 1.0 - H["GLPF"]
    out = {k: apply_filter(img, v) for k, v in H.items()}
    clean_lp = {k: apply_filter(clean, H[k]) for k in ("ILPF", "GLPF")}
    print(f"Блок 3. D0 = {D0}, sigma_f = {SIGF}:")
    for k in ("ILPF", "GLPF"):
        print(f"   {k}: діапазон результату [{out[k].min():.1f}, {out[k].max():.1f}]; PSNR відносно кадру без шуму: {psnr(out[k], clean):.2f} дБ (кадр після фільтра без шуму: {psnr(clean_lp[k], clean):.2f} дБ)")
    print(f"   GHPF: діапазон [{out['GHPF'].min():.1f}, {out['GHPF'].max():.1f}]; GLPF + GHPF = вихідне: похибка {np.abs(out['GLPF'] + out['GHPF'] - img).max():.1e}")
    res['filters'] = {k: dict(min=float(out[k].min()), max=float(out[k].max())) for k in out}
    for k in ('ILPF', 'GLPF'): res['filters'][k]['psnr_vs_noise_free'] = psnr(out[k], clean)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes[0, 0].imshow(gray, cmap='gray'); axes[0, 0].set_title("Вихідне зображення")
    for j, k in enumerate(("ILPF", "GLPF", "GHPF")):
        if k == "GHPF": axes[0, j + 1].imshow(out[k], cmap='gray', vmin=-60, vmax=60)
        else: axes[0, j + 1].imshow(np.clip(out[k], 0, 255), cmap='gray', vmin=0, vmax=255)
        axes[0, j + 1].set_title(f"Результат {k}")
        axes[1, j + 1].imshow(H[k], cmap='gray', vmin=0, vmax=1); axes[1, j + 1].set_title(f"Маска {k}: " + {"ILPF": f"D0 = {D0}", "GLPF": f"sigma_f = {SIGF}", "GHPF": "1 - GLPF"}[k])
    axes[1, 0].imshow(np.log10(np.abs(Fs) + 1), cmap='magma'); axes[1, 0].set_title("Спектр вихідного зображення, lg|F|")
    for a in axes.ravel(): a.axis('off')
    plt.tight_layout(); plt.savefig("figures/frequency_filtering_results.png", dpi=200); plt.show()

    # ---------------- Блок 4: градієнти Собеля та Шарра
    sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=K_SOBEL); sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=K_SOBEL)
    cx = cv2.Scharr(img, cv2.CV_64F, 1, 0); cy = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    mag_s, mag_c = np.hypot(sx, sy), np.hypot(cx, cy)
    ang_s = np.degrees(np.arctan2(sy, sx))
    print(f"Блок 4. Собель (k = {K_SOBEL}): max |G| = {mag_s.max():.1f}, середнє {mag_s.mean():.2f}; Шарр: max |G| = {mag_c.max():.1f}, середнє {mag_c.mean():.2f} (відношення середніх {mag_c.mean() / mag_s.mean():.2f}, коефіцієнти підсилення рампи 32/8 = 4)")
    res['gradients'] = dict(sobel_max=float(mag_s.max()), sobel_mean=float(mag_s.mean()), scharr_max=float(mag_c.max()), scharr_mean=float(mag_c.mean()))
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes[0, 0].imshow(np.abs(sx), cmap='inferno'); axes[0, 0].set_title("Собель: |Gx|")
    axes[0, 1].imshow(np.abs(sy), cmap='inferno'); axes[0, 1].set_title("Собель: |Gy|")
    axes[0, 2].imshow(np.where(mag_s > 120, ang_s, np.nan), cmap='hsv', vmin=-180, vmax=180); axes[0, 2].set_title("Собель: напрям градієнта (де |G| > 120)")
    axes[1, 0].imshow(mag_s, cmap='inferno'); axes[1, 0].set_title("Собель: |G|")
    axes[1, 1].imshow(mag_c, cmap='inferno'); axes[1, 1].set_title("Шарр: |G|")
    axes[1, 2].imshow(mag_c / 4 - mag_s, cmap='coolwarm', vmin=-np.percentile(np.abs(mag_c / 4 - mag_s), 99.9), vmax=np.percentile(np.abs(mag_c / 4 - mag_s), 99.9)); axes[1, 2].set_title("Різниця Шарр/4 - Собель")
    for a in axes.ravel(): a.axis('off')
    plt.tight_layout(); plt.savefig("figures/sobel_scharr_gradients.png", dpi=200); plt.show()

    # ---------------- Блок 5: детектор Кенні, режими порогів
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    res['canny'] = {}
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.4))
    print("Блок 5. Детектор Кенні (допуск 2 px до еталонного контуру):")
    print(f"   {'Режим':<24}{'пікселів':>10}{'TP':>8}{'хибні':>8}{'P':>8}{'recall':>8}{'F1':>8}")
    for ax, (nm, lo, hi) in zip(axes, MODES):
        e = cv2.Canny(blur, lo, hi, apertureSize=K_SOBEL, L2gradient=False)
        r = evaluate(e, gt); res['canny'][nm] = dict(t_low=lo, t_high=hi, **r)
        print(f"   {nm:<24}{r['n']:10d}{r['tp']:8d}{r['fp']:8d}{r['prec']:8.3f}{r['rec']:8.3f}{r['f1']:8.3f}")
        rgb = np.zeros((N, N, 3), np.float32)
        dt_e = cv2.distanceTransform((e == 0).astype(np.uint8), cv2.DIST_L2, 5)
        dt_gt = cv2.distanceTransform((~gt).astype(np.uint8), cv2.DIST_L2, 5)
        rgb[gt & (dt_e > 2)] = (1.0, 0.6, 0.0)
        rgb[(e > 0) & (dt_gt <= 2)] = (0.15, 0.9, 0.3)
        rgb[(e > 0) & (dt_gt > 2)] = (1.0, 0.1, 0.1)
        ax.imshow(rgb); ax.set_title(f"{nm}\n{r['n']} px, P = {r['prec']:.2f}, recall = {r['rec']:.2f}"); ax.axis('off')
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.86, wspace=0.03)
    plt.savefig("figures/canny_thresholds_study.png", dpi=200); plt.show()
    with open("data/results_base_scene.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("Готово: data/test_edge_scene.png, data/results_base_scene.json, figures/*.png")


if __name__ == "__main__":
    main()
