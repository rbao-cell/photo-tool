/* 写真書き出し — ブラウザ版エンジン
 * Python 版 processor.py をそのまま移植。画像はすべてブラウザ内で処理し、外部に送信しない。
 */
(function () {
  'use strict';

  // ------------------------------------------------------------ ルール
  const PRESETS = {
    main:    { slot: 'person',  mode: 'person',  w: 1600, h: 640, head_top: 152, face_ratio: 227 / 640 * 0.968, center: 0.62,
               positions: { left: 0.38, center: 0.5, right: 0.62 }, gray: false },
    wide:    { slot: 'person',  mode: 'person',  w: 910,  h: 400, head_top: 65,  face_ratio: 102 / 400 * 1.023, center: 0.705, positions: null, gray: true },
    top:     { slot: 'person',  mode: 'person',  w: 500,  h: 341, head_top: 20,  face_ratio: 110 / 341 * 1.026, center: 0.5,   positions: null, gray: false },
    profile: { slot: 'profile', mode: 'profile', w: 750,  h: 438, head_top: 20,  face_ratio: null, center: null, positions: null, gray: false, guides: [375 - 218, 375 + 218] },
    text01:  { slot: 'text01',  mode: 'plain',   w: 750,  h: 464, head_top: null, face_ratio: null, center: null, positions: null, gray: false },
    text02:  { slot: 'text02',  mode: 'plain',   w: 750,  h: 464, head_top: null, face_ratio: null, center: null, positions: null, gray: false },
  };
  const ORDER = ['main', 'wide', 'top', 'profile', 'text01', 'text02'];
  const GRAY_GAMMA = 1.35;
  const DET_MAX = 1600;
  const JPEG_OPTS = {
    quality: 94, baseline: false, arithmetic: false, progressive: true, optimize_coding: true,
    smoothing: 0, color_space: 3, quant_table: 0, trellis_multipass: false, trellis_opt_zero: false,
    trellis_opt_table: false, trellis_loops: 1, auto_subsample: false, chroma_subsample: 1,
    separate_chroma_quality: false, chroma_quality: 94,
  };
  const W16_PARTS = 9;   // u2net の重み（float16 で保存し、読み込み時に float32 へ戻す）
  const CACHE_NAME = 'photo-tool-models-v2';

  // ------------------------------------------------------------ ファイル名 → 用途
  function guessSlot(filename) {
    const n = filename.replace(/\.[^.]+$/, '').toLowerCase();
    if (/text\s*0?2/.test(n)) return 'text02';
    if (/text\s*0?1/.test(n)) return 'text01';
    if (n.includes('profile') || n.includes('prof')) return 'profile';
    if (/top|main|wide/.test(n)) return 'person';
    return null;
  }
  function commonPrefix(arr) {
    if (!arr.length) return '';
    let p = arr[0];
    for (const s of arr) { let i = 0; while (i < p.length && i < s.length && p[i] === s[i]) i++; p = p.slice(0, i); }
    return p;
  }
  function jobName(filenames) {
    const stems = filenames.map(f => f.replace(/\.[^.]+$/, ''));
    if (!stems.length) return 'photo';
    let p = stems.length > 1 ? commonPrefix(stems) : stems[0].split(/[_\-\s]/)[0];
    p = p.replace(/[\d_\-\s]+$/, '');
    return p || stems[0];
  }

  // ------------------------------------------------------------ モデル読み込み
  let yunet = null, u2net = null, srgbIcc = null, encodeFn = null, picaInst = null;
  const base = new URL('.', document.baseURI).href;

  async function fetchBytes(url, onChunk) {
    let cache = null;
    try { cache = await caches.open(CACHE_NAME); } catch (e) { /* file:// など */ }
    let res = cache ? await cache.match(url) : null;
    if (!res) {
      res = await fetch(url);
      if (!res.ok) throw new Error(`${url} を読み込めません (HTTP ${res.status})`);
      if (cache) { try { await cache.put(url, res.clone()); } catch (e) { /* 容量不足など */ } }
    }
    const reader = res.body.getReader();
    const chunks = []; let n = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); n += value.length; onChunk && onChunk(value.length);
    }
    const out = new Uint8Array(n); let o = 0;
    for (const c of chunks) { out.set(c, o); o += c.length; }
    return out;
  }

  async function init(onProgress) {
    const ort = window.ort;
    ort.env.wasm.wasmPaths = base + 'vendor/';
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    ort.env.wasm.proxy = true;
    const TOTAL = 87952768 + 240000 + 100000;
    let got = 0;
    const tick = n => { got += n; onProgress && onProgress(Math.min(0.99, got / TOTAL)); };
    const yBytes = await fetchBytes(base + 'models/yunet.onnx', tick);
    srgbIcc = await fetchBytes(base + 'models/srgb.icc');
    const graph = await fetchBytes(base + 'models/u2net.onnx', tick);
    const parts = [];
    for (let i = 0; i < W16_PARTS; i++) parts.push(await fetchBytes(base + `models/w16.part${i}`, tick));
    const total = parts.reduce((a, p) => a + p.length, 0);
    const h = new Uint8Array(total); let o = 0;
    for (const p of parts) { h.set(p, o); o += p.length; }
    const weights = f16ToF32(new Uint16Array(h.buffer, 0, total >> 1));
    onProgress && onProgress(1, 'init');
    const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    yunet = await ort.InferenceSession.create(yBytes, opt);
    u2net = await ort.InferenceSession.create(graph, { ...opt, externalData: [{ path: 'weights.bin', data: new Uint8Array(weights.buffer) }] });
    const mod = await import(base + 'vendor/jsquash/encode.js');
    encodeFn = mod.default;
    picaInst = window.pica ? window.pica({ features: ['js', 'wasm'] }) : null;
  }

  function f16ToF32(src) {
    const out = new Float32Array(src.length), u = new Uint32Array(out.buffer);
    for (let i = 0; i < src.length; i++) {
      const x = src[i], sgn = (x & 0x8000) << 16, e = (x >> 10) & 0x1f, m = x & 0x3ff;
      if (e === 0) { out[i] = (sgn ? -1 : 1) * m * 5.960464477539063e-8; }
      else if (e === 31) { u[i] = sgn | 0x7f800000 | (m << 13); }
      else { u[i] = sgn | ((e + 112) << 23) | (m << 13); }
    }
    return out;
  }

  // ------------------------------------------------------------ 共通ユーティリティ
  function canvas(w, h) {
    const c = document.createElement('canvas'); c.width = Math.max(1, Math.round(w)); c.height = Math.max(1, Math.round(h)); return c;
  }
  function ctx2d(c) { const x = c.getContext('2d', { willReadFrequently: true }); x.imageSmoothingEnabled = true; x.imageSmoothingQuality = 'high'; return x; }

  // 段階的に縮小（drawImage の 1 回縮小によるジャギーを防ぐ）
  function drawScaled(src, sx, sy, sw, sh, w, h) {
    let cur = src, cx = sx, cy = sy, cw = sw, ch = sh;
    while (cw / 2 > w * 1.01 && ch / 2 > h * 1.01) {
      const nw = Math.round(cw / 2), nh = Math.round(ch / 2);
      const c = canvas(nw, nh); ctx2d(c).drawImage(cur, cx, cy, cw, ch, 0, 0, nw, nh);
      cur = c; cx = 0; cy = 0; cw = nw; ch = nh;
    }
    const out = canvas(w, h); ctx2d(out).drawImage(cur, cx, cy, cw, ch, 0, 0, out.width, out.height);
    return out;
  }

  // u2net：人物の確率マップ（320×320, min-max 正規化）
  async function segment(srcCanvas) {
    const c = canvas(320, 320); ctx2d(c).drawImage(srcCanvas, 0, 0, 320, 320);
    const d = ctx2d(c).getImageData(0, 0, 320, 320).data;
    const N = 320 * 320, t = new Float32Array(3 * N);
    const m = [0.485, 0.456, 0.406], s = [0.229, 0.224, 0.225];
    for (let i = 0; i < N; i++) {
      t[i] = (d[i * 4] / 255 - m[0]) / s[0];
      t[N + i] = (d[i * 4 + 1] / 255 - m[1]) / s[1];
      t[2 * N + i] = (d[i * 4 + 2] / 255 - m[2]) / s[2];
    }
    const feeds = {}; feeds[u2net.inputNames[0]] = new ort.Tensor('float32', t, [1, 3, 320, 320]);
    const out = await u2net.run(feeds, [u2net.outputNames[0]]);
    const o = out[u2net.outputNames[0]].data;
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < N; i++) { if (o[i] < mn) mn = o[i]; if (o[i] > mx) mx = o[i]; }
    const p = new Float32Array(N), r = mx - mn + 1e-8;
    for (let i = 0; i < N; i++) p[i] = (o[i] - mn) / r;
    return p;
  }
  // 320×320 の確率マップを w×h に拡大して 0.5 で 2 値化
  function upMask(p, w, h) {
    const m = new Uint8Array(w * h), sx = 320 / w, sy = 320 / h;
    for (let y = 0; y < h; y++) {
      let fy = (y + 0.5) * sy - 0.5; if (fy < 0) fy = 0; const y0 = Math.min(319, Math.floor(fy)), y1 = Math.min(319, y0 + 1), wy = fy - y0;
      for (let x = 0; x < w; x++) {
        let fx = (x + 0.5) * sx - 0.5; if (fx < 0) fx = 0; const x0 = Math.min(319, Math.floor(fx)), x1 = Math.min(319, x0 + 1), wx = fx - x0;
        const v = (p[y0 * 320 + x0] * (1 - wx) + p[y0 * 320 + x1] * wx) * (1 - wy) + (p[y1 * 320 + x0] * (1 - wx) + p[y1 * 320 + x1] * wx) * wy;
        m[y * w + x] = v > 0.5 ? 1 : 0;
      }
    }
    return m;
  }
  // 8 近傍の連結成分ラベル
  function label(mask, w, h) {
    const lab = new Int32Array(w * h); const sizes = [0]; let n = 0;
    const stack = new Int32Array(w * h);
    for (let i = 0; i < w * h; i++) {
      if (!mask[i] || lab[i]) continue;
      n++; let sp = 0, size = 0; stack[sp++] = i; lab[i] = n;
      while (sp) {
        const j = stack[--sp]; size++;
        const y = (j / w) | 0, x = j - y * w;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy; if (yy < 0 || yy >= h) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx; if (xx < 0 || xx >= w) continue;
            const k = yy * w + xx; if (mask[k] && !lab[k]) { lab[k] = n; stack[sp++] = k; }
          }
        }
      }
      sizes.push(size);
    }
    return { lab, sizes, n };
  }

  // YuNet 顔検出（640×640 にレターボックス）
  async function detectFaces(srcCanvas, th = 0.6) {
    const w = srcCanvas.width, h = srcCanvas.height, s = 640 / Math.max(w, h);
    const c = canvas(640, 640); const x = ctx2d(c); x.fillStyle = '#000'; x.fillRect(0, 0, 640, 640);
    x.drawImage(srcCanvas, 0, 0, Math.round(w * s), Math.round(h * s));
    const d = x.getImageData(0, 0, 640, 640).data; const N = 640 * 640, t = new Float32Array(3 * N);
    for (let i = 0; i < N; i++) { t[i] = d[i * 4 + 2]; t[N + i] = d[i * 4 + 1]; t[2 * N + i] = d[i * 4]; } // BGR, 0-255
    const feeds = {}; feeds[yunet.inputNames[0]] = new ort.Tensor('float32', t, [1, 3, 640, 640]);
    const out = await yunet.run(feeds);
    const faces = [];
    for (const st of [8, 16, 32]) {
      const cols = 640 / st, cls = out['cls_' + st].data, obj = out['obj_' + st].data, bb = out['bbox_' + st].data;
      for (let i = 0; i < cls.length; i++) {
        const sc = Math.sqrt(Math.min(1, Math.max(0, cls[i])) * Math.min(1, Math.max(0, obj[i])));
        if (sc < th) continue;
        const r = (i / cols) | 0, cc = i % cols;
        const cx = (cc + bb[i * 4]) * st, cy = (r + bb[i * 4 + 1]) * st, fw = Math.exp(bb[i * 4 + 2]) * st, fh = Math.exp(bb[i * 4 + 3]) * st;
        faces.push({ x: (cx - fw / 2) / s, y: (cy - fh / 2) / s, w: fw / s, h: fh / s, score: sc });
      }
    }
    faces.sort((a, b) => b.score - a.score);
    const keep = [];
    for (const f of faces) {
      if (keep.every(k => iou(k, f) < 0.3)) keep.push(f);
    }
    return keep;
  }
  function iou(a, b) {
    const x0 = Math.max(a.x, b.x), y0 = Math.max(a.y, b.y), x1 = Math.min(a.x + a.w, b.x + b.w), y1 = Math.min(a.y + a.h, b.y + b.h);
    const i = Math.max(0, x1 - x0) * Math.max(0, y1 - y0); return i / (a.w * a.h + b.w * b.h - i);
  }

  // ------------------------------------------------------------ 検出
  async function detect(bitmap) {
    const W0 = bitmap.width, H0 = bitmap.height;
    const f = Math.min(1, DET_MAX / Math.max(W0, H0));
    const small = drawScaled(bitmap, 0, 0, W0, H0, Math.round(W0 * f), Math.round(H0 * f));
    const w = small.width, h = small.height;

    let faces = await detectFaces(small);
    if (faces.length) {
      // 2 回目：検出スケールのまま顔の周り 640×640 を切り出して再検出（Python/OpenCV 版と同じ精度に）
      const b0 = faces.reduce((a, c) => (c.w * c.h > a.w * a.h ? c : a));
      const ox = Math.round(Math.min(Math.max(0, b0.x + b0.w / 2 - 320), Math.max(0, w - 640)));
      const oy = Math.round(Math.min(Math.max(0, b0.y + b0.h / 2 - 320), Math.max(0, h - 640)));
      const win = canvas(640, 640); const wx = ctx2d(win); wx.fillStyle = '#000'; wx.fillRect(0, 0, 640, 640);
      wx.drawImage(small, ox, oy, Math.min(640, w - ox), Math.min(640, h - oy), 0, 0, Math.min(640, w - ox), Math.min(640, h - oy));
      const f2 = await detectFaces(win);
      const c0x = b0.x + b0.w / 2 - ox, c0y = b0.y + b0.h / 2 - oy;
      const hit = f2.filter(q => Math.abs(q.x + q.w / 2 - c0x) < b0.w && Math.abs(q.y + q.h / 2 - c0y) < b0.h)
                    .sort((a, b) => b.w * b.h - a.w * a.h)[0];
      if (hit) faces = [{ x: hit.x + ox, y: hit.y + oy, w: hit.w, h: hit.h, score: hit.score }];
    }
    const mask = upMask(await segment(small), w, h);
    let fx, fy, fw, fh;
    const hasFace = faces.length > 0;
    if (hasFace) {
      const b = faces.reduce((a, c) => (c.w * c.h > a.w * a.h ? c : a));
      fx = b.x; fy = b.y; fw = b.w; fh = b.h;
    } else {
      const { lab, sizes, n } = label(mask, w, h);
      if (!n) throw new Error('人物を検出できませんでした');
      let big = 1; for (let i = 2; i <= n; i++) if (sizes[i] > sizes[big]) big = i;
      let top = h, bot = 0; for (let i = 0; i < w * h; i++) if (lab[i] === big) { const y = (i / w) | 0; if (y < top) top = y; if (y > bot) bot = y; }
      fh = Math.max(10, (bot - top) * 0.25);
      let mn = w, mx = 0, sum = 0, cnt = 0;
      for (let y = top; y < Math.min(h, top + fh); y++) for (let x = 0; x < w; x++) if (lab[y * w + x] === big) { if (x < mn) mn = x; if (x > mx) mx = x; sum += x; cnt++; }
      fw = Math.max(10, (mx - mn) * 0.8); fx = sum / cnt - fw / 2; fy = top + 0.2 * fh;
    }

    // 顔とつながった人物領域だけ残す
    const { lab, sizes, n } = label(mask, w, h);
    const cx = Math.round(fx + fw / 2), cy = Math.round(fy + fh / 2);
    let lid = lab[Math.min(cy, h - 1) * w + Math.min(cx, w - 1)];
    if (!lid && n) { lid = 1; for (let i = 2; i <= n; i++) if (sizes[i] > sizes[lid]) lid = i; }
    const person = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) person[i] = lid ? (lab[i] === lid ? 1 : 0) : mask[i];

    // 頭頂：顔の上、顔の横幅内でいちばん上の人物ピクセル
    const x0 = Math.max(0, Math.floor(fx - 0.15 * fw)), x1 = Math.min(w, Math.floor(fx + 1.15 * fw));
    const need = Math.max(2, Math.floor(0.03 * fw));
    let headY = null;
    for (let y = 0; y < cy; y++) {
      let c = 0; for (let x = x0; x < x1; x++) c += person[y * w + x];
      if (c >= need) { headY = y; break; }
    }
    let headSrc = 'mask';
    if (headY === null) { headY = Math.max(0, fy - 0.22 * fh); headSrc = 'face'; }

    // 頭部の左右（髪を含む）：頭の周りを高解像度で切り出してもう一度分割
    const inv = 1 / f;
    let hx0 = fx, hx1 = fx + fw;
    try {
      const X0 = Math.max(0, Math.floor((fx - 1.0 * fw) * inv)), Y0 = Math.max(0, Math.floor((headY - 0.3 * fh) * inv));
      const X1 = Math.min(W0, (fx + 2.0 * fw) * inv), Y1 = Math.min(H0, (fy + 1.6 * fh) * inv);
      const cw = X1 - X0, ch = Y1 - Y0, cs = Math.min(1, 640 / Math.max(cw, ch));
      const crop = drawScaled(bitmap, X0, Y0, cw, ch, Math.round(cw * cs), Math.round(ch * cs));
      const hw = crop.width, hh = crop.height;
      const hp = upMask(await segment(crop), hw, hh);
      const L = label(hp, hw, hh);
      const fcx = Math.round(((fx + fw / 2) * inv - X0) * cs), fcy = Math.round((cy * inv - Y0) * cs);
      const id = L.lab[Math.min(fcy, hh - 1) * hw + Math.min(fcx, hw - 1)];
      if (id) {
        let mn = Infinity, mx = -1;
        for (let y = 0; y < Math.min(fcy, hh); y++) for (let x = 0; x < hw; x++) if (L.lab[y * hw + x] === id) { if (x < mn) mn = x; if (x > mx) mx = x; }
        if (mx >= 0) { hx0 = (X0 + mn / cs) * f; hx1 = (X0 + (mx + 1) / cs) * f; }
      }
    } catch (e) { console.warn(e); }

    return {
      img_w: W0, img_h: H0, scale: f, mw: w, mh: h,
      face: [fx * inv, fy * inv, fw * inv, fh * inv],
      head_y: headY * inv, head_src: headSrc, head_x: [hx0 * inv, hx1 * inv],
      mask: person, has_face: hasFace,
    };
  }

  function bodyCenter(det, y0, y1) {
    const f = det.scale, m = det.mask, w = det.mw, h = det.mh;
    const a = Math.max(0, Math.floor(y0 * f)), b = Math.min(h, Math.floor(y1 * f) + 1);
    let mn = Infinity, mx = -1;
    for (let y = a; y < b; y++) for (let x = 0; x < w; x++) if (m[y * w + x]) { if (x < mn) mn = x; if (x > mx) mx = x; }
    if (mx < 0) return det.face[0] + det.face[2] / 2;
    return (mn + mx + 1) / 2 / f;
  }

  // ------------------------------------------------------------ 切り抜き計算
  function computeCrop(det, name, o = {}, imgSize = null) {
    const size = o.size ?? 1, pos = o.pos ?? null, dy = o.dy ?? 0, px = o.px ?? 0.5, py = o.py ?? 0.5, dpos = o.dpos ?? 0;
    let side = o.side || 'auto';
    const p = PRESETS[name], W = p.w, H = p.h;
    const [iw, ih] = det ? [det.img_w, det.img_h] : imgSize;
    const warns = [];
    const kCover = Math.max(W / iw, H / ih);

    if (p.mode === 'plain') {
      const k = kCover * Math.max(1, size), cw = W / k, ch = H / k;
      const x0 = (iw - cw) * Math.min(Math.max(px, 0), 1), y0 = (ih - ch) * Math.min(Math.max(py, 0), 1);
      if (k > 1.02) warns.push(`元画像の解像度が不足、${k.toFixed(2)} 倍に拡大（ぼやけます）`);
      return { box: [x0, y0, x0 + cw, y0 + ch], k, head_top_out: null, target_head_top: null, warnings: warns };
    }

    const T = p.head_top + dy, hy = det.head_y;
    const kRef = p.mode === 'person' ? p.face_ratio * H / det.face[3] * size : kCover * Math.max(1, size);
    const need = [kRef, kCover];
    if (T > 0 && hy > 0) need.push(T / hy);
    if (ih - hy > 0) need.push((H - T) / (ih - hy));
    const [hx0, hx1] = det.head_x || [det.face[0], det.face[0] + det.face[2]];
    if (p.mode === 'profile') {
      if (side !== 'left' && side !== 'right') side = (hx0 + hx1) / 2 >= iw / 2 ? 'right' : 'left';
      const [gl, gr] = p.guides;
      if (side === 'left') need.push(hx0 > 0 ? gl / hx0 : kRef, (W - gl) / Math.max(1, iw - hx0));
      else need.push(hx1 > 0 ? gr / hx1 : kRef, (W - gr) / Math.max(1, iw - hx1));
    }
    const k = Math.max(...need);
    if (p.mode === 'person' && k > kRef * 1.03) warns.push(`元画像の範囲が足りず、人物を ${(k / kRef * size * 100).toFixed(0)}% に拡大`);
    if (p.mode === 'profile' && k > kRef * 1.10) warns.push(`元画像の範囲が足りず、画面を ${(k / kCover * 100).toFixed(0)}% に拡大`);

    const cw = W / k, ch = H / k;
    const y0 = Math.min(Math.max(0, hy - T / k), ih - ch);

    let x0;
    if (p.mode === 'profile') {
      const [gl, gr] = p.guides;
      x0 = side === 'left' ? hx0 - gl / k : hx1 - gr / k;
      x0 -= dpos * cw;
      const xc = Math.min(Math.max(0, x0), iw - cw);
      if (Math.abs(xc - x0) > 1) warns.push('元画像の左右が足りず、頭部をガイド線に合わせられません');
      x0 = xc;
    } else {
      const cx = bodyCenter(det, y0, y0 + ch);
      let target = pos !== null ? pos : p.center;
      target += dpos;
      x0 = cx - target * cw;
      const xc = Math.min(Math.max(0, x0), iw - cw);
      if (Math.abs(xc - x0) > 1) warns.push('元画像の左右が足りず、人物の位置がずれています');
      x0 = xc;
    }
    if (k > 1.02) warns.push(`元画像の解像度が不足、${k.toFixed(2)} 倍に拡大（ぼやけます）`);
    const realTop = (hy - y0) * k;
    if (Math.abs(realTop - T) > 2) warns.push(`頭頂の実際の位置 ${realTop.toFixed(0)}px`);
    const r = { box: [x0, y0, x0 + cw, y0 + ch], k, head_top_out: Math.round(realTop * 10) / 10, target_head_top: T, warnings: warns };
    if (p.mode === 'profile') { r.side = side; r.head_x_out = [(hx0 - x0) * k, (hx1 - x0) * k]; }
    return r;
  }

  // ------------------------------------------------------------ 描画・書き出し
  const GRAY_LUT = new Uint8Array(256).map((_, i) => Math.round(255 * Math.pow(i / 255, GRAY_GAMMA)));
  function toGray(c) {
    const x = ctx2d(c), im = x.getImageData(0, 0, c.width, c.height), d = im.data;
    for (let i = 0; i < d.length; i += 4) {
      const L = GRAY_LUT[(d[i] * 19595 + d[i + 1] * 38470 + d[i + 2] * 7471 + 0x8000) >> 16];
      d[i] = d[i + 1] = d[i + 2] = L;
    }
    x.putImageData(im, 0, 0);
  }
  // プレビュー（速さ優先）
  function renderPreview(bitmap, name, crop) {
    const p = PRESETS[name], [x0, y0, x1, y1] = crop.box;
    const out = drawScaled(bitmap, x0, y0, x1 - x0, y1 - y0, p.w, p.h);
    if (p.gray) toGray(out);
    return out;
  }
  // 書き出し（Lanczos で高画質）
  async function renderFinal(bitmap, name, crop) {
    const p = PRESETS[name], [x0, y0, x1, y1] = crop.box;
    const out = canvas(p.w, p.h);
    try {
      if (!picaInst) throw new Error('no pica');
      const sw = Math.round(x1 - x0), sh = Math.round(y1 - y0);
      const src = canvas(sw, sh); ctx2d(src).drawImage(bitmap, x0, y0, x1 - x0, y1 - y0, 0, 0, sw, sh);
      await picaInst.resize(src, out, { filter: 'lanczos3', unsharpAmount: 0 });
      src.width = src.height = 1;
    } catch (e) {
      console.warn('pica fallback', e);
      ctx2d(out).drawImage(drawScaled(bitmap, x0, y0, x1 - x0, y1 - y0, p.w, p.h), 0, 0);
    }
    if (p.gray) toGray(out);
    return out;
  }
  // JPEG（Q94 / プログレッシブ / 4:4:4）+ 300ppi + sRGB プロファイル埋め込み
  async function encodeJpeg(c) {
    const im = ctx2d(c).getImageData(0, 0, c.width, c.height);
    const buf = new Uint8Array(await encodeFn(im, JPEG_OPTS));
    return patchJpeg(buf);
  }
  function patchJpeg(b) {
    let pos = 2; let app0End = 2;
    if (b[2] === 0xFF && b[3] === 0xE0 && String.fromCharCode(...b.slice(6, 10)) === 'JFIF') {
      b[13] = 1; b[14] = 300 >> 8; b[15] = 300 & 255; b[16] = 300 >> 8; b[17] = 300 & 255;
      app0End = 4 + ((b[4] << 8) | b[5]);
    } else {
      // JFIF がない場合は作る
      const app0 = new Uint8Array([0xFF, 0xE0, 0, 16, 0x4A, 0x46, 0x49, 0x46, 0, 1, 1, 1, 1, 44, 1, 44, 0, 0]);
      b = concat([b.slice(0, 2), app0, b.slice(2)]); app0End = 2 + app0.length;
    }
    const sig = new TextEncoder().encode('ICC_PROFILE\0');
    const len = 2 + sig.length + 2 + srgbIcc.length;
    const app2 = concat([new Uint8Array([0xFF, 0xE2, len >> 8, len & 255]), sig, new Uint8Array([1, 1]), srgbIcc]);
    return concat([b.slice(0, app0End), app2, b.slice(app0End)]);
  }
  function concat(arrs) {
    const n = arrs.reduce((a, x) => a + x.length, 0), o = new Uint8Array(n); let i = 0;
    for (const a of arrs) { o.set(a, i); i += a.length; }
    return o;
  }

  // ------------------------------------------------------------ 画像の管理
  const STORE = {};
  let seq = 0;
  async function loadBitmap(file) {
    try { return await createImageBitmap(file, { imageOrientation: 'from-image', colorSpaceConversion: 'default' }); }
    catch (e) { throw new Error('このファイルを読み込めません'); }
  }
  function info(id) {
    const it = STORE[id], d = it.det;
    const crops = {};
    if (it.slot) for (const n of ORDER.filter(n => PRESETS[n].slot === it.slot)) {
      try { crops[n] = crop(id, n, {}); } catch (e) { crops[n] = { error: e.message }; }
    }
    return { id, name: it.name, slot: it.slot, w: it.bitmap.width, h: it.bitmap.height,
      face: d ? d.face : null, head_y: d ? d.head_y : null, head_src: d ? d.head_src : null,
      det_error: it.det_error || null, thumb: it.thumb, crops };
  }
  async function addImage(file, slotOverride) {
    const name = file.name.replace(/\.[^.]+$/, '');
    const slot = slotOverride || guessSlot(file.name);
    const bitmap = await loadBitmap(file);
    const id = 'i' + (++seq);
    const tc = drawScaled(bitmap, 0, 0, bitmap.width, bitmap.height,
      Math.round(bitmap.width * Math.min(1, 1200 / Math.max(bitmap.width, bitmap.height))),
      Math.round(bitmap.height * Math.min(1, 1200 / Math.max(bitmap.width, bitmap.height))));
    const thumb = URL.createObjectURL(await new Promise(r => tc.toBlob(r, 'image/jpeg', 0.85)));
    STORE[id] = { bitmap, name, slot, det: null, det_error: null, thumb };
    if (slot !== 'text01' && slot !== 'text02') {
      try { STORE[id].det = await detect(bitmap); } catch (e) { STORE[id].det_error = e.message; }
    }
    return info(id);
  }
  async function setSlot(id, slot) {
    const it = STORE[id]; it.slot = slot || null;
    if ((slot === 'person' || slot === 'profile') && !it.det) {
      try { it.det = await detect(it.bitmap); it.det_error = null; } catch (e) { it.det_error = e.message; }
    }
    return info(id);
  }
  function remove(id) {
    const it = STORE[id]; if (!it) return;
    try { it.bitmap.close(); } catch (e) {}
    URL.revokeObjectURL(it.thumb); delete STORE[id];
  }
  function crop(id, preset, o) {
    const it = STORE[id];
    if (PRESETS[preset].mode !== 'plain' && !it.det) throw new Error(`${it.name}：人物を検出できないため ${preset} に使えません`);
    return computeCrop(it.det, preset, o, [it.bitmap.width, it.bitmap.height]);
  }
  async function previewBlob(id, preset, o) {
    const it = STORE[id], c = crop(id, preset, o);
    const cv = renderPreview(it.bitmap, preset, c);
    return await new Promise(r => cv.toBlob(r, 'image/jpeg', 0.9));
  }
  async function exportFile(id, preset, o) {
    const it = STORE[id], c = crop(id, preset, o);
    const cv = await renderFinal(it.bitmap, preset, c);
    const bytes = await encodeJpeg(cv);
    cv.width = cv.height = 1;
    return bytes;
  }

  window.Engine = { PRESETS, ORDER, init, guessSlot, jobName, addImage, setSlot, remove, crop, previewBlob, exportFile,
    _store: STORE, _computeCrop: computeCrop };
})();
