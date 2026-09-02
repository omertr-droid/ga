// lib.js — tiny DOM + image helpers (no framework, no build). Copied from the viewer; unchanged.
export function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'class') e.className = v;
    else if (k === 'style') e.style.cssText = v;
    else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2).toLowerCase(), v);
    else if (v === true) e.setAttribute(k, '');
    else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const k of kids.flat()) {
    if (k == null) continue;
    e.append(k.nodeType ? k : document.createTextNode(k));
  }
  return e;
}

export function loadImage(src) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error('image load failed: ' + src));
    im.src = src;
  });
}

export function fmtArea(v) {
  return (v == null || isNaN(v)) ? 'n/a' : Number(v).toFixed(2) + ' mm²';
}
