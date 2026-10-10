// User interface: the 2D page controls (title, XR buttons, loading/error overlays, toasts,
// video controls) and the in-VR pieces (menu panel with ray-picked buttons, head-locked toast).

import * as THREE from 'three';
import { MODE_INFO } from './manifest.js';
import { FriendlyError } from './util.js';

const $ = (id) => document.getElementById(id);

// ============================================================================ 2D page UI
export class PageUI {
  constructor() {
    this.el = {
      title: $('scene-title'),
      mode: $('scene-mode'),
      loading: $('loading'),
      loadingText: $('loading-text'),
      loadingBar: $('loading-bar'),
      error: $('error'),
      errorTitle: $('error-title'),
      errorText: $('error-text'),
      errorDetails: $('error-details'),
      errorDetailsText: $('error-details-text'),
      retry: $('error-retry'),
      enterVR: $('enter-vr'),
      enterAR: $('enter-ar'),
      xrStatus: $('xr-status'),
      xrNote: $('xr-note'),
      notices: $('notices'),
      mediaBar: $('media-bar'),
      play: $('media-play'),
      mute: $('media-mute'),
      help: $('help'),
      helpBtn: $('help-btn'),
      toast: $('toast'),
    };
    this._toastTimer = 0;
    this._video = null;
    this._onVideoEvent = () => this._syncMedia();
    this.el.helpBtn?.addEventListener('click', () => this.el.help.classList.toggle('hidden'));
    this.el.play?.addEventListener('click', () => this._togglePlay());
    this.el.mute?.addEventListener('click', () => this._toggleMute());
  }

  setTitle(title, mode) {
    this.el.title.textContent = title || 'Scene';
    document.title = `${title || 'Scene'} - VR Photo Studio`;
    const info = MODE_INFO[mode];
    this.el.mode.textContent = info ? info.label : '';
    this.el.mode.className = `badge mode-${mode || ''}`;
    this.el.mode.title = info?.hint || '';
    this.el.mode.classList.toggle('hidden', !info);
  }

  setLoading(fraction, message = 'Loading…') {
    if (fraction === null) {
      this.el.loading.classList.add('hidden');
      return;
    }
    this.el.loading.classList.remove('hidden');
    this.el.loadingText.textContent = message;
    const pct = fraction < 0 ? 100 : Math.round(Math.min(1, Math.max(0, fraction)) * 100);
    this.el.loadingBar.style.width = `${pct}%`;
    this.el.loadingBar.parentElement.classList.toggle('indeterminate', fraction < 0);
  }

  /**
   * Show the error overlay.  ``onRetry`` adds the main button (labelled ``retryLabel``); pass
   * ``{ onRetry: () => ui.hideError(), retryLabel: 'OK' }`` for a message the user just dismisses.
   */
  showError(err, { onRetry, retryLabel = 'Try again' } = {}) {
    const friendly = err instanceof FriendlyError;
    this.el.errorTitle.textContent = friendly ? err.title : 'Something went wrong';
    this.el.errorText.textContent = friendly
      ? err.message
      : 'The viewer hit an unexpected problem. Try again, or go back to the gallery and pick the scene again.';
    const details = friendly ? err.details : String(err?.stack || err?.message || err);
    this.el.errorDetailsText.textContent = details || '';
    this.el.errorDetails.classList.toggle('hidden', !details);
    this.el.retry.textContent = retryLabel;
    this.el.retry.classList.toggle('hidden', !onRetry);
    this.el.retry.onclick = onRetry ? () => onRetry() : null;
    this.setLoading(null);
    this.el.error.classList.remove('hidden');
  }

  hideError() {
    this.el.error.classList.add('hidden');
  }

  toast(message, ms = 2600) {
    const t = this.el.toast;
    t.textContent = message;
    t.classList.remove('hidden');
    clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => t.classList.add('hidden'), ms);
  }

  setNotices(list) {
    const items = (list || []).filter((s) => typeof s === 'string' && s.trim());
    this.el.notices.textContent = items.join(' • ');
    this.el.notices.classList.toggle('hidden', items.length === 0);
  }

  /**
   * Show the XR entry buttons.  ``support`` = { vr, ar, reason } from XRManager.checkSupport().
   * ``allowAR`` = whether the current scene makes sense in passthrough.
   */
  setXRSupport(support, { allowAR = true, prefer = 'vr', presenting = false } = {}) {
    const { enterVR, enterAR, xrStatus, xrNote } = this.el;
    const showVR = !!support.vr && !presenting;
    const showAR = !!support.ar && allowAR && !presenting;
    enterVR.classList.toggle('hidden', !showVR);
    enterAR.classList.toggle('hidden', !showAR);
    // Preferred session first (object scenes prefer AR on Quest 3).
    if (prefer === 'ar' && showAR) enterAR.parentElement.insertBefore(enterAR, enterVR);
    else enterVR.parentElement.insertBefore(enterVR, enterAR);
    enterAR.classList.toggle('primary', prefer === 'ar');
    enterAR.classList.toggle('accent', prefer !== 'ar');
    enterVR.classList.toggle('primary', prefer !== 'ar' || !showAR);
    xrNote.classList.add('hidden');
    xrStatus.textContent = '';
    if (presenting) return;
    if (support.reason === 'insecure') {
      xrNote.innerHTML =
        '<strong>VR needs a secure connection.</strong> Open the <b>https://</b> link shown in the ' +
        'VR Photo Studio desktop app (or scan its QR code) and accept the certificate warning ' +
        '(<i>Advanced &rarr; Proceed</i>), or use <b>USB mode</b> (<code>http://localhost</code>).';
      xrNote.classList.remove('hidden');
    } else if (support.reason === 'no-webxr') {
      xrStatus.textContent =
        'Preview mode: this browser has no WebXR. On your Quest, open this page in the Meta Quest Browser to enter VR.';
    } else if (support.reason === 'no-device') {
      xrStatus.textContent = 'Preview mode: no VR headset found. Open this page on your Quest to enter VR.';
    } else if (support.reason === 'checking') {
      xrStatus.textContent = 'Checking for VR support…';
    }
  }

  setXRBusy(busy) {
    this.el.enterVR.disabled = busy;
    this.el.enterAR.disabled = busy;
  }

  /** Bind the bottom-right play/pause + mute buttons to a video element (or null to hide). */
  bindVideo(video) {
    if (this._video) {
      for (const ev of ['play', 'pause', 'volumechange']) this._video.removeEventListener(ev, this._onVideoEvent);
    }
    this._video = video;
    this.el.mediaBar.classList.toggle('hidden', !video);
    if (!video) return;
    for (const ev of ['play', 'pause', 'volumechange']) video.addEventListener(ev, this._onVideoEvent);
    this._syncMedia();
  }

  _syncMedia() {
    const v = this._video;
    if (!v) return;
    this.el.play.textContent = v.paused ? '▶ Play' : '⏸ Pause';
    this.el.mute.textContent = v.muted ? '\u{1F507} Sound off' : '\u{1F50A} Sound on';
  }

  _togglePlay() {
    const v = this._video;
    if (!v) return;
    if (v.paused) v.play().catch(() => this.toast('Press play again to start the video'));
    else v.pause();
  }

  _toggleMute() {
    const v = this._video;
    if (!v) return;
    v.muted = !v.muted;
    if (!v.muted && v.paused) v.play().catch(() => {});
  }
}

// ============================================================================ canvas helpers
function roundRect(g, x, y, w, h, r) {
  g.beginPath();
  g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r);
  g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r);
  g.arcTo(x, y, x + w, y, r);
  g.closePath();
}

const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif';

function fitText(g, text, maxWidth, size, weight = 600) {
  let s = size;
  g.font = `${weight} ${s}px ${FONT}`;
  while (s > 14 && g.measureText(text).width > maxWidth) {
    s -= 2;
    g.font = `${weight} ${s}px ${FONT}`;
  }
  return s;
}

// ============================================================================ in-VR menu
const MENU_W = 640;
const COLS = 2;
const PAD = 22;
const HEADER = 92;
const BTN_H = 96;
const GAP = 16;
const PANEL_WIDTH_M = 0.46;

/**
 * Floating menu panel for VR/AR.  Items: ``{ id, label, action, enabled?, active? }`` where
 * ``label`` may be a function (re-evaluated whenever the menu redraws).
 */
export class VRMenu {
  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'vr-menu';
    this.group.visible = false;
    this.canvas = document.createElement('canvas');
    this.canvas.width = MENU_W;
    this.canvas.height = 256;
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.texture.anisotropy = 4;
    this.material = new THREE.MeshBasicMaterial({ map: this.texture, transparent: true, depthTest: false, toneMapped: false });
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), this.material);
    this.mesh.renderOrder = 100;
    this.group.add(this.mesh);
    this.items = [];
    this.title = 'Menu';
    this.hover = new Map(); // input -> index
    this.pressed = -1;
    this.raycaster = new THREE.Raycaster();
    this._tmpM = new THREE.Matrix4();
    this._rects = [];
    this._dirty = true;
  }

  get isOpen() {
    return this.group.visible;
  }

  setTitle(title) {
    this.title = title || 'Menu';
    this._dirty = true;
  }

  setItems(items) {
    this.items = items.filter(Boolean);
    this._dirty = true;
    this._layout();
  }

  _layout() {
    const rows = Math.max(1, Math.ceil(this.items.length / COLS));
    const h = HEADER + rows * BTN_H + (rows - 1) * GAP + PAD;
    if (this.canvas.height !== h) {
      this.canvas.height = h;
      // Canvas size changed: three needs a fresh texture of the new size.
      this.texture.dispose();
      this.texture = new THREE.CanvasTexture(this.canvas);
      this.texture.colorSpace = THREE.SRGBColorSpace;
      this.texture.anisotropy = 4;
      this.material.map = this.texture;
      this.material.needsUpdate = true;
    }
    const bw = (MENU_W - PAD * 2 - GAP * (COLS - 1)) / COLS;
    this._rects = this.items.map((_, i) => {
      const r = Math.floor(i / COLS);
      const c = i % COLS;
      return { x: PAD + c * (bw + GAP), y: HEADER + r * (BTN_H + GAP), w: bw, h: BTN_H };
    });
    this.mesh.scale.set(PANEL_WIDTH_M, (PANEL_WIDTH_M * h) / MENU_W, 1);
  }

  _draw() {
    const g = this.canvas.getContext('2d');
    const W = this.canvas.width;
    const H = this.canvas.height;
    g.clearRect(0, 0, W, H);
    roundRect(g, 0, 0, W, H, 34);
    g.fillStyle = 'rgba(20, 22, 29, 0.94)';
    g.fill();
    g.lineWidth = 3;
    g.strokeStyle = 'rgba(255,255,255,0.12)';
    g.stroke();
    g.fillStyle = '#eef0f5';
    fitText(g, this.title, W - PAD * 2 - 60, 34, 700);
    g.textBaseline = 'middle';
    g.textAlign = 'left';
    g.fillText(this.title, PAD + 6, HEADER / 2);
    g.fillStyle = 'rgba(163,169,183,0.9)';
    g.font = `500 22px ${FONT}`;
    g.textAlign = 'right';
    g.fillText('Y / B', W - PAD - 6, HEADER / 2);
    const hovered = new Set(this.hover.values());
    this.items.forEach((item, i) => {
      const r = this._rects[i];
      const enabled = item.enabled !== false;
      roundRect(g, r.x, r.y, r.w, r.h, 22);
      let fill = 'rgba(44, 49, 61, 0.95)';
      if (item.active) fill = 'rgba(79, 140, 255, 0.55)';
      if (hovered.has(i) && enabled) fill = item.active ? 'rgba(79, 140, 255, 0.85)' : 'rgba(79, 140, 255, 0.6)';
      if (this.pressed === i) fill = 'rgba(139, 92, 246, 0.9)';
      g.fillStyle = fill;
      g.fill();
      const label = typeof item.label === 'function' ? item.label() : item.label;
      g.fillStyle = enabled ? '#ffffff' : 'rgba(255,255,255,0.35)';
      g.textAlign = 'center';
      fitText(g, label, r.w - 24, 30, 650);
      g.fillText(label, r.x + r.w / 2, r.y + r.h / 2 + 1);
    });
    this.texture.needsUpdate = true;
    this._dirty = false;
  }

  /** Place the menu ~0.6 m in front of the head, slightly below eye level, facing the user. */
  open(headPosition, headYaw) {
    this._layout();
    this._draw();
    const fwd = new THREE.Vector3(-Math.sin(headYaw), 0, -Math.cos(headYaw));
    this.group.position.copy(headPosition).addScaledVector(fwd, 0.6);
    this.group.position.y -= 0.18;
    this.group.rotation.set(-0.25, headYaw, 0, 'YXZ');
    this.group.visible = true;
  }

  close() {
    this.group.visible = false;
    this.hover.clear();
    this.pressed = -1;
  }

  /** Index of the button under the target ray of ``rayObject`` (controller group) or -1. */
  pick(rayObject) {
    if (!this.isOpen || !rayObject) return { index: -1, distance: Infinity };
    this._tmpM.identity().extractRotation(rayObject.matrixWorld);
    this.raycaster.ray.origin.setFromMatrixPosition(rayObject.matrixWorld);
    this.raycaster.ray.direction.set(0, 0, -1).applyMatrix4(this._tmpM).normalize();
    const hit = this.raycaster.intersectObject(this.mesh, false)[0];
    if (!hit || !hit.uv) return { index: -1, distance: Infinity };
    const x = hit.uv.x * this.canvas.width;
    const y = (1 - hit.uv.y) * this.canvas.height;
    const index = this._rects.findIndex((r) => x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h);
    return { index, distance: hit.distance, onPanel: true };
  }

  /** Update hover highlights for the given ray objects; returns Map(rayObject -> pick result). */
  update(rayObjects) {
    const results = new Map();
    if (!this.isOpen) return results;
    let changed = false;
    for (const obj of rayObjects) {
      const res = this.pick(obj);
      results.set(obj, res);
      const prev = this.hover.has(obj) ? this.hover.get(obj) : -1;
      if (prev !== res.index) {
        changed = true;
        if (res.index >= 0) this.hover.set(obj, res.index);
        else this.hover.delete(obj);
      }
    }
    if (changed || this._dirty) this._draw();
    return results;
  }

  /** Activate button ``index`` (returns true if an action ran). */
  activate(index) {
    const item = this.items[index];
    if (!item || item.enabled === false) return false;
    this.pressed = index;
    this._draw();
    setTimeout(() => {
      this.pressed = -1;
      this._dirty = true;
    }, 160);
    try {
      item.action?.();
    } finally {
      this._dirty = true;
    }
    return true;
  }

  refresh() {
    this._dirty = true;
  }

  dispose() {
    this.texture.dispose();
    this.material.dispose();
    this.mesh.geometry.dispose();
  }
}

// ============================================================================ VR toast
/** A short head-locked message ("Loading...", "Tap the floor to place"). */
export class VRToast {
  constructor() {
    this.canvas = document.createElement('canvas');
    this.canvas.width = 1024;
    this.canvas.height = 128;
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(0.64, 0.08),
      new THREE.MeshBasicMaterial({ map: this.texture, transparent: true, depthTest: false, toneMapped: false }),
    );
    this.mesh.name = 'vr-toast';
    this.mesh.renderOrder = 200;
    this.mesh.visible = false;
    this.until = 0;
    this.message = '';
    this._offset = new THREE.Vector3(0, -0.16, -1.0);
  }

  show(message, ms = 2500) {
    this.message = message;
    const g = this.canvas.getContext('2d');
    g.clearRect(0, 0, this.canvas.width, this.canvas.height);
    roundRect(g, 4, 4, this.canvas.width - 8, this.canvas.height - 8, 56);
    g.fillStyle = 'rgba(20, 22, 29, 0.9)';
    g.fill();
    g.fillStyle = '#ffffff';
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    fitText(g, message, this.canvas.width - 80, 50, 650);
    g.fillText(message, this.canvas.width / 2, this.canvas.height / 2 + 2);
    this.texture.needsUpdate = true;
    this.mesh.visible = true;
    this.until = ms > 0 ? performance.now() + ms : Infinity;
  }

  hide() {
    this.mesh.visible = false;
  }

  /** Keep the toast in front of the head (call every XR frame with the current head pose). */
  update(headPosition, headQuaternion) {
    if (!this.mesh.visible) return;
    if (performance.now() > this.until) {
      this.mesh.visible = false;
      return;
    }
    this.mesh.position.copy(this._offset).applyQuaternion(headQuaternion).add(headPosition);
    this.mesh.quaternion.copy(headQuaternion);
  }

  dispose() {
    this.texture.dispose();
    this.mesh.material.dispose();
    this.mesh.geometry.dispose();
  }
}
