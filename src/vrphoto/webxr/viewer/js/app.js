// view.html entry point: loads scenes/<id>/scene.json, shows an immediate desktop preview, and
// offers "Enter VR" / "Enter AR (passthrough)".  Scenes can be switched inside VR (Next/Previous)
// without leaving the immersive session.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { MODE_INFO, validateManifest } from './manifest.js';
import { PageUI, VRMenu, VRToast } from './ui.js';
import { XRManager } from './xr.js';
import { AbortedError, FriendlyError, clamp, disposeObject3D, fetchJSON, isAbort, isQuest3, log } from './util.js';

/** One loader per mode; literal specifiers so the static import checker can verify them. */
const MODE_LOADERS = {
  photo3d: () => import('./modes/photo3d.js'),
  stereo: () => import('./modes/stereo.js'),
  sphere: () => import('./modes/sphere.js'),
  object: () => import('./modes/object.js'),
  splat: () => import('./modes/splat.js'),
};

const DEFAULT_BG = '#0b0c10';

class ViewerApp {
  constructor() {
    this.ui = new PageUI();
    this.sceneId = null;
    this.manifest = null;
    this.handle = null;
    this.loadToken = 0;
    this.abort = null;
    this.sceneList = null;
    this.passthrough = true;
    this.autoPassthroughOff = false; // passthrough was turned off for a VR-only scene (see _syncPassthroughWithMode)
    this.pendingRecenter = false;
    this.keys = new Set();
    this.lastTime = 0;
    this.savedCamera = null;
    this.debug = {
      ready: false,
      error: null,
      sceneId: null,
      mode: null,
      frames: 0,
      xrFrames: 0,
      presenting: false,
      sessionMode: null,
    };
    window.__vrphoto = this._debugApi();
  }

  // ------------------------------------------------------------------ setup
  async start() {
    const params = new URLSearchParams(location.search);
    const id = params.get('scene');
    if (!id) {
      this.ui.setTitle('No scene selected');
      this.ui.showError(new FriendlyError('No scene selected', 'Go back to the gallery and pick a scene to view.'));
      this._setError('no-scene');
      return;
    }
    this.ui.setLoading(0, 'Loading scene…');
    let manifest;
    try {
      manifest = await this._fetchManifest(id);
    } catch (err) {
      this._fail(err, () => location.reload());
      return;
    }
    try {
      this._initRenderer(manifest.mode);
    } catch (err) {
      this._fail(err);
      return;
    }
    this._initXR();
    this.renderer.setAnimationLoop((t, frame) => this._frame(t, frame));
    await this.loadScene(id, manifest);
  }

  _initRenderer(firstMode) {
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: firstMode !== 'splat', // MSAA does not help splats and costs a lot on Quest
        alpha: true,
        powerPreference: 'high-performance',
      });
    } catch (err) {
      throw new FriendlyError(
        '3D graphics are not available',
        'This browser could not start WebGL 2. Use the Meta Quest Browser or a current Chrome / Edge / Firefox, and make sure hardware acceleration is on.',
        { details: String(err?.message || err), cause: err },
      );
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NoToneMapping;
    document.getElementById('stage').appendChild(renderer.domElement);
    renderer.domElement.tabIndex = 0;
    this.renderer = renderer;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(DEFAULT_BG);
    // Only lit materials (controller / hand models) use these; scene content is unlit.
    scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 2.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.position.set(1, 3, 2);
    scene.add(sun);
    this.scene = scene;

    this.camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.05, 300);
    this.camera.position.set(0, 1.6, 0);
    this.camera.layers.enable(1); // desktop preview shows the left-eye image of stereo content
    scene.add(this.camera);

    this.root = new THREE.Group();
    this.root.name = 'scene-root';
    scene.add(this.root);
    this.extra = []; // scene-level objects owned by the current mode

    const controls = new OrbitControls(this.camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    controls.screenSpacePanning = true;
    this.controls = controls;

    this.menu = new VRMenu();
    this.vrToast = new VRToast();
    scene.add(this.menu.group, this.vrToast.mesh);

    window.addEventListener('resize', () => this._onResize());
    window.addEventListener('keydown', (e) => this._onKey(e, true));
    window.addEventListener('keyup', (e) => this._onKey(e, false));
    window.addEventListener('blur', () => this.keys.clear());
    renderer.domElement.addEventListener('wheel', (e) => this._onWheel(e), { passive: false });
    renderer.domElement.addEventListener('webglcontextlost', (e) => {
      e.preventDefault();
      this.ui.toast('Graphics were reset by the browser - reloading…', 4000);
      setTimeout(() => location.reload(), 1500);
    });
  }

  _initXR() {
    this.xr = new XRManager({
      renderer: this.renderer,
      scene: this.scene,
      handlers: {
        onSessionStart: (mode) => this._onSessionStart(mode),
        onSessionEnd: () => this._onSessionEnd(),
        onSelectStart: (input, ev) => this._onSelectStart(input, ev),
        onSelect: (input, ev) => this._onSelect(input, ev),
        onSelectEnd: (input, ev) => this._onSelectEnd(input, ev),
        onSqueezeStart: (input, ev) => this.handle?.onSqueezeStart?.(input, ev),
        onSqueezeEnd: (input, ev) => this.handle?.onSqueezeEnd?.(input, ev),
        onMenuButton: () => this.toggleMenu(),
        onButton: (name, input) => this.handle?.onButton?.(name, input),
        onReferenceReset: () => {
          this.pendingRecenter = true;
        },
        onSupportChange: () => this._refreshXRButtons(),
      },
    });
    const ui = this.ui.el;
    ui.enterVR.addEventListener('click', () => this.enterXR('immersive-vr'));
    ui.enterAR.addEventListener('click', () => this.enterXR('immersive-ar'));
    this.ui.setXRSupport({ reason: 'checking' });
    this.xr.checkSupport().then(() => this._refreshXRButtons());
  }

  _refreshXRButtons() {
    const prefer = this.manifest?.env?.prefer || 'vr';
    this.ui.setXRSupport(this.xr.support, {
      allowAR: this.handle ? this.handle.supportsAR !== false : true,
      prefer,
      presenting: this.xr.presenting,
    });
  }

  // ------------------------------------------------------------------ scene loading
  async _fetchManifest(id) {
    const url = `scenes/${encodeURIComponent(id)}/scene.json`;
    let raw;
    try {
      raw = await fetchJSON(url);
    } catch (err) {
      if (err instanceof FriendlyError && /404/.test(err.details)) {
        throw new FriendlyError(
          'Scene not found',
          `There is no scene called "${id}" on your PC. It may have been deleted in the desktop app - go back to the gallery and pick another one.`,
          { details: err.details },
        );
      }
      throw err;
    }
    const { ok, errors, manifest } = validateManifest(raw);
    if (!ok) {
      throw new FriendlyError(
        'This scene cannot be shown',
        `${errors[0]} Export the scene again from the desktop app.`,
        { details: errors.join('\n') },
      );
    }
    return manifest;
  }

  /** Load (or switch to) a scene.  Safe to call while in VR. */
  async loadScene(id, manifest = null) {
    const token = ++this.loadToken;
    this.abort?.abort();
    const abort = new AbortController();
    this.abort = abort;
    this.debug.ready = false;
    this.debug.error = null;
    this.ui.hideError();
    this.ui.setLoading(0, 'Loading scene…');
    if (this.xr?.presenting) this.vrToast.show('Loading…', 0);
    try {
      if (!manifest) manifest = await this._fetchManifest(id);
      if (token !== this.loadToken) return;
      this._disposeMode();
      this.sceneId = id;
      this.manifest = manifest;
      this.ui.setTitle(manifest.title, manifest.mode);
      this.menu.setTitle(manifest.title);
      this.ui.setNotices(manifest.notices);
      const loader = MODE_LOADERS[manifest.mode];
      const mod = await loader();
      if (token !== this.loadToken) return;
      const ctx = this._makeContext(id, manifest, abort.signal);
      const label = MODE_INFO[manifest.mode]?.label || 'scene';
      this.ui.setLoading(0.02, `Loading ${label}…`);
      const handle = await mod.load(ctx, manifest);
      if (token !== this.loadToken || abort.signal.aborted) {
        handle?.dispose?.();
        return;
      }
      this.handle = handle;
      this._syncPassthroughWithMode();
      this._applyModeSettings();
      if (this.xr.presenting) {
        // Title first: a mode's own instructions (e.g. "point at the floor") replace it.
        this.vrToast.show(manifest.title, 1800);
        handle.onSessionStart?.({ ar: this.xr.isAR });
        this.xr.resetRig();
        this.pendingRecenter = true;
      }
      this.ui.bindVideo(handle.media?.video || null);
      this.ui.setLoading(null);
      this._refreshXRButtons();
      this._rebuildMenu();
      history.replaceState(null, '', `?scene=${encodeURIComponent(id)}`);
      this.debug.ready = true;
      this.debug.sceneId = id;
      this.debug.mode = manifest.mode;
      log.info(`Scene "${id}" (${manifest.mode}) ready`);
    } catch (err) {
      if (isAbort(err) || token !== this.loadToken) return;
      this._disposeMode();
      this._fail(err, () => this.loadScene(id));
    }
  }

  _makeContext(id, manifest, signal) {
    const base = `scenes/${encodeURIComponent(id)}/`;
    const app = this;
    return {
      renderer: this.renderer,
      scene: this.scene,
      camera: this.camera,
      root: this.root,
      manifest,
      sceneId: id,
      signal,
      xr: this.xr,
      asset: (name) => base + String(name).split('/').map(encodeURIComponent).join('/'),
      progress: (fraction, message) => {
        if (signal.aborted) return;
        app.ui.setLoading(fraction, message || 'Loading…');
        if (app.xr.presenting && message) app.vrToast.show(message, 0);
      },
      toast: (message, ms) => {
        if (app.xr.presenting) app.vrToast.show(message, ms);
        else app.ui.toast(message, ms);
      },
      /**
       * Add an object directly to the scene (not under root); removed when the mode is disposed.
       * A load that was superseded (scene switched meanwhile) must not leave objects behind, so
       * after an abort the object is disposed and AbortedError is thrown instead.
       */
      addToScene: (obj) => {
        if (signal.aborted) {
          disposeObject3D(obj);
          try {
            obj.dispose?.();
          } catch (err) {
            log.warn('dispose failed', err);
          }
          throw new AbortedError();
        }
        app.scene.add(obj);
        app.extra.push(obj);
        return obj;
      },
      refreshMenu: () => app._rebuildMenu(),
      get presenting() {
        return app.xr.presenting;
      },
      get isAR() {
        return app.xr.isAR;
      },
      get passthrough() {
        return app.xr.isAR && app.passthrough;
      },
      checkAborted: () => {
        if (signal.aborted) throw new AbortedError();
      },
    };
  }

  /**
   * Switching (inside an AR session) to a scene that is meant for VR only - a 180/360 sphere -
   * turns passthrough off so it is not drawn over the room; the next AR-capable scene turns it
   * back on unless the user changed it in the menu meanwhile.
   */
  _syncPassthroughWithMode() {
    if (!this.xr?.presenting || !this.xr.isAR || !this.handle) return;
    if (this.handle.supportsAR === false) {
      if (this.passthrough) {
        this.passthrough = false;
        this.autoPassthroughOff = true;
      }
    } else if (this.autoPassthroughOff) {
      this.passthrough = true;
      this.autoPassthroughOff = false;
    }
  }

  _applyModeSettings({ desktopView = true } = {}) {
    const h = this.handle;
    if (!h) return;
    const loc = typeof h.locomotion === 'function' ? h.locomotion(this.xr.isAR) : h.locomotion;
    this.xr.locomotion = { move: true, turn: true, teleport: false, ...(loc || {}) };
    this.xr.floorY = h.floorY ?? 0;
    this._applyBackground();
    if (desktopView && !this.xr.presenting) this._applyDesktopView(h.desktopView || {});
  }

  _applyBackground() {
    const h = this.handle;
    const ar = this.xr?.isAR;
    if (ar && this.passthrough) {
      this.scene.background = null;
    } else {
      const bg = h?.background === undefined ? DEFAULT_BG : h.background;
      this.scene.background = bg === null ? null : new THREE.Color(bg);
    }
    h?.onPassthroughChange?.(!!(ar && this.passthrough));
  }

  _applyDesktopView(view) {
    const c = this.controls;
    const cam = this.camera;
    const pos = view.position || new THREE.Vector3(0, 1.6, 0);
    const target = view.target || new THREE.Vector3(0, 1.6, -2);
    cam.position.copy(pos);
    cam.fov = view.fov || 70;
    cam.updateProjectionMatrix();
    c.target.copy(target);
    c.minDistance = view.minDistance ?? 0.01;
    c.maxDistance = view.maxDistance ?? 50;
    c.minPolarAngle = view.minPolarAngle ?? 0;
    c.maxPolarAngle = view.maxPolarAngle ?? Math.PI;
    c.minAzimuthAngle = view.minAzimuthAngle ?? -Infinity;
    c.maxAzimuthAngle = view.maxAzimuthAngle ?? Infinity;
    c.enableZoom = view.lookAround ? false : view.enableZoom !== false;
    c.enablePan = view.enablePan ?? !view.lookAround;
    c.rotateSpeed = view.lookAround ? -0.35 : 0.8;
    this.lookAround = !!view.lookAround;
    this.fovZoom = !!view.fovZoom;
    this.keyboardMove = view.keyboardMove !== false && !view.lookAround;
    c.update();
  }

  _disposeMode() {
    this.ui.bindVideo(null);
    if (this.handle) {
      try {
        this.handle.dispose?.();
      } catch (err) {
        log.warn('mode dispose failed', err);
      }
    }
    this.handle = null;
    disposeObject3D(this.root);
    this.root.clear();
    this.root.position.set(0, 0, 0);
    this.root.quaternion.identity();
    this.root.scale.set(1, 1, 1);
    for (const obj of this.extra) {
      this.scene.remove(obj);
      disposeObject3D(obj);
      if (typeof obj.dispose === 'function') {
        try {
          obj.dispose();
        } catch (err) {
          log.warn('dispose failed', err);
        }
      }
    }
    this.extra = [];
  }

  _fail(err, retry) {
    log.error(err);
    this.debug.error = err instanceof FriendlyError ? `${err.title}: ${err.message}` : String(err?.message || err);
    this.debug.ready = false;
    if (this.xr?.presenting) this.vrToast.show(err instanceof FriendlyError ? err.title : 'Could not load the scene', 6000);
    this.ui.showError(err, { onRetry: retry });
  }

  _setError(code) {
    this.debug.error = code;
  }

  // ------------------------------------------------------------------ XR
  async enterXR(mode) {
    if (this.xr.presenting) return;
    if (!this.handle) {
      // Buttons appear as soon as XR support is known, possibly before the scene has loaded.
      if (!this.debug.error) this.ui.toast('The scene is still loading - try again in a moment.', 3000);
      return;
    }
    this.ui.setXRBusy(true);
    const heavy = this.handle.heavy === true;
    const scale = this.handle.framebufferScale ?? (isQuest3() && !heavy ? 1.2 : 1.0);
    try {
      // requestSession() runs synchronously inside enter() up to its first await, so it still
      // sees this click's user activation; the click also lets a video start with sound.
      const entering = this.xr.enter(mode, { framebufferScale: scale, frameRate: heavy ? 72 : 90 });
      this._userGesture();
      await entering;
    } catch (err) {
      log.error(err);
      // A dismissible message: the desktop preview stays usable behind it.
      this.ui.showError(err, { onRetry: () => this.ui.hideError(), retryLabel: 'OK' });
    } finally {
      this.ui.setXRBusy(false);
    }
  }

  _onSessionStart(mode) {
    this.debug.presenting = true;
    this.debug.sessionMode = mode;
    this.savedCamera = {
      position: this.camera.position.clone(),
      quaternion: this.camera.quaternion.clone(),
      target: this.controls.target.clone(),
      fov: this.camera.fov,
    };
    this.controls.enabled = false;
    this.passthrough = mode === 'immersive-ar';
    this.autoPassthroughOff = false;
    this.root.position.set(0, 0, 0);
    this.root.quaternion.identity();
    this._applyModeSettings({ desktopView: false });
    this.handle?.onSessionStart?.({ ar: mode === 'immersive-ar' });
    this.pendingRecenter = true;
    this._rebuildMenu();
    this._refreshXRButtons();
    if (this.handle?.media?.video?.muted) this.vrToast.show('Pull the trigger for sound', 3000);
  }

  _onSessionEnd() {
    this.debug.presenting = false;
    this.debug.sessionMode = null;
    this.menu.close();
    this.vrToast.hide();
    this.root.position.set(0, 0, 0);
    this.root.quaternion.identity();
    this.handle?.onSessionEnd?.();
    this.controls.enabled = true;
    const s = this.savedCamera;
    if (s) {
      this.camera.position.copy(s.position);
      this.camera.quaternion.copy(s.quaternion);
      this.camera.fov = s.fov;
      this.controls.target.copy(s.target);
    }
    this._onResize();
    this.passthrough = true;
    this.autoPassthroughOff = false;
    this._applyModeSettings({ desktopView: false });
    this.controls.update();
    this._refreshXRButtons();
  }

  /**
   * Put the user back at the scene's starting viewpoint.  ``auto`` is true for the automatic
   * recenter at session start / scene switch / headset recenter (modes may treat it differently).
   */
  recenter({ auto = false } = {}) {
    const h = this.handle;
    if (!h || !this.xr.presenting) return;
    if (h.onRecenter?.(this.xr.head, { auto }) === true) return;
    const anchor = h.recenterAnchor || { position: new THREE.Vector3(0, 1.6, 0), vertical: false };
    if (this.xr.isAR) {
      const m = this.xr.contentTransformFor(anchor.position, { vertical: anchor.vertical });
      if (m) m.decompose(this.root.position, this.root.quaternion, this.root.scale);
    } else {
      this.xr.recenterTo(anchor.position, { vertical: anchor.vertical });
    }
  }

  // ------------------------------------------------------------------ input routing
  _rayObjects() {
    return this.xr.activeInputs().map((i) => i.controller);
  }

  _onSelectStart(input, ev) {
    this._userGesture();
    if (this.menu.isOpen) {
      const pick = this.menu.pick(input.controller);
      if (pick.onPanel) {
        input.consumed = true;
        if (pick.index >= 0) this.menu.activate(pick.index);
        return;
      }
    }
    input.consumed = false;
    if (this.handle?.onSelectStart?.(input, ev) === true) {
      input.consumed = true;
      return;
    }
    if (this.xr.canTeleport(input)) {
      this.xr.beginTeleport(input);
      input.consumed = true;
    }
  }

  _onSelect(input, ev) {
    if (input.consumed) return;
    this.handle?.onSelect?.(input, ev);
  }

  _onSelectEnd(input, ev) {
    if (this.xr.teleport.input === input) this.xr.endTeleport(input);
    this.handle?.onSelectEnd?.(input, ev);
    input.consumed = false;
  }

  _userGesture() {
    this.handle?.onUserGesture?.();
  }

  toggleMenu() {
    if (!this.xr.presenting) return;
    if (this.menu.isOpen) {
      this.menu.close();
      return;
    }
    this._rebuildMenu();
    this.menu.open(this.xr.head.position, this.xr.head.yaw);
    this._loadSceneList(); // refresh Next/Previous in the background
  }

  _rebuildMenu() {
    const h = this.handle;
    const items = [{ id: 'recenter', label: 'Recenter', action: () => this.recenter() }];
    const list = this.sceneList;
    const canSwitch = Array.isArray(list) && list.length > 1;
    items.push({ id: 'prev', label: '‹ Previous scene', enabled: canSwitch, action: () => this.switchScene(-1) });
    items.push({ id: 'next', label: 'Next scene ›', enabled: canSwitch, action: () => this.switchScene(1) });
    if (this.xr?.isAR) {
      items.push({
        id: 'passthrough',
        label: () => (this.passthrough ? 'Passthrough: on' : 'Passthrough: off'),
        active: this.passthrough,
        action: () => {
          this.passthrough = !this.passthrough;
          this.autoPassthroughOff = false;
          this._applyBackground();
          this._rebuildMenu();
        },
      });
    }
    for (const item of h?.menuItems || []) items.push(item);
    items.push({ id: 'exit', label: this.xr?.isAR ? 'Exit AR' : 'Exit VR', action: () => this.xr.exit() });
    this.menu.setItems(items);
  }

  async _loadSceneList() {
    try {
      const res = await fetch('api/scenes', { cache: 'no-store' });
      if (!res.ok) return;
      const data = await res.json();
      const list = Array.isArray(data) ? data : data?.scenes;
      if (Array.isArray(list)) {
        this.sceneList = list.filter((s) => s && typeof s.id === 'string');
        this._rebuildMenu();
      }
    } catch (err) {
      log.warn('scene list unavailable', err);
    }
  }

  async switchScene(step) {
    if (!this.sceneList) await this._loadSceneList();
    const list = this.sceneList || [];
    if (list.length < 2) return;
    const i = Math.max(0, list.findIndex((s) => s.id === this.sceneId));
    const next = list[(i + step + list.length) % list.length];
    this.menu.close();
    await this.loadScene(next.id);
  }

  // ------------------------------------------------------------------ desktop controls
  _onResize() {
    if (!this.renderer || this.xr?.presenting) return;
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  _onKey(e, down) {
    if (e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
    if (typeof e.key !== 'string' || (down && (e.ctrlKey || e.metaKey || e.altKey))) return; // leave browser shortcuts alone
    const k = e.key.toLowerCase();
    if (['w', 'a', 's', 'd', 'q', 'e', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(k)) {
      if (down) this.keys.add(k);
      else this.keys.delete(k);
      e.preventDefault();
    }
    if (down && k === ' ' && this.handle?.media?.video) {
      const v = this.handle.media.video;
      if (v.paused) v.play().catch(() => {});
      else v.pause();
      e.preventDefault();
    }
  }

  _onWheel(e) {
    if (!this.fovZoom || this.xr?.presenting) return;
    e.preventDefault();
    this.camera.fov = clamp(this.camera.fov + Math.sign(e.deltaY) * 4, 30, 100);
    this.camera.updateProjectionMatrix();
  }

  _desktopUpdate(dt) {
    if (this.keyboardMove && this.keys.size) {
      const fwd = new THREE.Vector3();
      this.camera.getWorldDirection(fwd);
      fwd.y = 0;
      if (fwd.lengthSq() < 1e-6) fwd.set(0, 0, -1);
      fwd.normalize();
      const right = new THREE.Vector3(-fwd.z, 0, fwd.x);
      const move = new THREE.Vector3();
      if (this.keys.has('w') || this.keys.has('arrowup')) move.add(fwd);
      if (this.keys.has('s') || this.keys.has('arrowdown')) move.sub(fwd);
      if (this.keys.has('d') || this.keys.has('arrowright')) move.add(right);
      if (this.keys.has('a') || this.keys.has('arrowleft')) move.sub(right);
      if (this.keys.has('e')) move.y += 1;
      if (this.keys.has('q')) move.y -= 1;
      if (move.lengthSq() > 0) {
        move.normalize().multiplyScalar(1.5 * dt);
        this.camera.position.add(move);
        this.controls.target.add(move);
      }
    }
    this.controls.update();
  }

  // ------------------------------------------------------------------ frame loop
  _frame(time, frame) {
    const dt = this.lastTime ? clamp((time - this.lastTime) / 1000, 0, 0.1) : 0.016;
    this.lastTime = time;
    this.debug.frames += 1;
    try {
      if (frame && this.xr.presenting) {
        this.debug.xrFrames += 1;
        this.xr.update(dt, frame);
        if (this.pendingRecenter && this.xr.head.valid && this.xr.headBase.valid) {
          this.pendingRecenter = false;
          this.recenter({ auto: true });
        }
        const picks = this.menu.update(this._rayObjects());
        for (const input of this.xr.activeInputs()) {
          const pick = picks.get(input.controller);
          this.xr.setRayLength(input, pick && pick.onPanel ? pick.distance : 1.2);
          if (input.ray) input.ray.visible = !input.isHand || this.menu.isOpen;
        }
        this.vrToast.update(this.xr.head.position, this.xr.head.quaternion);
      } else {
        this._desktopUpdate(dt);
      }
      this.handle?.update?.(dt, frame || null);
    } catch (err) {
      log.error('frame update failed', err);
    }
    this.renderer.render(this.scene, this.camera);
    if (this._captureRequests?.length) this._capture();
  }

  /** Read back the frame just rendered (used by the automated browser tests). */
  _capture() {
    const requests = this._captureRequests;
    this._captureRequests = [];
    const gl = this.renderer.getContext();
    let stats = null;
    try {
      const session = this.xr.session;
      const fb = session ? session.renderState.baseLayer?.framebuffer ?? null : null;
      if (!session || session.renderState.baseLayer) {
        this.renderer.state.bindFramebuffer(gl.FRAMEBUFFER, fb);
        const w = gl.drawingBufferWidth;
        const h = gl.drawingBufferHeight;
        const px = new Uint8Array(w * h * 4);
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
        let lit = 0;
        let sum = 0;
        let n = 0;
        const colors = new Set();
        for (let i = 0; i < px.length; i += 4 * 7) {
          const m = Math.max(px[i], px[i + 1], px[i + 2]);
          if (m > 24) lit += 1;
          sum += (px[i] + px[i + 1] + px[i + 2]) / 3;
          colors.add((px[i] >> 4) * 256 + (px[i + 1] >> 4) * 16 + (px[i + 2] >> 4));
          n += 1;
        }
        stats = { width: w, height: h, litFraction: lit / n, mean: sum / n, distinctColors: colors.size };
      }
    } catch (err) {
      stats = { error: String(err) };
    }
    for (const resolve of requests) resolve(stats);
  }

  _debugApi() {
    const app = this;
    return {
      get state() {
        return {
          ...app.debug,
          handle: app.handle?.debug?.() || null,
          menuOpen: app.menu?.isOpen || false,
          vrToast: app.vrToast?.mesh.visible ? app.vrToast.message : null,
        };
      },
      /** Resolve with pixel statistics of the next rendered frame. */
      capture() {
        return new Promise((resolve) => {
          app._captureRequests = app._captureRequests || [];
          app._captureRequests.push(resolve);
        });
      },
      recenter: () => app.recenter(),
      get pendingRecenter() {
        return app.pendingRecenter;
      },
      toggleMenu: () => app.toggleMenu(),
      menuItems: () => app.menu.items.map((i) => (typeof i.label === 'function' ? i.label() : i.label)),
      activateMenu: (id) => {
        const idx = app.menu.items.findIndex((i) => i.id === id);
        return idx >= 0 ? app.menu.activate(idx) : false;
      },
      loadScene: (id) => app.loadScene(id),
      get app() {
        return app;
      },
    };
  }
}

const app = new ViewerApp();
app.start().catch((err) => {
  log.error(err);
  app.ui.showError(err);
});
window.addEventListener('unhandledrejection', (e) => {
  if (isAbort(e.reason)) e.preventDefault();
});
