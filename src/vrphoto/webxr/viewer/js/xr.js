// WebXR session management and common XR interaction:
//  * entering VR / AR (required local-floor; optional hands, hit-test, anchors, planes, layers)
//  * controller + hand models from the vendored input profiles (never a CDN)
//  * a "rig" (user origin in the scene) applied with XRReferenceSpace.getOffsetReferenceSpace:
//    smooth move (left stick), 30 degree snap turn (right stick), arc teleport (right trigger),
//    and recentering so a scene starts exactly where the user's head is
//  * button edge detection (Y / B toggle the menu, A / X for mode actions) and a wrist button
//    for hand tracking.

import * as THREE from 'three';
import { XRControllerModelFactory } from 'three/addons/webxr/XRControllerModelFactory.js';
import { XRHandModelFactory } from 'three/addons/webxr/XRHandModelFactory.js';
import { getGLTFLoader } from './loaders.js';
import { createReticle } from './env.js';
import { FriendlyError, log } from './util.js';

/** Local input-profile assets (resolved relative to this module, so any page location works). */
export const PROFILES_PATH = new URL('../vendor/webxr-input-profiles/profiles', import.meta.url).href;
export const HAND_PROFILE_PATH = `${PROFILES_PATH}/generic-hand/`;

export const OPTIONAL_FEATURES = ['hand-tracking', 'hit-test', 'anchors', 'plane-detection', 'layers', 'bounded-floor'];

const MOVE_SPEED = 1.4; // m/s at full stick (comfortable walking pace)
const SNAP_ANGLE = THREE.MathUtils.degToRad(30);
const DEADZONE = 0.18;
const TELEPORT_SPEED = 7.0; // m/s launch speed of the arc
const TELEPORT_MAX_DIST = 15;
const GRAVITY = 9.8;
const ARC_POINTS = 64;

const _v = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _m = new THREE.Matrix4();
const _up = new THREE.Vector3(0, 1, 0);

/** Yaw (rotation about +y) of a quaternion's forward (-z) direction. */
export function yawOf(quaternion) {
  const f = _v2.set(0, 0, -1).applyQuaternion(quaternion);
  return Math.atan2(-f.x, -f.z);
}

/** Horizontal forward unit vector for a yaw angle. */
export function forwardFromYaw(yaw, out = new THREE.Vector3()) {
  return out.set(-Math.sin(yaw), 0, -Math.cos(yaw));
}

function poseToObject(transform, position, quaternion) {
  const p = transform.position;
  const o = transform.orientation;
  position.set(p.x, p.y, p.z);
  quaternion.set(o.x, o.y, o.z, o.w);
}

/** One tracked input (controller or hand) as seen by modes. */
class XRInput {
  constructor(index, controller, grip, hand) {
    this.index = index;
    this.controller = controller; // target-ray space group (select/squeeze events come from here)
    this.grip = grip; // grip space group (holds the controller model)
    this.hand = hand; // hand group with joints
    this.inputSource = null;
    this.handedness = 'none';
    this.isHand = false;
    this.connected = false;
    this.buttons = []; // previous pressed states for edge detection
    this.ray = null; // visual pointer line
    this.selecting = false;
    this.squeezing = false;
  }

  get gamepad() {
    return this.inputSource?.gamepad || null;
  }

  /** Thumbstick (or touchpad) axes as {x, y}; y < 0 is "forward" (xr-standard mapping). */
  stick() {
    const gp = this.gamepad;
    if (!gp || !gp.axes) return { x: 0, y: 0 };
    const a = gp.axes;
    if (a.length >= 4) return { x: a[2] || 0, y: a[3] || 0 };
    if (a.length >= 2) return { x: a[0] || 0, y: a[1] || 0 };
    return { x: 0, y: 0 };
  }

  /** World position and direction of the target ray. */
  rayWorld(origin, direction) {
    this.controller.updateMatrixWorld();
    origin.setFromMatrixPosition(this.controller.matrixWorld);
    _m.extractRotation(this.controller.matrixWorld);
    direction.set(0, 0, -1).applyMatrix4(_m).normalize();
  }
}

export class XRManager {
  /**
   * @param {object} opts
   * @param {THREE.WebGLRenderer} opts.renderer
   * @param {THREE.Scene} opts.scene
   * @param {object} opts.handlers callbacks: onSessionStart(mode), onSessionEnd(), onSelectStart(input, ev),
   *   onSelect(input, ev), onSelectEnd(input, ev), onSqueezeStart(input, ev), onSqueezeEnd(input, ev),
   *   onMenuButton(input), onButton(name, input), onReferenceReset()
   */
  constructor({ renderer, scene, handlers = {} }) {
    this.renderer = renderer;
    this.scene = scene;
    this.handlers = handlers;
    this.session = null;
    this.sessionMode = null;
    this.baseRefSpace = null;
    this.refSpaceType = 'local-floor';
    this.rigPosition = new THREE.Vector3();
    this.rigYaw = 0;
    this.locomotion = { move: true, turn: true, teleport: true };
    this.floorY = 0;
    this.head = { position: new THREE.Vector3(0, 1.6, 0), quaternion: new THREE.Quaternion(), yaw: 0, valid: false };
    this.headBase = { position: new THREE.Vector3(0, 1.6, 0), quaternion: new THREE.Quaternion(), yaw: 0, valid: false };
    this.snapArmed = true;
    this.teleport = { input: null, valid: false, point: new THREE.Vector3() };
    this.wrist = { lastToggle: 0, mesh: null };
    this.frames = 0;
    this._support = { vr: false, ar: false, reason: 'checking' };

    renderer.xr.enabled = true;
    renderer.xr.setReferenceSpaceType('local-floor');
    renderer.xr.addEventListener('sessionstart', () => this._onSessionStart());
    renderer.xr.addEventListener('sessionend', () => this._onSessionEnd());

    const controllerModels = new XRControllerModelFactory(getGLTFLoader()).setPath(PROFILES_PATH);
    const handModels = new XRHandModelFactory(getGLTFLoader()).setPath(HAND_PROFILE_PATH);
    this.inputs = [0, 1].map((i) => {
      const controller = renderer.xr.getController(i);
      const grip = renderer.xr.getControllerGrip(i);
      const hand = renderer.xr.getHand(i);
      grip.add(controllerModels.createControllerModel(grip));
      hand.add(handModels.createHandModel(hand, 'mesh'));
      scene.add(controller, grip, hand);
      const input = new XRInput(i, controller, grip, hand);
      this._wireInput(input);
      return input;
    });

    // Teleport arc + landing reticle.
    const arcGeo = new THREE.BufferGeometry();
    arcGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(ARC_POINTS * 3), 3));
    this.arc = new THREE.Line(arcGeo, new THREE.LineBasicMaterial({ color: 0x5ec8ff, transparent: true, opacity: 0.9 }));
    this.arc.frustumCulled = false;
    this.arc.visible = false;
    this.arc.renderOrder = 60;
    this.teleportReticle = createReticle('#5ec8ff');
    scene.add(this.arc, this.teleportReticle);

    // Wrist button for hand tracking (palm towards face -> touch with the other index finger).
    const wristMesh = new THREE.Mesh(
      new THREE.CircleGeometry(0.018, 32),
      new THREE.MeshBasicMaterial({ color: 0x4f8cff, transparent: true, opacity: 0.95, depthTest: false }),
    );
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.019, 0.024, 32),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9, depthTest: false }),
    );
    wristMesh.add(ring);
    wristMesh.renderOrder = ring.renderOrder = 120;
    wristMesh.visible = false;
    this.wrist.mesh = wristMesh;
    scene.add(wristMesh);

    if (navigator.xr?.addEventListener) {
      navigator.xr.addEventListener('devicechange', () => this.checkSupport().then((s) => this.handlers.onSupportChange?.(s)));
    }
  }

  // ------------------------------------------------------------------ support / sessions
  get presenting() {
    return !!this.session;
  }

  get isAR() {
    return this.sessionMode === 'immersive-ar';
  }

  get support() {
    return this._support;
  }

  /** Detect VR / AR support: { vr, ar, reason } with reason "insecure" | "no-webxr" | "no-device" | null. */
  async checkSupport() {
    let result;
    if (!window.isSecureContext) result = { vr: false, ar: false, reason: 'insecure' };
    else if (!navigator.xr || typeof navigator.xr.isSessionSupported !== 'function') {
      result = { vr: false, ar: false, reason: 'no-webxr' };
    } else {
      const check = (mode) => navigator.xr.isSessionSupported(mode).catch(() => false);
      const [vr, ar] = await Promise.all([check('immersive-vr'), check('immersive-ar')]);
      result = { vr, ar, reason: vr || ar ? null : 'no-device' };
    }
    this._support = result;
    return result;
  }

  /**
   * Start an immersive session.  Must be called from a user gesture (button click).
   * @param {'immersive-vr'|'immersive-ar'} mode
   */
  async enter(mode, { framebufferScale = 1.0, frameRate = 0 } = {}) {
    if (this.session) return;
    if (!navigator.xr) throw new FriendlyError('VR is not available', 'This browser does not support WebXR.');
    let session;
    this.refSpaceType = 'local-floor';
    try {
      session = await navigator.xr.requestSession(mode, {
        requiredFeatures: ['local-floor'],
        optionalFeatures: OPTIONAL_FEATURES,
      });
    } catch (err) {
      if (err?.name !== 'NotSupportedError') throw this._sessionError(mode, err);
      // Some runtimes have no floor-level space: fall back to "local" and lift the user ~1.6 m.
      try {
        session = await navigator.xr.requestSession(mode, { optionalFeatures: ['local-floor', ...OPTIONAL_FEATURES] });
      } catch (err2) {
        throw this._sessionError(mode, err2);
      }
      this.refSpaceType = session.enabledFeatures?.includes?.('local-floor') ? 'local-floor' : 'local';
    }
    this.sessionMode = mode;
    this.renderer.xr.setReferenceSpaceType(this.refSpaceType);
    this.renderer.xr.setFramebufferScaleFactor(framebufferScale);
    this.renderer.xr.setFoveation(1);
    this._pendingFrameRate = frameRate;
    try {
      await this.renderer.xr.setSession(session);
    } catch (err) {
      session.end().catch(() => {});
      this.sessionMode = null;
      throw this._sessionError(mode, err);
    }
  }

  _sessionError(mode, err) {
    const what = mode === 'immersive-ar' ? 'passthrough AR' : 'VR';
    let msg = `The headset could not start ${what}.`;
    if (err?.name === 'SecurityError' || err?.name === 'NotAllowedError') {
      msg += ' Allow the permission prompt in the headset, and make sure this page was opened with https:// (or USB mode).';
    } else if (err?.name === 'NotSupportedError') {
      msg += mode === 'immersive-ar' ? ' This headset does not support passthrough in the browser.' : ' This device does not support immersive VR.';
    } else {
      msg += ' Close other VR apps and try again.';
    }
    return new FriendlyError(`Could not start ${what}`, msg, { details: `${err?.name}: ${err?.message}`, cause: err });
  }

  /** End the current session (returns to the 2D page). */
  async exit() {
    if (this.session) await this.session.end().catch(() => {});
  }

  _onSessionStart() {
    this.session = this.renderer.xr.getSession();
    this.baseRefSpace = this.renderer.xr.getReferenceSpace();
    this.frames = 0;
    this.head.valid = this.headBase.valid = false;
    this.rigPosition.set(0, this.refSpaceType === 'local' ? 1.6 : 0, 0);
    this.rigYaw = 0;
    this._applyRig();
    this._onRefReset = () => this.handlers.onReferenceReset?.();
    this.baseRefSpace?.addEventListener?.('reset', this._onRefReset);
    const fr = this._pendingFrameRate;
    const rates = this.session.supportedFrameRates;
    if (fr && rates && typeof this.session.updateTargetFrameRate === 'function') {
      const best = [...rates].filter((r) => r <= fr).sort((a, b) => b - a)[0];
      if (best) this.session.updateTargetFrameRate(best).catch(() => {});
    }
    log.info(`XR session started: ${this.sessionMode}, features: ${(this.session.enabledFeatures || []).join(', ')}`);
    this.handlers.onSessionStart?.(this.sessionMode);
  }

  _onSessionEnd() {
    this.cancelTeleport();
    this.baseRefSpace?.removeEventListener?.('reset', this._onRefReset);
    this.session = null;
    this.baseRefSpace = null;
    this.sessionMode = null;
    this.wrist.mesh.visible = false;
    for (const input of this.inputs) {
      input.connected = false;
      input.inputSource = null;
      input.selecting = input.squeezing = false;
    }
    this.handlers.onSessionEnd?.();
  }

  hasFeature(name) {
    return !!this.session?.enabledFeatures?.includes?.(name);
  }

  // ------------------------------------------------------------------ inputs
  _wireInput(input) {
    const c = input.controller;
    c.addEventListener('connected', (e) => {
      const src = e.data;
      input.inputSource = src;
      input.handedness = src.handedness || 'none';
      input.isHand = !!src.hand;
      input.connected = true;
      input.buttons = [];
      if (src.targetRayMode === 'tracked-pointer' && !input.ray) input.ray = this._makeRay(c);
      if (input.ray) input.ray.visible = !input.isHand;
    });
    c.addEventListener('disconnected', () => {
      input.connected = false;
      input.inputSource = null;
      input.selecting = input.squeezing = false;
      if (this.teleport.input === input) this.cancelTeleport();
      if (input.ray) input.ray.visible = false;
    });
    const h = this.handlers;
    c.addEventListener('selectstart', (e) => {
      input.selecting = true;
      h.onSelectStart?.(input, e);
    });
    c.addEventListener('select', (e) => h.onSelect?.(input, e));
    c.addEventListener('selectend', (e) => {
      input.selecting = false;
      h.onSelectEnd?.(input, e);
    });
    c.addEventListener('squeezestart', (e) => {
      input.squeezing = true;
      h.onSqueezeStart?.(input, e);
    });
    c.addEventListener('squeezeend', (e) => {
      input.squeezing = false;
      h.onSqueezeEnd?.(input, e);
    });
  }

  _makeRay(controller) {
    const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, -1)]);
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.45 }));
    line.name = 'pointer-ray';
    line.scale.z = 1.2;
    line.renderOrder = 55;
    controller.add(line);
    return line;
  }

  /** Inputs currently connected (controllers and hands). */
  activeInputs() {
    return this.inputs.filter((i) => i.connected);
  }

  inputFor(handedness) {
    return this.inputs.find((i) => i.connected && i.handedness === handedness) || null;
  }

  /** Set a pointer ray's visible length (e.g. to end at the menu). */
  setRayLength(input, meters) {
    if (input.ray) input.ray.scale.z = meters;
  }

  // ------------------------------------------------------------------ rig / reference space
  /** Reset the user origin (start of a scene / session). */
  resetRig() {
    this.rigPosition.set(0, this.refSpaceType === 'local' ? 1.6 : 0, 0);
    this.rigYaw = 0;
    this._applyRig();
  }

  _applyRig() {
    if (!this.baseRefSpace) return;
    if (this.rigYaw === 0 && this.rigPosition.lengthSq() === 0) {
      // Identity rig (always the case in AR): use the tracking space itself.
      this.renderer.xr.setReferenceSpace(this.baseRefSpace);
      return;
    }
    if (typeof XRRigidTransform === 'undefined') return;
    _q.setFromAxisAngle(_up, this.rigYaw);
    _m.compose(this.rigPosition, _q, _v.set(1, 1, 1)).invert();
    const p = new THREE.Vector3();
    const q = new THREE.Quaternion();
    _m.decompose(p, q, _v);
    const offset = new XRRigidTransform({ x: p.x, y: p.y, z: p.z, w: 1 }, { x: q.x, y: q.y, z: q.z, w: q.w });
    this.renderer.xr.setReferenceSpace(this.baseRefSpace.getOffsetReferenceSpace(offset));
  }

  /**
   * Move the rig so the head (as tracked in the base space) ends up at ``anchor`` looking down -z.
   * ``vertical``: also match the height (photo viewpoints) instead of keeping the real floor.
   */
  recenterTo(anchor, { vertical = false } = {}) {
    if (!this.headBase.valid) return false;
    const yaw = this.headBase.yaw;
    const h = _v.copy(this.headBase.position);
    if (!vertical) h.y = 0;
    this.rigYaw = -yaw;
    _q.setFromAxisAngle(_up, this.rigYaw);
    h.applyQuaternion(_q);
    this.rigPosition.set(anchor.x - h.x, (vertical ? anchor.y : 0) - h.y, anchor.z - h.z);
    if (!vertical && this.refSpaceType === 'local') this.rigPosition.y = 1.6;
    this._applyRig();
    return true;
  }

  /**
   * For AR (where the rig must stay locked to the real world) return the content transform that
   * puts ``anchor`` (in content coordinates) at the head, yaw-aligned.  Null without a head pose.
   */
  contentTransformFor(anchor, { vertical = false } = {}) {
    if (!this.head.valid) return null;
    const hp = _v.copy(this.head.position);
    if (!vertical) hp.y = 0;
    const a = _v2.copy(anchor);
    if (!vertical) a.y = 0;
    const H = new THREE.Matrix4().compose(hp, new THREE.Quaternion().setFromAxisAngle(_up, this.head.yaw), new THREE.Vector3(1, 1, 1));
    const A = new THREE.Matrix4().makeTranslation(a.x, a.y, a.z);
    return H.multiply(A.invert());
  }

  /** Teleport so the head lands above ``point`` (horizontal move; keeps the floor height). */
  teleportTo(point) {
    this.rigPosition.x += point.x - this.head.position.x;
    this.rigPosition.z += point.z - this.head.position.z;
    this._applyRig();
  }

  snapTurn(angle) {
    const hw = this.head.position;
    this.rigYaw += angle;
    _q.setFromAxisAngle(_up, angle);
    _v.copy(this.rigPosition).sub(hw).applyQuaternion(_q).add(hw);
    this.rigPosition.set(_v.x, this.rigPosition.y, _v.z);
    this._applyRig();
  }

  // ------------------------------------------------------------------ teleport
  canTeleport(input) {
    return this.presenting && !this.isAR && this.locomotion.teleport && !input.isHand && input.handedness !== 'left';
  }

  beginTeleport(input) {
    this.teleport.input = input;
    this.teleport.valid = false;
    this.arc.visible = true;
  }

  /** Finish a teleport started with beginTeleport(); returns true if the user moved. */
  endTeleport(input) {
    if (this.teleport.input !== input) return false;
    const ok = this.teleport.valid;
    const point = this.teleport.point.clone();
    this.cancelTeleport();
    if (ok) this.teleportTo(point);
    return ok;
  }

  cancelTeleport() {
    this.teleport.input = null;
    this.teleport.valid = false;
    this.arc.visible = false;
    this.teleportReticle.visible = false;
  }

  _updateTeleportArc() {
    const input = this.teleport.input;
    if (!input) return;
    const origin = new THREE.Vector3();
    const dir = new THREE.Vector3();
    input.rayWorld(origin, dir);
    const pos = this.arc.geometry.attributes.position;
    const vel = dir.multiplyScalar(TELEPORT_SPEED);
    let n = 0;
    let hit = null;
    const p = new THREE.Vector3();
    const prev = origin.clone();
    const dt = 0.04;
    for (let i = 0; i < ARC_POINTS; i++) {
      const t = i * dt;
      p.set(origin.x + vel.x * t, origin.y + vel.y * t - 0.5 * GRAVITY * t * t, origin.z + vel.z * t);
      if (i > 0 && p.y <= this.floorY) {
        const f = (prev.y - this.floorY) / Math.max(1e-6, prev.y - p.y);
        hit = prev.clone().lerp(p, f);
        pos.setXYZ(n++, hit.x, hit.y, hit.z);
        break;
      }
      pos.setXYZ(n++, p.x, p.y, p.z);
      prev.copy(p);
    }
    pos.needsUpdate = true;
    this.arc.geometry.setDrawRange(0, n);
    this.arc.geometry.computeBoundingSphere();
    const valid = !!hit && Math.hypot(hit.x - this.head.position.x, hit.z - this.head.position.z) < TELEPORT_MAX_DIST;
    this.teleport.valid = valid;
    this.arc.material.color.set(valid ? 0x5ec8ff : 0xff7a7a);
    if (valid) {
      this.teleport.point.copy(hit);
      this.teleportReticle.position.copy(hit);
      this.teleportReticle.position.y += 0.005;
    }
    this.teleportReticle.visible = valid;
  }

  // ------------------------------------------------------------------ per-frame update
  /** Read head pose, buttons, sticks; apply locomotion.  Call once per XR frame. */
  update(dt, frame) {
    if (!frame || !this.session) return;
    this.frames += 1;
    const ref = this.renderer.xr.getReferenceSpace();
    const pose = ref ? frame.getViewerPose(ref) : null;
    if (pose) {
      poseToObject(pose.transform, this.head.position, this.head.quaternion);
      this.head.yaw = yawOf(this.head.quaternion);
      this.head.valid = true;
    }
    const basePose = this.baseRefSpace ? frame.getViewerPose(this.baseRefSpace) : null;
    if (basePose) {
      poseToObject(basePose.transform, this.headBase.position, this.headBase.quaternion);
      this.headBase.yaw = yawOf(this.headBase.quaternion);
      this.headBase.valid = true;
    }

    for (const input of this.inputs) {
      const gp = input.connected ? input.gamepad : null;
      if (!gp) continue;
      const pressed = gp.buttons.map((b) => !!b?.pressed);
      const edge = (i) => pressed[i] && !input.buttons[i];
      if (edge(5)) this.handlers.onMenuButton?.(input); // B (right) / Y (left)
      if (edge(4)) this.handlers.onButton?.('primary', input); // A (right) / X (left)
      if (edge(3)) this.handlers.onButton?.('stick', input);
      input.buttons = pressed;
    }

    if (!this.isAR) this._locomotion(dt);
    this._updateTeleportArc();
    this._updateWrist();
  }

  _locomotion(dt) {
    const loc = this.locomotion;
    const left = this.inputFor('left');
    const right = this.inputFor('right');
    if (loc.move) {
      const mover = left && !left.isHand ? left : (!right && this.inputs.find((i) => i.connected && !i.isHand)) || null;
      if (mover) {
        const { x, y } = mover.stick();
        const mag = Math.hypot(x, y);
        if (mag > DEADZONE) {
          const k = ((mag - DEADZONE) / (1 - DEADZONE)) ** 2 * MOVE_SPEED * dt / mag;
          const fwd = forwardFromYaw(this.head.yaw, _v);
          const rightV = _v2.set(-fwd.z, 0, fwd.x);
          this.rigPosition.addScaledVector(fwd, -y * k).addScaledVector(rightV, x * k);
          this._applyRig();
        }
      }
    }
    if (loc.turn && right && !right.isHand) {
      const { x } = right.stick();
      if (this.snapArmed && Math.abs(x) > 0.6) {
        this.snapTurn(x > 0 ? -SNAP_ANGLE : SNAP_ANGLE);
        this.snapArmed = false;
      } else if (Math.abs(x) < 0.3) {
        this.snapArmed = true;
      }
    }
  }

  _updateWrist() {
    const btn = this.wrist.mesh;
    const left = this.inputs.find((i) => i.connected && i.isHand && i.handedness === 'left');
    const right = this.inputs.find((i) => i.connected && i.isHand && i.handedness === 'right');
    const wrist = left?.hand?.joints?.wrist;
    if (!wrist || !wrist.visible) {
      btn.visible = false;
      return;
    }
    wrist.updateMatrixWorld();
    const wp = new THREE.Vector3().setFromMatrixPosition(wrist.matrixWorld);
    const wq = new THREE.Quaternion().setFromRotationMatrix(_m.extractRotation(wrist.matrixWorld));
    // WebXR hand joints: -Y points out of the palm.  Show the button when the palm faces the face.
    const palm = new THREE.Vector3(0, -1, 0).applyQuaternion(wq);
    const toHead = _v.copy(this.head.position).sub(wp).normalize();
    const facing = palm.dot(toHead) > 0.55;
    btn.visible = facing;
    if (!facing) return;
    btn.position.copy(wp).addScaledVector(palm, 0.03).addScaledVector(new THREE.Vector3(0, 0, -1).applyQuaternion(wq), 0.03);
    btn.lookAt(this.head.position);
    const tip = right?.hand?.joints?.['index-finger-tip'];
    if (tip && tip.visible) {
      const tp = new THREE.Vector3().setFromMatrixPosition(tip.matrixWorld);
      const now = performance.now();
      if (tp.distanceTo(btn.position) < 0.025 && now - this.wrist.lastToggle > 800) {
        this.wrist.lastToggle = now;
        this.handlers.onMenuButton?.(right);
      }
    }
  }
}
