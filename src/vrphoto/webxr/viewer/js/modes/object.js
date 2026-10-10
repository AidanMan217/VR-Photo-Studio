// object: a person / object (GLB mesh or Gaussian splat) at real-world size, standing on the floor.
//  * AR (Quest 3 passthrough): a hit-test reticle follows the floor where you point; pull the
//    trigger (or pinch) to place it.  It stays put (WebXR anchors when available) while you walk
//    around it.  Pull the trigger on the floor again to move it there.
//  * VR: a neutral room (soft sky + floor grid) with the object distance_m in front of you.
//  * Both: grip (squeeze) to grab & move + turn it, both grips to resize, thumbstick left/right
//    to rotate, menu "Reset 1:1 scale".  Hands: pinch on the object to grab it.

import * as THREE from 'three';
import { backgroundColor } from '../manifest.js';
import { createEnvironment, createReticle, createShadowBlob } from '../env.js';
import {
  createSparkRenderer,
  loadGLB,
  loadSplatMesh,
  makeUnlit,
  splatBounds,
  upAxisQuaternion,
} from '../loaders.js';
import { clamp, log } from '../util.js';

const ROTATE_SPEED = 1.8; // rad/s at full stick
const MIN_SCALE = 0.05;
const MAX_SCALE = 20;
const CENTER_PIVOT_HEIGHT = 1.2; // pivot "center": float the object's centre at this height

const _v = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _m = new THREE.Matrix4();

function controllerPose(input, pos) {
  input.controller.updateMatrixWorld();
  pos.setFromMatrixPosition(input.controller.matrixWorld);
  _q.setFromRotationMatrix(_m.extractRotation(input.controller.matrixWorld));
  const f = _v.set(0, 0, -1).applyQuaternion(_q);
  return Math.atan2(-f.x, -f.z);
}

const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));

export async function load(ctx, scene) {
  const cfg = scene.object;
  const isGLB = cfg.format === 'glb';

  // content (placed in the world, user yaw/scale) -> model (normalisation scale) -> orient (up axis + pivot offset) -> asset
  const content = new THREE.Group();
  content.name = 'object';
  const model = new THREE.Group();
  const orient = new THREE.Group();
  orient.quaternion.copy(upAxisQuaternion(cfg.up));
  content.add(model);
  model.add(orient);

  let asset;
  let splat = null;
  if (isGLB) {
    const gltf = await loadGLB(ctx.asset(cfg.asset), {
      signal: ctx.signal,
      onProgress: (f) => ctx.progress(f < 0 ? -1 : f * 0.95, 'Loading 3D object…'),
    });
    makeUnlit(gltf.scene, { anisotropy: Math.min(4, ctx.renderer.capabilities.getMaxAnisotropy()) });
    asset = gltf.scene;
  } else {
    ctx.progress(0.01, 'Starting the splat renderer…');
    const spark = await createSparkRenderer(ctx.renderer);
    ctx.addToScene(spark);
    splat = await loadSplatMesh(ctx.asset(cfg.asset), cfg.format, {
      signal: ctx.signal,
      onProgress: (f) => ctx.progress(f < 0 ? -1 : f * 0.95, 'Loading splat object…'),
    });
    asset = splat;
  }
  ctx.checkAborted();
  orient.add(asset);

  // ---------------------------------------------------------------- normalise size + pivot
  orient.updateMatrix();
  orient.updateMatrixWorld(true);
  let box;
  if (isGLB) {
    box = new THREE.Box3().setFromObject(orient, true);
  } else {
    box = splatBounds(splat).clone().applyMatrix4(orient.matrix);
  }
  if (box.isEmpty()) box = new THREE.Box3(new THREE.Vector3(-0.5, 0, -0.5), new THREE.Vector3(0.5, 1, 0.5));
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const nativeHeight = Math.max(size.y, 1e-4);
  const fitScale = cfg.height_m ? cfg.height_m / nativeHeight : 1;
  model.scale.setScalar(fitScale);
  const base = cfg.pivot !== 'center';
  orient.position.set(-center.x, base ? -box.min.y : -center.y, -center.z);
  if (!base) model.position.y = CENTER_PIVOT_HEIGHT;
  const realHeight = nativeHeight * fitScale;
  const footprint = Math.max(size.x, size.z) * fitScale;
  // Content-space bounds (before user scale), used for hand picking.
  const halfX = (size.x / 2) * fitScale;
  const halfZ = (size.z / 2) * fitScale;
  const yMin = base ? 0 : CENTER_PIVOT_HEIGHT - realHeight / 2;
  const localBox = new THREE.Box3(
    new THREE.Vector3(-halfX, yMin, -halfZ),
    new THREE.Vector3(halfX, yMin + realHeight, halfZ),
  ).expandByScalar(0.05);

  if (base) content.add(createShadowBlob(Math.max(0.2, footprint * 0.6), 0.45));
  ctx.root.add(content);

  // ---------------------------------------------------------------- VR environment + reticle
  const env = createEnvironment({ sky: true, floor: true });
  ctx.root.add(env.group);
  const reticle = createReticle('#ffffff');
  ctx.root.add(reticle);

  const state = {
    userScale: 1,
    placed: true,
    placedBy: 'front-floor',
    anchor: null,
    anchorPending: false,
    wantAnchor: false,
    hitSources: new Map(), // XRInputSource|'viewer' -> XRHitTestSource
    hitRequests: new Set(),
    hitFailed: new Set(), // sources the runtime refused (do not retry every frame)
    lastHit: null, // { position, result, source }
    lastInput: null,
    grabs: new Map(), // input -> grab record
    twoHand: null,
    hitTestAvailable: false,
    pendingFront: false,
    selects: 0,
    disposed: false, // set by dispose(): late async results (anchors, hit-test sources) are released
  };

  function placeAt(pos, yaw) {
    content.position.copy(pos);
    content.rotation.set(0, yaw, 0);
    content.visible = true;
  }

  function placeFrontFloor(head) {
    const yaw = head ? head.yaw : 0;
    const fwd = new THREE.Vector3(-Math.sin(yaw), 0, -Math.cos(yaw));
    const p = head ? head.position.clone().setY(0).addScaledVector(fwd, cfg.distance_m) : new THREE.Vector3(0, 0, -cfg.distance_m);
    placeAt(p, yaw);
    state.placed = true;
    state.placedBy = 'front-floor';
  }

  function faceYaw(pos) {
    const head = ctx.xr.head;
    return Math.atan2(head.position.x - pos.x, head.position.z - pos.z);
  }

  function dropAnchor() {
    if (state.anchor) {
      try {
        state.anchor.delete?.();
      } catch (err) {
        log.warn('anchor delete', err);
      }
    }
    state.anchor = null;
  }

  /** Anchor the object where it is now (must run inside an XR frame callback). */
  async function anchorHere(frame) {
    state.wantAnchor = false;
    if (!ctx.xr.hasFeature('anchors') || state.anchorPending || !frame?.createAnchor) return;
    if (typeof XRRigidTransform === 'undefined') return;
    state.anchorPending = true;
    const p = content.position;
    const session = ctx.xr.session;
    try {
      // In AR the rig is identity, so the tracking space equals the scene space.
      const anchor = await frame.createAnchor(
        new XRRigidTransform({ x: p.x, y: p.y, z: p.z, w: 1 }, { x: 0, y: 0, z: 0, w: 1 }),
        ctx.xr.baseRefSpace,
      );
      if (anchor && !state.disposed && ctx.xr.session === session && state.grabs.size === 0) {
        dropAnchor();
        state.anchor = anchor;
      } else {
        anchor?.delete?.();
      }
    } catch (err) {
      log.info('anchor not created', err?.message || err);
    } finally {
      state.anchorPending = false;
    }
  }

  // ---------------------------------------------------------------- hit testing (AR)
  function ensureHitSources(session) {
    if (!ctx.xr.hasFeature('hit-test') || typeof session.requestHitTestSource !== 'function') return;
    state.hitTestAvailable = true;
    const want = new Map();
    for (const input of ctx.xr.activeInputs()) {
      const src = input.inputSource;
      if (src && src.targetRayMode === 'tracked-pointer') want.set(src, src.targetRaySpace);
    }
    // Drop sources of controllers / hands that went away (e.g. switching to hand tracking).
    for (const [key, source] of state.hitSources) {
      if (key !== 'viewer' && !want.has(key)) {
        try {
          source.cancel();
        } catch {
          /* already gone */
        }
        state.hitSources.delete(key);
      }
    }
    const request = (key, getSpace) => {
      if (state.hitSources.has(key) || state.hitRequests.has(key) || state.hitFailed.has(key)) return;
      state.hitRequests.add(key);
      Promise.resolve()
        .then(getSpace)
        .then((space) => session.requestHitTestSource({ space }))
        .then((source) => {
          if (!state.disposed && ctx.xr.session === session) state.hitSources.set(key, source);
          else source.cancel();
        })
        .catch((err) => {
          state.hitFailed.add(key);
          log.info('hit-test source unavailable', err?.message || err);
        })
        .finally(() => state.hitRequests.delete(key));
    };
    for (const [key, space] of want) request(key, () => space);
    request('viewer', () => session.requestReferenceSpace('viewer'));
  }

  function cancelHitSources() {
    for (const s of state.hitSources.values()) {
      try {
        s.cancel();
      } catch {
        /* session already ended */
      }
    }
    state.hitSources.clear();
    state.hitRequests.clear();
    state.hitFailed.clear();
    state.lastHit = null;
  }

  function hitFor(frame, key, refSpace) {
    const source = state.hitSources.get(key);
    if (!source) return null;
    const results = frame.getHitTestResults(source);
    for (const result of results) {
      const pose = result.getPose(refSpace);
      if (!pose) continue;
      const o = pose.transform.orientation;
      const normal = _v.set(0, 1, 0).applyQuaternion(_q.set(o.x, o.y, o.z, o.w));
      if (normal.y < 0.6) continue; // only horizontal surfaces (floor, tables)
      const p = pose.transform.position;
      return { position: new THREE.Vector3(p.x, p.y, p.z), result, key };
    }
    return null;
  }

  function updateHitTest(frame) {
    const session = ctx.xr.session;
    ensureHitSources(session);
    const ref = ctx.renderer.xr.getReferenceSpace();
    const preferred = [];
    if (state.lastInput?.inputSource) preferred.push(state.lastInput.inputSource);
    const right = ctx.xr.inputFor('right');
    if (right?.inputSource) preferred.push(right.inputSource);
    for (const i of ctx.xr.activeInputs()) if (i.inputSource) preferred.push(i.inputSource);
    preferred.push('viewer');
    let hit = null;
    for (const key of preferred) {
      hit = hitFor(frame, key, ref);
      if (hit) break;
    }
    state.lastHit = hit;
    const showReticle = !!hit && state.grabs.size === 0;
    reticle.visible = showReticle;
    if (showReticle) reticle.position.copy(hit.position);
    if (!state.placed) {
      content.visible = !!hit;
      if (hit) placeAt(hit.position, faceYaw(hit.position));
    }
  }

  // ---------------------------------------------------------------- grabbing
  function grabStart(input) {
    if (!content.visible) return false;
    const pos = new THREE.Vector3();
    const yaw = controllerPose(input, pos);
    state.grabs.set(input, { startCtrl: pos, startYaw: yaw, startPos: content.position.clone(), startObjYaw: content.rotation.y });
    if (state.grabs.size >= 2) startTwoHand();
    dropAnchor();
    reticle.visible = false;
    return true;
  }

  function startTwoHand() {
    const [a, b] = [...state.grabs.keys()];
    const pa = new THREE.Vector3();
    const pb = new THREE.Vector3();
    controllerPose(a, pa);
    controllerPose(b, pb);
    state.twoHand = { d0: Math.max(0.02, pa.distanceTo(pb)), s0: state.userScale };
  }

  function grabEnd(input) {
    if (!state.grabs.delete(input)) return;
    state.twoHand = null;
    if (state.grabs.size === 1) {
      // Re-base the remaining hand so the object does not jump.
      const [other] = state.grabs.keys();
      state.grabs.delete(other);
      grabStart(other);
    }
    if (state.grabs.size === 0) {
      state.placed = true;
      state.placedBy = 'grab';
      if (ctx.isAR) state.wantAnchor = true;
    }
  }

  function updateGrab() {
    if (state.grabs.size === 0) return;
    if (state.grabs.size >= 2 && state.twoHand) {
      const [a, b] = [...state.grabs.keys()];
      const pa = new THREE.Vector3();
      const pb = new THREE.Vector3();
      controllerPose(a, pa);
      controllerPose(b, pb);
      state.userScale = clamp((state.twoHand.s0 * pa.distanceTo(pb)) / state.twoHand.d0, MIN_SCALE, MAX_SCALE);
      content.scale.setScalar(state.userScale);
      return;
    }
    const [input, g] = [...state.grabs.entries()][0];
    const pos = new THREE.Vector3();
    const yaw = controllerPose(input, pos);
    content.position.set(g.startPos.x + (pos.x - g.startCtrl.x), g.startPos.y, g.startPos.z + (pos.z - g.startCtrl.z));
    content.rotation.y = g.startObjYaw + wrapAngle(yaw - g.startYaw);
  }

  /** Does the input's ray (or the hand itself) touch the object? */
  function touches(input) {
    if (!content.visible) return false;
    content.updateMatrixWorld(true);
    const inv = _m.copy(content.matrixWorld).invert();
    const origin = new THREE.Vector3();
    const dir = new THREE.Vector3();
    input.rayWorld(origin, dir);
    const ray = new THREE.Ray(origin.clone().applyMatrix4(inv), dir.clone().transformDirection(inv));
    const hit = ray.intersectBox(localBox, new THREE.Vector3());
    if (!hit) return false;
    return hit.applyMatrix4(content.matrixWorld).distanceTo(origin) < 6;
  }

  function rotateWithSticks(dt) {
    for (const input of ctx.xr.activeInputs()) {
      if (input.isHand) continue;
      // In VR the left stick walks; the right stick turns the object.  In AR any stick turns it.
      if (!ctx.isAR && input.handedness === 'left') continue;
      const { x } = input.stick();
      if (Math.abs(x) > 0.2) content.rotation.y -= x * ROTATE_SPEED * dt;
    }
  }

  function resetScale() {
    state.userScale = 1;
    content.scale.setScalar(1);
    ctx.toast(`Real size: ${realHeight.toFixed(2)} m tall`);
  }

  placeFrontFloor(null);

  const background = backgroundColor(scene.env, '#2a2d33');
  return {
    background,
    supportsAR: true,
    heavy: !isGLB,
    locomotion: (ar) => (ar ? { move: false, turn: false, teleport: false } : { move: true, turn: false, teleport: true }),
    floorY: 0,
    recenterAnchor: { position: new THREE.Vector3(0, 1.6, 0), vertical: false },
    desktopView: {
      // Far enough back to see the whole object standing distance_m in front of the origin.
      position: new THREE.Vector3(0, Math.min(1.6, realHeight * 0.75 + 0.4), Math.max(0.5, realHeight * 1.1 - cfg.distance_m)),
      target: new THREE.Vector3(0, base ? realHeight * 0.5 : CENTER_PIVOT_HEIGHT, -cfg.distance_m),
      minDistance: 0.3,
      maxDistance: 12,
      minPolarAngle: 0.1,
      maxPolarAngle: Math.PI / 2 + 0.08,
    },
    menuItems: [
      { id: 'reset-scale', label: 'Reset 1:1 scale', action: resetScale },
      {
        id: 'place-again',
        label: () => (ctx.isAR ? 'Place again' : 'Put back in front'),
        action: () => {
          if (ctx.isAR && state.hitTestAvailable) {
            dropAnchor();
            state.placed = false;
            ctx.toast('Point at the floor and pull the trigger');
          } else {
            placeFrontFloor(ctx.xr.head.valid ? ctx.xr.head : null);
          }
        },
      },
    ],
    onSessionStart({ ar }) {
      state.grabs.clear();
      state.twoHand = null;
      dropAnchor();
      if (ar) {
        env.setVisible(false);
        const wantHitTest = cfg.placement !== 'front-floor' && ctx.xr.hasFeature('hit-test');
        if (wantHitTest) {
          state.placed = false;
          content.visible = false;
          ctx.toast('Point at the floor and pull the trigger (or pinch) to place', 5000);
        } else {
          state.pendingFront = true;
          if (cfg.placement !== 'front-floor') ctx.toast('Placed in front of you - squeeze the grip to move it', 4000);
        }
      } else {
        env.setVisible(true);
        placeFrontFloor(null);
      }
    },
    onSessionEnd() {
      cancelHitSources();
      dropAnchor();
      state.grabs.clear();
      state.twoHand = null;
      reticle.visible = false;
      env.setVisible(true);
      placeFrontFloor(null);
      content.scale.setScalar(state.userScale);
    },
    onPassthroughChange(on) {
      env.setVisible(!on);
    },
    onRecenter(head, { auto = false } = {}) {
      if (ctx.isAR) {
        // Automatic recenter (session start, headset recenter) must not undo a placement.
        if (auto) return true;
        dropAnchor();
        placeFrontFloor(head);
        return true;
      }
      if (!auto) placeFrontFloor(null);
      return false; // the app moves the user back to the start, facing the object
    },
    onSelectStart(input) {
      state.lastInput = input;
      if (input.isHand && touches(input)) return grabStart(input);
      return false;
    },
    onSelectEnd(input) {
      if (input.isHand && state.grabs.has(input)) grabEnd(input);
    },
    onSelect() {
      state.selects += 1;
      if (!ctx.isAR || state.grabs.size) return;
      // lastHit comes from the most recently used input (see updateHitTest).
      const hit = state.lastHit;
      if (!hit) {
        if (!state.placed) ctx.toast('Point at the floor first');
        return;
      }
      dropAnchor();
      placeAt(hit.position, faceYaw(hit.position));
      state.placed = true;
      state.placedBy = 'hit-test';
      state.wantAnchor = true;
      ctx.toast('Placed! Walk around it. Grip to move, both grips to resize.', 3500);
    },
    onSqueezeStart(input) {
      state.lastInput = input;
      grabStart(input);
    },
    onSqueezeEnd(input) {
      grabEnd(input);
    },
    update(dt, frame) {
      if (!frame || !ctx.presenting) return;
      if (state.pendingFront && ctx.xr.head.valid) {
        state.pendingFront = false;
        placeFrontFloor(ctx.xr.head);
      }
      if (ctx.isAR) updateHitTest(frame);
      updateGrab();
      if (state.grabs.size === 0) rotateWithSticks(dt);
      if (state.wantAnchor && state.grabs.size === 0 && ctx.isAR) anchorHere(frame);
      if (state.anchor && state.grabs.size === 0) {
        let pose = null;
        try {
          pose = frame.getPose(state.anchor.anchorSpace, ctx.xr.baseRefSpace);
        } catch (err) {
          log.info('anchor lost', err?.message || err);
          dropAnchor();
        }
        const p = pose?.transform.position;
        if (p && Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z)) {
          content.position.set(p.x, p.y, p.z);
        }
      }
    },
    dispose() {
      state.disposed = true;
      cancelHitSources();
      dropAnchor();
    },
    debug: () => ({
      format: cfg.format,
      placed: state.placed,
      placedBy: state.placedBy,
      visible: content.visible,
      position: content.position.toArray().map((v) => Math.round(v * 1000) / 1000),
      yaw: content.rotation.y,
      scale: state.userScale,
      realHeight,
      hitTest: state.hitSources.size > 0,
      reticle: reticle.visible,
      anchored: !!state.anchor,
      grabs: state.grabs.size,
      selects: state.selects,
    }),
  };
}
