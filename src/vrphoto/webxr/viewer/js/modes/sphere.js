// sphere: VR180 / 360 photos and videos on an inverted sphere around the viewer, mono or stereo
// (side-by-side / top-bottom, per-eye UVs on camera layers 1/2).  An optional depth.png pushes
// each vertex to its estimated distance, giving a little real parallax when you move your head.
// Equirect convention (docs/CONTRACTS.md E): image centre = straight ahead (-z), left edge of a
// 180 image = -x, image top = up.

import * as THREE from 'three';
import { backgroundColor } from '../manifest.js';
import { disposeVideo, loadEyeTextures, loadGrayscale, loadVideo } from '../loaders.js';
import { videoControls } from '../media.js';
import { eyeWindows } from './stereo.js';
import { log } from '../util.js';

const CENTER_Y = 1.6; // sphere centre height; the head is recentered here

/**
 * Inverted sphere covering ``coverage`` degrees of longitude, centred on -z.
 * With phiStart = pi (180) or pi/2 (360) and the x-flip, texture u = 0.5 maps to -z.
 */
export function sphereGeometry(radius, coverage, widthSegments, heightSegments) {
  const phiStart = coverage === 180 ? Math.PI : Math.PI / 2;
  const phiLength = coverage === 180 ? Math.PI : Math.PI * 2;
  const geo = new THREE.SphereGeometry(radius, widthSegments, heightSegments, phiStart, phiLength);
  geo.scale(-1, 1, 1); // look at the inside, un-mirrored
  return geo;
}

/** Copy of ``geo`` with UVs remapped into the window [u0,u1]x[v0,v1]. */
function remapUVs(geo, [u0, u1, v0, v1]) {
  const g = geo.clone();
  const uv = g.attributes.uv;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, u0 + uv.getX(i) * (u1 - u0), v0 + uv.getY(i) * (v1 - v0));
  uv.needsUpdate = true;
  return g;
}

/**
 * Displace sphere vertices radially by a disparity map (1 = near).  ``depth`` is sampled at
 * the vertex grid resolution (cols = widthSegments + 1, rows = heightSegments + 1).
 */
export function displaceByDepth(geo, depth, cols, rows, { near, far, radius, wrap }) {
  const pos = geo.attributes.position;
  const uv = geo.attributes.uv;
  const invNear = 1 / near;
  const invFar = 1 / far;
  const dir = new THREE.Vector3();
  // Pole rows collapse to one point: use the row average there so the pole stays closed.
  const rowAvg = (r) => {
    let s = 0;
    for (let c = 0; c < cols; c++) s += depth[r * cols + c];
    return s / cols;
  };
  const top = rowAvg(0);
  const bottom = rowAvg(rows - 1);
  for (let i = 0; i < pos.count; i++) {
    const u = uv.getX(i);
    const v = 1 - uv.getY(i);
    const c = Math.min(cols - 1, Math.max(0, Math.round(u * (cols - 1))));
    const r = Math.min(rows - 1, Math.max(0, Math.round(v * (rows - 1))));
    let d;
    if (r === 0) d = top;
    else if (r === rows - 1) d = bottom;
    else if (wrap && (c === 0 || c === cols - 1)) d = 0.5 * (depth[r * cols] + depth[r * cols + cols - 1]);
    else d = depth[r * cols + c];
    const dist = Math.min(radius, 1 / (invFar + Math.min(1, Math.max(0, d)) * (invNear - invFar)));
    dir.fromBufferAttribute(pos, i).normalize();
    pos.setXYZ(i, dir.x * dist, dir.y * dist, dir.z * dist);
  }
  pos.needsUpdate = true;
  geo.computeBoundingSphere();
}

export async function load(ctx, scene) {
  const cfg = scene.sphere;
  const layout = cfg.layout;
  const stereo = layout === 'sbs' || layout === 'tb';
  const coverage = cfg.coverage;
  const radius = cfg.radius_m;
  const withDepth = typeof cfg.depth === 'string' && cfg.depth.length > 0;
  const wSeg = coverage === 180 ? (withDepth ? 128 : 64) : withDepth ? 192 : 96;
  const hSeg = withDepth ? 96 : 48;

  const group = new THREE.Group();
  group.name = 'sphere';
  group.position.set(0, CENTER_Y, 0);
  ctx.root.add(group);

  const base = sphereGeometry(radius, coverage, wSeg, hSeg);
  if (withDepth) {
    try {
      ctx.progress(0.02, 'Loading depth…');
      const depth = await loadGrayscale(ctx.asset(cfg.depth), wSeg + 1, hSeg + 1, { signal: ctx.signal });
      displaceByDepth(base, depth, wSeg + 1, hSeg + 1, {
        near: cfg.depth_near_m,
        far: cfg.depth_far_m || radius,
        radius,
        wrap: coverage === 360,
      });
    } catch (err) {
      if (err?.name === 'AbortError') throw err;
      log.warn('depth map unavailable, showing a flat sphere', err);
      ctx.toast('Depth map missing - showing the image without parallax');
    }
  }
  ctx.checkAborted();

  let video = null;
  let videoTex = null;
  const meshes = [];
  if (cfg.kind === 'video') {
    ctx.progress(-1, 'Opening video…');
    const v = await loadVideo(ctx.asset(cfg.media), { signal: ctx.signal });
    video = v.video;
    videoTex = v.texture;
    if (ctx.signal.aborted) disposeVideo(video, videoTex); // superseded: stop it (checkAborted throws below)
    const mat = new THREE.MeshBasicMaterial({ map: videoTex, toneMapped: false });
    const win = eyeWindows(layout);
    meshes.push(new THREE.Mesh(remapUVs(base, win.left), mat));
    if (stereo) meshes.push(new THREE.Mesh(remapUVs(base, win.right), mat));
    base.dispose();
  } else {
    const tex = await loadEyeTextures(ctx.asset(cfg.media), layout, {
      signal: ctx.signal,
      onProgress: (f) => ctx.progress(f < 0 ? -1 : 0.05 + f * 0.9, coverage === 180 ? 'Loading VR180 photo…' : 'Loading 360° photo…'),
    });
    meshes.push(new THREE.Mesh(base, new THREE.MeshBasicMaterial({ map: tex.left, toneMapped: false })));
    if (stereo) meshes.push(new THREE.Mesh(base, new THREE.MeshBasicMaterial({ map: tex.right, toneMapped: false })));
  }
  ctx.checkAborted();
  if (stereo) {
    meshes[0].layers.set(1);
    meshes[1].layers.set(2);
  }
  meshes.forEach((m, i) => {
    m.name = i === 0 ? 'sphere-left' : 'sphere-right';
    m.frustumCulled = false;
    group.add(m);
  });

  const controls = video ? videoControls(video, ctx) : null;
  const center = new THREE.Vector3(0, CENTER_Y, 0);
  const maxYaw = coverage === 180 ? THREE.MathUtils.degToRad(100) : Infinity;
  return {
    background: backgroundColor(scene.env, '#000000'),
    supportsAR: false,
    locomotion: { move: false, turn: true, teleport: false },
    recenterAnchor: { position: center, vertical: true },
    desktopView: {
      position: center.clone(),
      target: new THREE.Vector3(0, CENTER_Y, -0.01),
      lookAround: true,
      fovZoom: true,
      minAzimuthAngle: -maxYaw,
      maxAzimuthAngle: maxYaw,
      minPolarAngle: 0.15,
      maxPolarAngle: Math.PI - 0.15,
      fov: 75,
    },
    media: video ? { video } : null,
    menuItems: controls ? controls.menuItems : [],
    onUserGesture: controls ? controls.onUserGesture : undefined,
    onButton: controls ? controls.onButton : undefined,
    update() {},
    dispose() {
      if (video) disposeVideo(video, videoTex);
    },
    debug: () => ({ coverage, layout, stereo, depth: withDepth, radius, kind: cfg.kind }),
  };
}
