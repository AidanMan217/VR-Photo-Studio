// Neutral VR environment pieces: gradient sky dome, subtle floor grid, soft shadow blob, reticle.
// Everything is unlit and cheap (no real-time lights or shadows) to keep Quest at 72-90 fps.

import * as THREE from 'three';

/** Gradient sky dome (vertex colours, no shader compilation surprises). */
export function createSkyDome({ radius = 60, top = '#3a4a66', horizon = '#a9b6c8', bottom = '#2a2d33' } = {}) {
  const geo = new THREE.SphereGeometry(radius, 32, 16);
  const cTop = new THREE.Color(top);
  const cHor = new THREE.Color(horizon);
  const cBot = new THREE.Color(bottom);
  const pos = geo.attributes.position;
  const colors = new Float32Array(pos.count * 3);
  const c = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    const t = pos.getY(i) / radius; // -1 .. 1
    if (t >= 0) c.copy(cHor).lerp(cTop, Math.pow(t, 0.6));
    else c.copy(cHor).lerp(cBot, Math.min(1, Math.pow(-t, 0.4)));
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const mat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide, depthWrite: false, fog: false });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.name = 'sky';
  mesh.renderOrder = -10;
  mesh.frustumCulled = false;
  return mesh;
}

function radialTexture(stops, size = 256) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const g = canvas.getContext('2d');
  const grad = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  for (const [t, color] of stops) grad.addColorStop(t, color);
  g.fillStyle = grad;
  g.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/** A soft floor disc with a faint grid that fades out with distance. */
export function createFloor({ radius = 12, color = '#8d97a8' } = {}) {
  const group = new THREE.Group();
  group.name = 'floor';
  const disc = new THREE.Mesh(
    new THREE.CircleGeometry(radius, 64),
    new THREE.MeshBasicMaterial({
      map: radialTexture([[0, 'rgba(70,76,88,0.95)'], [0.55, 'rgba(62,67,78,0.75)'], [1, 'rgba(55,60,70,0)']]),
      transparent: true,
      depthWrite: false,
    }),
  );
  disc.rotation.x = -Math.PI / 2;
  disc.renderOrder = -5;
  group.add(disc);

  const grid = new THREE.GridHelper(radius * 2, radius * 4, color, color);
  const gm = grid.material;
  gm.transparent = true;
  gm.opacity = 0.22;
  gm.depthWrite = false;
  grid.position.y = 0.001;
  grid.renderOrder = -4;
  group.add(grid);
  return group;
}

/** Soft circular contact shadow (a transparent radial gradient, no shadow maps). */
export function createShadowBlob(radius = 0.5, opacity = 0.5) {
  const mesh = new THREE.Mesh(
    new THREE.CircleGeometry(1, 48),
    new THREE.MeshBasicMaterial({
      map: radialTexture([[0, `rgba(0,0,0,${opacity})`], [0.55, `rgba(0,0,0,${opacity * 0.6})`], [1, 'rgba(0,0,0,0)']], 128),
      transparent: true,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -1,
    }),
  );
  mesh.name = 'shadow-blob';
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = 0.003;
  mesh.scale.setScalar(radius);
  // Draw before other transparent things (e.g. Gaussian splats, which do not write depth) so
  // the shadow never paints over the object standing on it.
  mesh.renderOrder = -1;
  return mesh;
}

/** Floor reticle (ring + dot) used for AR placement and teleport targets. */
export function createReticle(color = '#ffffff') {
  const group = new THREE.Group();
  group.name = 'reticle';
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.11, 0.14, 48).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, depthTest: false }),
  );
  const dot = new THREE.Mesh(
    new THREE.CircleGeometry(0.025, 24).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, depthTest: false }),
  );
  ring.renderOrder = dot.renderOrder = 50;
  group.add(ring, dot);
  group.visible = false;
  group.matrixAutoUpdate = true;
  return group;
}

/**
 * A complete neutral environment for VR modes.  ``setVisible(false)`` hides it (used for AR
 * passthrough).  Returns ``{ group, setVisible, dispose }``.
 */
export function createEnvironment({ sky = true, floor = true, background } = {}) {
  const group = new THREE.Group();
  group.name = 'environment';
  if (sky) group.add(createSkyDome(background ? { horizon: background, top: background, bottom: background } : {}));
  if (floor) group.add(createFloor());
  return {
    group,
    setVisible(v) {
      group.visible = v;
    },
  };
}
