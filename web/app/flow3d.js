'use client';
// Injector → producer connectivity in real 3D (three.js), drawn on its own transparent canvas over the rock layers.
// The camera is rebuilt every frame from the same yaw / pitch / distance / focal length the SVG layers use, so the
// tubes line up exactly with the wells and the reservoir drawn around them.
import * as THREE from 'three';

const GLASS_VS = `
  varying vec3 vN; varying vec3 vV; varying vec2 vUv;
  void main() {
    vUv = uv;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz);
    gl_Position = projectionMatrix * mv;
  }`;
// Glass: almost clear facing you, bright at the rim (fresnel), with a faint moving sheen.
const GLASS_FS = `
  uniform vec3 uColor; uniform float uOpacity; uniform float uTime;
  varying vec3 vN; varying vec3 vV; varying vec2 vUv;
  void main() {
    float fr = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.4);
    float sheen = 0.5 + 0.5 * sin(vUv.x * 40.0 - uTime * 2.0);
    vec3 rim = mix(uColor, vec3(0.06, 0.24, 0.46), fr);
    float spec = pow(max(0.0, 1.0 - abs(vUv.y - 0.25) * 9.0), 3.0) * 0.55;   // a highlight running along the top
    gl_FragColor = vec4(rim + spec, (0.14 + 0.7 * fr + 0.05 * sheen + spec * 0.6) * uOpacity);
  }`;
// Water core: bands of light travel from injector to producer; faster and brighter for larger shares.
const WATER_FS = `
  uniform vec3 uColor; uniform float uOpacity; uniform float uTime; uniform float uSpeed; uniform float uRepeat;
  varying vec3 vN; varying vec3 vV; varying vec2 vUv;
  void main() {
    float x = fract(vUv.x * uRepeat - uTime * uSpeed);
    float band = smoothstep(0.0, 0.18, x) * (1.0 - smoothstep(0.3, 0.62, x));
    float ripple = 0.5 + 0.5 * sin(vUv.y * 6.2832 * 3.0 + vUv.x * uRepeat * 9.0 - uTime * uSpeed * 9.0);
    float fr = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 1.6);
    vec3 col = mix(uColor, vec3(0.82, 0.94, 1.0), band * 0.6 + ripple * 0.08) + fr * 0.22;
    gl_FragColor = vec4(col, (0.9 + 0.1 * band) * uOpacity);
  }`;
// Soft additive glow (halo and orbs): strongest through the middle, fading to the edge.
const GLOW_FS = `
  uniform vec3 uColor; uniform float uOpacity;
  varying vec3 vN; varying vec3 vV; varying vec2 vUv;
  void main() {
    float core = pow(abs(dot(normalize(vN), normalize(vV))), 1.8);
    gl_FragColor = vec4(mix(vec3(1.0), uColor, 0.35 + 0.65 * core), (0.15 + 0.85 * core) * uOpacity);
  }`;

const hexColor = h => new THREE.Color(h);
const water = t => new THREE.Color('#a8d4f5').lerp(new THREE.Color('#0a3a8a'), Math.max(0, Math.min(1, t)));

function material(fs, uniforms, {additive = false} = {}) {
  return new THREE.ShaderMaterial({
    vertexShader: GLASS_VS, fragmentShader: fs, transparent: true, depthWrite: false,
    blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
    uniforms: Object.fromEntries(Object.entries(uniforms).map(([k, v]) => [k, {value: v}])),
  });
}

/** Start the WebGL layer. Returns a cleanup function. Does nothing (and the SVG view still works) without WebGL. */
export function startConnectivity3D(canvas, sceneRef) {
  if (!canvas) return () => {};
  let renderer;
  try { renderer = new THREE.WebGLRenderer({canvas, alpha: true, antialias: true}); } catch { return () => {}; }
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 1, 1e7);
  const root = new THREE.Group();
  scene.add(root);
  const still = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  let key = '', items = [], wells = [], raf = 0, size = '';
  const t0 = performance.now();

  function clear() {
    root.traverse(o => { o.geometry?.dispose(); o.material?.dispose?.(); });
    root.clear();
    items = []; wells = [];
  }

  function build(S) {
    clear();
    if (!S.on || !S.flows?.channels.length) return;
    const {cx, cy, tz} = S.cam;
    const rel = (x, y, z) => new THREE.Vector3(x - cx, y - cy, z - tz);
    const R = S.half * 0.058;                                  // glass tube radius (world m)
    for (const ch of S.flows.channels) {
      const share = ch.f / S.flows.maxF, t = Math.pow(share, 1.4);
      const zA = S.zOf(ch.inj), zB = S.zOf(ch.prod);
      const pts = Array.from({length: 13}, (_, k) => {
        const u = k / 12;
        return rel(ch.inj.easting + (ch.prod.easting - ch.inj.easting) * u, ch.inj.northing + (ch.prod.northing - ch.inj.northing) * u,
          zA + (zB - zA) * u - S.thick * 0.18 * Math.sin(Math.PI * u));
      });
      const curve = new THREE.CatmullRomCurve3(pts);
      const length = curve.getLength();
      const color = water(0.25 + 0.75 * t);
      const group = new THREE.Group();
      // The water core fills the tube's cross-section in proportion to the injector's share (area ∝ f).
      const rCore = R * Math.max(0.2, Math.sqrt(Math.min(1, ch.f)));
      const glass = new THREE.Mesh(new THREE.TubeGeometry(curve, 120, R, 24, false),
        material(GLASS_FS, {uColor: hexColor('#bfe0f7'), uOpacity: 1, uTime: 0}));
      const core = new THREE.Mesh(new THREE.TubeGeometry(curve, 120, rCore, 18, false),
        material(WATER_FS, {uColor: color, uOpacity: 1, uTime: 0, uSpeed: 0.25 + 0.75 * share, uRepeat: Math.max(3, length / (R * 7))}));
      core.renderOrder = 1; glass.renderOrder = 2;
      group.add(core, glass);
      let halo = null;
      if (ch.main) {
        halo = new THREE.Mesh(new THREE.TubeGeometry(curve, 80, R * 2.4, 16, false),
          material(GLOW_FS, {uColor: color.clone().lerp(new THREE.Color('#5fb4ff'), 0.5), uOpacity: 0.22}));
        halo.renderOrder = 0;
        group.add(halo);
      }
      // Droplets racing through the tube; more and faster where more water flows.
      const n = Math.round(3 + 11 * share);
      const drops = new THREE.InstancedMesh(new THREE.SphereGeometry(Math.min(rCore * 0.42, R * 0.3), 12, 10),
        new THREE.MeshBasicMaterial({color: '#f2fbff', transparent: true, opacity: 0.85, depthWrite: false}), n);
      drops.renderOrder = 3;
      group.add(drops);
      root.add(group);
      items.push({ch, curve, glass, core, halo, drops, n, speed: 0.06 + 0.16 * share, phase: Array.from({length: n}, (_, k) => k / n + Math.random() * 0.4 / n)});
    }
    // Orbs where water enters (injectors) and leaves (producers), with ripples through the rock.
    const seen = new Map();
    for (const ch of S.flows.channels) {
      for (const [w, kind] of [[ch.inj, 'source'], [ch.prod, 'sink']]) {
        if (seen.has(w.id)) continue;
        const pos = rel(w.easting, w.northing, S.zOf(w));
        const color = kind === 'source' ? '#1f7fe0' : '#14a889';
        const orb = new THREE.Mesh(new THREE.SphereGeometry(R * 1.6, 32, 24), material(GLOW_FS, {uColor: hexColor(color), uOpacity: 0.95}));
        orb.position.copy(pos); orb.renderOrder = 4;
        const rings = [0, 1, 2].map(() => {
          const ring = new THREE.Mesh(new THREE.RingGeometry(0.9, 1, 72),
            new THREE.MeshBasicMaterial({color, transparent: true, opacity: 0.5, side: THREE.DoubleSide, depthWrite: false}));
          ring.position.copy(pos); root.add(ring); return ring;
        });
        root.add(orb);
        seen.set(w.id, true);
        wells.push({w, kind, orb, rings, reach: S.half * (kind === 'source' ? 0.3 : 0.22)});
      }
    }
  }

  function frame() {
    const S = sceneRef.current;
    if (S) {
      const W = S.W, H = S.H, dpr = Math.min(2, window.devicePixelRatio || 1);
      if (size !== `${W}x${H}x${dpr}`) { size = `${W}x${H}x${dpr}`; renderer.setPixelRatio(dpr); renderer.setSize(W, H, false); }
      const k = S.key + `|${S.cam.cx}|${S.cam.cy}|${S.cam.tz}`;
      if (k !== key) { key = k; build(S); }
      // Camera: same model as the SVG projection (see Field3D's P), incl. its vertical offset.
      const {c, s, cp, sp, R, f, oy} = S.cam;
      camera.fov = 2 * Math.atan(H / 2 / f) * 180 / Math.PI;
      camera.aspect = W / H;
      camera.near = R * 0.02; camera.far = R * 4;
      camera.position.set(-R * cp * s, -R * cp * c, R * sp);
      camera.up.set(sp * s, sp * c, cp);
      camera.lookAt(0, 0, 0);
      camera.setViewOffset(W, H, 0, -oy, W, H);
      camera.updateProjectionMatrix();
      const time = still ? 0 : (performance.now() - t0) / 1000;
      for (const it of items) {
        const faded = S.focus && it.ch.inj.id !== S.focus && it.ch.prod.id !== S.focus;
        const o = faded ? 0.14 : 1;
        it.glass.material.uniforms.uTime.value = time; it.glass.material.uniforms.uOpacity.value = o;
        it.core.material.uniforms.uTime.value = time; it.core.material.uniforms.uOpacity.value = o;
        if (it.halo) it.halo.material.uniforms.uOpacity.value = 0.22 * o * (0.7 + 0.3 * Math.sin(time * 2.2));
        it.drops.material.opacity = 0.85 * o;
        const m = new THREE.Matrix4();
        for (let j = 0; j < it.n; j++) {
          const u = (it.phase[j] + time * it.speed) % 1;
          const p = it.curve.getPointAt(u);
          const swell = Math.sin(Math.PI * u);                   // droplets grow away from the wells
          m.makeScale(0.6 + 0.6 * swell, 0.6 + 0.6 * swell, 0.6 + 0.6 * swell).setPosition(p);
          it.drops.setMatrixAt(j, m);
        }
        it.drops.instanceMatrix.needsUpdate = true;
      }
      for (const wl of wells) {
        const faded = S.focus && wl.w.id !== S.focus && !S.flows?.channels.some(x => (x.inj.id === S.focus || x.prod.id === S.focus) && (x.inj.id === wl.w.id || x.prod.id === wl.w.id));
        const o = faded ? 0.15 : 1;
        wl.orb.material.uniforms.uOpacity.value = 0.95 * o;
        wl.orb.scale.setScalar(1 + 0.12 * Math.sin(time * 3 + (wl.kind === 'sink' ? 1.5 : 0)));
        wl.rings.forEach((ring, j) => {
          let ph = ((time / 3.2) + j / 3) % 1;
          if (wl.kind === 'sink') ph = 1 - ph;                    // producers draw water in, injectors push it out
          ring.scale.setScalar(Math.max(1, wl.reach * ph));
          ring.material.opacity = 0.55 * Math.sin(Math.PI * ph) * o;
        });
      }
      renderer.render(scene, camera);
    }
    raf = requestAnimationFrame(frame);
  }
  raf = requestAnimationFrame(frame);
  return () => { cancelAnimationFrame(raf); clear(); renderer.dispose(); };
}
