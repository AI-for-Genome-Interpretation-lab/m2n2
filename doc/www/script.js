/* 3D point clouds: <div class="cloud3d" data-src="widgets/xxx.json"></div>
   JSON: {coords: [[x, y, z], ...], group: [int], levels: [str], colors: [hex], inertia: [3]} */
(function () {
  "use strict";

  const CFG = {
    width: 640,
    height: 480,
    scale: 0.40,     // radius of the cloud, as a fraction of the smallest side
    axisLen: 1.15,   // axis half-length, in cloud radii
    persp: 0.25,     // depth foreshortening strength
    radius: 2.6,     // point radius at zero depth, px
    yaw: 0.6,
    pitch: -0.35
  };

  const SVG_NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const key in attrs) node.setAttribute(key, attrs[key]);
    return node;
  }

  /* Rotate a point by yaw (around the vertical) then pitch. */
  function rotate(p, yaw, pitch) {
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const x1 =  p[0] * cy + p[2] * sy;
    const z1 = -p[0] * sy + p[2] * cy;
    const y2 =  p[1] * cp - z1 * sp;
    const z2 =  p[1] * sp + z1 * cp;
    return [x1, y2, z2];
  }

  function build(root, data) {
    const w = CFG.width, h = CFG.height;
    const state = { yaw: CFG.yaw, pitch: CFG.pitch, drag: false, lastX: 0, lastY: 0 };

    // cloud normalized to a unit radius
    const R = Math.max.apply(null, data.coords.map(p => Math.hypot(p[0], p[1], p[2]))) || 1;
    const pts = data.coords.map(p => [p[0] / R, p[1] / R, p[2] / R]);
    const px = CFG.scale * Math.min(w, h);

    const svg = el("svg", {
      viewBox: "0 0 " + w + " " + h,
      class: "c3-svg", preserveAspectRatio: "xMidYMid meet"
    });
    const tip = document.createElement("div");
    tip.className = "c3-tip";
    root.appendChild(svg);
    root.appendChild(tip);

    const gAxes = el("g", {});
    const gPts = el("g", {});
    svg.appendChild(gAxes);
    svg.appendChild(gPts);

    const hint = el("text", { class: "c3-hint", x: w - 8, y: h - 8, "text-anchor": "end" });
    svg.appendChild(hint);

    const axes = [0, 1, 2].map(function (j) {
      const line = el("line", { class: "c3-axis" });
      const lab = el("text", { class: "c3-axlab", "text-anchor": "middle", "dominant-baseline": "middle" });
      lab.textContent = "PC" + (j + 1) + " (" + data.inertia[j].toFixed(1) + " %)";
      gAxes.appendChild(line);
      gAxes.appendChild(lab);
      return { line: line, lab: lab };
    });

    const nodes = pts.map(function () {
      const c = el("circle", { class: "c3-pt" });
      gPts.appendChild(c);
      return c;
    });

    function project(p) {
      const r = rotate(p, state.yaw, state.pitch);
      const f = 1 / (1 + CFG.persp * (1 - r[2]));
      return { x: w / 2 + r[0] * px * f, y: h / 2 - r[1] * px * f, z: r[2], f: f };
    }

    function render() {
      axes.forEach(function (a, j) {
        const lo = [0, 0, 0], hi = [0, 0, 0];
        lo[j] = -CFG.axisLen;
        hi[j] = CFG.axisLen;
        const p0 = project(lo), p1 = project(hi);
        a.line.setAttribute("x1", p0.x.toFixed(1));
        a.line.setAttribute("y1", p0.y.toFixed(1));
        a.line.setAttribute("x2", p1.x.toFixed(1));
        a.line.setAttribute("y2", p1.y.toFixed(1));
        const o = project([0, 0, 0]);
        a.lab.setAttribute("x", (o.x + (p1.x - o.x) * 1.12).toFixed(1));
        a.lab.setAttribute("y", (o.y + (p1.y - o.y) * 1.12).toFixed(1));
      });

      // painted back to front: node idx shows point k
      const proj = pts.map(function (p, k) {
        const q = project(p);
        q.k = k;
        return q;
      });
      proj.sort((a, b) => a.z - b.z);
      proj.forEach(function (q, idx) {
        const node = nodes[idx];
        node.setAttribute("cx", q.x.toFixed(1));
        node.setAttribute("cy", q.y.toFixed(1));
        node.setAttribute("r", (CFG.radius * q.f).toFixed(2));
        node.setAttribute("fill", data.colors[data.group[q.k]]);
        node.dataset.k = q.k;
      });
    }

    // rotation: pointer events cover mouse, pen and touch alike
    svg.addEventListener("pointerdown", function (ev) {
      state.drag = true;
      state.lastX = ev.clientX;
      state.lastY = ev.clientY;
      svg.setPointerCapture(ev.pointerId);
      tip.style.display = "none";
    });
    svg.addEventListener("pointermove", function (ev) {
      if (state.drag) {
        state.yaw += (ev.clientX - state.lastX) * 0.01;
        state.pitch = Math.max(-1.5, Math.min(1.5, state.pitch + (ev.clientY - state.lastY) * 0.01));
        state.lastX = ev.clientX;
        state.lastY = ev.clientY;
        render();
        return;
      }
      // tooltip: group of the hovered point
      const k = ev.target.dataset ? ev.target.dataset.k : undefined;
      if (k === undefined) {
        tip.style.display = "none";
        return;
      }
      const box = root.getBoundingClientRect();
      tip.textContent = data.levels[data.group[k]];
      tip.style.left = (ev.clientX - box.left + 12) + "px";
      tip.style.top = (ev.clientY - box.top + 12) + "px";
      tip.style.display = "block";
    });
    function release() { state.drag = false; }
    svg.addEventListener("pointerup", release);
    svg.addEventListener("pointercancel", release);
    svg.addEventListener("pointerleave", function () { tip.style.display = "none"; });

    render();
  }

  function init() {
    document.querySelectorAll(".cloud3d[data-src]").forEach(function (root) {
      fetch(root.dataset.src)
        .then(r => r.json())
        .then(data => build(root, data))
        .catch(function () { root.textContent = "Could not load " + root.dataset.src; });
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
