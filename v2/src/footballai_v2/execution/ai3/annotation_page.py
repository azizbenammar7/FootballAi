"""Self-contained local HTML page for clicking pitch landmarks on one frame.

The page embeds the frame image as a data URI and needs no network. Clicking
records the landmark's pixel at the image's native resolution (continuous,
origin top-left, the AI1 box convention). "Export" downloads one ``frames[]``
entry of ``footballai.pitch-landmark-annotations/v1`` that a person pastes into
the clip's annotation file. Every exported point starts ``pending`` review.

Write pages outside the Git repository: they contain private footage.
"""

from __future__ import annotations

import base64
import json

from footballai_v2.execution.ai3.geometry import LANDMARK_DESCRIPTIONS, PitchGeometry


def _script_json(value: object) -> str:
    return json.dumps(value, sort_keys=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def annotation_page_html(
    *,
    image_png: bytes,
    frame_index: int,
    timestamp_seconds: float,
    shot_id: str,
    frame_image_sha256: str,
    geometry: PitchGeometry,
) -> str:
    landmarks = [
        {"id": name, "x": x, "y": y, "description": LANDMARK_DESCRIPTIONS[name]}
        for name, (x, y) in geometry.landmarks().items()
    ]
    config = {
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "shot_id": shot_id,
        "frame_image_sha256": frame_image_sha256,
        "landmarks": landmarks,
    }
    image = base64.b64encode(image_png).decode("ascii")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pitch landmarks</title>
<style>
body {{ font-family: sans-serif; margin: 12px; background: #fff; color: #111; }}
#wrap {{ position: relative; display: inline-block; max-width: 100%; }}
#frame {{ max-width: 100%; cursor: crosshair; display: block; }}
.mark {{ position: absolute; width: 10px; height: 10px; margin: -6px 0 0 -6px; border: 2px solid #e00; border-radius: 50%; pointer-events: none; }}
.mark.validation {{ border-color: #06c; }}
table {{ border-collapse: collapse; margin-top: 8px; }} td, th {{ border: 1px solid #ccc; padding: 2px 6px; font-size: 13px; }}
</style></head><body>
<p>Frame <b id="fi"></b>. Choose a landmark, then click its exact line intersection. Blue = validation (never used for fitting).</p>
<label>Landmark <select id="landmark"></select></label>
<label><input type="checkbox" id="validation"> validation point</label>
<label>Quality <select id="quality"><option>clear</option><option>estimated</option></select></label>
<label>Annotator <input id="annotator" size="12"></label>
<button id="export">Export frame JSON</button>
<div id="wrap"><img id="frame" alt="video frame" src="data:image/png;base64,{image}"></div>
<table><thead><tr><th>landmark</th><th>pixel</th><th>role</th><th>quality</th><th></th></tr></thead><tbody id="rows"></tbody></table>
<script type="application/json" id="config">{_script_json(config)}</script>
<script>
"use strict";
const config = JSON.parse(document.getElementById("config").textContent);
const image = document.getElementById("frame");
const select = document.getElementById("landmark");
const points = new Map();
document.getElementById("fi").textContent = String(config.frame_index);
for (const item of config.landmarks) {{
  const option = document.createElement("option");
  option.value = item.id; option.textContent = item.id + " — " + item.description;
  select.appendChild(option);
}}
function render() {{
  document.querySelectorAll(".mark").forEach((node) => node.remove());
  const rows = document.getElementById("rows"); rows.textContent = "";
  const scale = image.clientWidth / image.naturalWidth;
  for (const [id, point] of points) {{
    const mark = document.createElement("div");
    mark.className = "mark" + (point.role === "validation" ? " validation" : "");
    mark.style.left = (point.pixel_x * scale) + "px"; mark.style.top = (point.pixel_y * scale) + "px";
    mark.title = id; document.getElementById("wrap").appendChild(mark);
    const row = document.createElement("tr");
    for (const text of [id, point.pixel_x.toFixed(1) + ", " + point.pixel_y.toFixed(1), point.role, point.quality]) {{
      const cell = document.createElement("td"); cell.textContent = text; row.appendChild(cell);
    }}
    const remove = document.createElement("button"); remove.textContent = "remove";
    remove.addEventListener("click", () => {{ points.delete(id); render(); }});
    const cell = document.createElement("td"); cell.appendChild(remove); row.appendChild(cell);
    rows.appendChild(row);
  }}
}}
image.addEventListener("click", (event) => {{
  const box = image.getBoundingClientRect();
  const scale = image.naturalWidth / box.width;
  const landmark = config.landmarks.find((item) => item.id === select.value);
  points.set(landmark.id, {{
    landmark_id: landmark.id,
    pixel_x: Math.round((event.clientX - box.left) * scale * 10) / 10,
    pixel_y: Math.round((event.clientY - box.top) * scale * 10) / 10,
    pitch_x_m: landmark.x, pitch_y_m: landmark.y,
    role: document.getElementById("validation").checked ? "validation" : "fit",
    quality: document.getElementById("quality").value,
    review_status: "pending",
    annotator: document.getElementById("annotator").value.trim() || "unnamed",
  }});
  render();
}});
window.addEventListener("resize", render);
document.getElementById("export").addEventListener("click", () => {{
  const frame = {{
    frame_index: config.frame_index, timestamp_seconds: config.timestamp_seconds, shot_id: config.shot_id,
    frame_image_sha256: config.frame_image_sha256,
    landmarks: Array.from(points.values()).sort((a, b) => a.landmark_id.localeCompare(b.landmark_id)),
  }};
  const blob = new Blob([JSON.stringify(frame, null, 2) + "\\n"], {{ type: "application/json" }});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob); link.download = "frame-" + config.frame_index + "-landmarks.json";
  link.click(); URL.revokeObjectURL(link.href);
}});
</script></body></html>
"""
