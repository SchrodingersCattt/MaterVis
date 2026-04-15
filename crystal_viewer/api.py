from __future__ import annotations

import json
import time
from typing import Any

from flask import Blueprint, Response, jsonify, request

try:
    from flask_sock import Sock
except Exception:  # pragma: no cover - optional dependency
    Sock = None


def register_api(dash_app, backend) -> None:
    server = dash_app.server
    blueprint = Blueprint("crystal_viewer_api", __name__, url_prefix="/api/v1")

    @blueprint.get("/state")
    def get_state():
        return jsonify(backend.get_state())

    @blueprint.post("/state")
    def post_state():
        payload = request.get_json(force=True, silent=True) or {}
        return jsonify(backend.patch_state(payload))

    @blueprint.get("/camera")
    def get_camera():
        return jsonify({"camera": backend.get_camera()})

    @blueprint.post("/camera")
    def post_camera():
        payload = request.get_json(force=True, silent=True) or {}
        camera = payload.get("camera", payload)
        return jsonify({"camera": backend.set_camera(camera)})

    @blueprint.post("/camera/action")
    def post_camera_action():
        payload = request.get_json(force=True, silent=True) or {}
        action = payload.get("action")
        if not action:
            return jsonify({"error": "action is required"}), 400
        rest = {key: value for key, value in payload.items() if key != "action"}
        return jsonify({"camera": backend.camera_action(action, **rest)})

    @blueprint.post("/upload")
    def upload_cif():
        if "file" not in request.files:
            return jsonify({"error": "missing multipart file field 'file'"}), 400
        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "empty filename"}), 400
        content = file.read()
        bundle = backend.add_uploaded_file_bytes(content, file.filename)
        return jsonify(bundle.metadata())

    @blueprint.get("/structures")
    def structures():
        return jsonify({"structures": backend.list_structures()})

    @blueprint.get("/scene/<name>")
    def scene(name: str):
        return jsonify(backend.get_scene_json(name))

    @blueprint.post("/topology")
    def topology():
        payload = request.get_json(force=True, silent=True) or {}
        structure = payload.get("structure") or backend.get_state().get("structure")
        center_index = payload.get("center_index")
        cutoff = float(payload.get("cutoff", 10.0))
        if center_index is None:
            return jsonify({"error": "center_index is required"}), 400
        return jsonify(backend.query_topology(structure=structure, center_index=int(center_index), cutoff=cutoff))

    @blueprint.get("/screenshot")
    def screenshot():
        png = backend.render_current_png()
        return Response(png, mimetype="image/png")

    @blueprint.post("/preset/save")
    def preset_save():
        payload = request.get_json(force=True, silent=True) or {}
        path = payload.get("path")
        return jsonify(backend.save_preset(path=path))

    @blueprint.post("/preset/load")
    def preset_load():
        payload = request.get_json(force=True, silent=True) or {}
        path = payload.get("path")
        return jsonify(backend.load_preset_from_path(path))

    @blueprint.post("/export")
    def export_static():
        payload = request.get_json(force=True, silent=True) or {}
        return jsonify(backend.export_static(output_path=payload.get("output_path")))

    server.register_blueprint(blueprint)

    if Sock is None:
        return

    sock = Sock(server)

    @sock.route("/api/v1/ws")
    def ws_state(socket):
        last_version = -1
        while True:
            snapshot = backend.websocket_snapshot()
            version = snapshot["version"]
            if version != last_version:
                socket.send(json.dumps(snapshot, ensure_ascii=False))
                last_version = version
            try:
                message = socket.receive(timeout=0.5)
            except TypeError:
                message = None
                time.sleep(0.5)
            if message:
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    payload = {"type": "raw", "message": message}
                if payload.get("type") == "set_state":
                    backend.patch_state(payload.get("payload", {}))
