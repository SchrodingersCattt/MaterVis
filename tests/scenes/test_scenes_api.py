from __future__ import annotations

from crystal_viewer.app import create_app


def test_v2_scene_crud_and_state_targeting(tmp_path):
    app = create_app(
        preset_path=str(tmp_path / "preset.json"),
        root_dir=str(tmp_path),
    )
    client = app.server.test_client()

    response = client.get("/api/v2/scenes")
    assert response.status_code == 200
    scenes = response.get_json()["scenes"]
    active_id = response.get_json()["active_id"]
    assert active_id

    created = client.post("/api/v2/scenes", json={"structure": "__upload__", "label": "API scene"}).get_json()
    assert created["label"] == "API scene"

    patch = client.post(f"/api/v2/state?scene_id={created['id']}", json={"display_mode": "cluster"})
    assert patch.status_code == 200
    assert patch.get_json()["display_mode"] == "cluster"
    assert client.get(f"/api/v2/state?scene_id={created['id']}").get_json()["display_mode"] == "cluster"
    assert client.get(f"/api/v2/state?scene_id={active_id}").get_json()["display_mode"] == "formula_unit"

    duplicate = client.post(f"/api/v2/scenes/{created['id']}/duplicate", json={"label": "Copy"}).get_json()
    assert duplicate["label"] == "Copy"
    order = [item["id"] for item in client.get("/api/v2/scenes").get_json()["scenes"]]
    assert client.post("/api/v2/scenes/reorder", json={"order": list(reversed(order))}).status_code == 200
    assert client.delete(f"/api/v2/scenes/{duplicate['id']}").status_code == 200


def test_v1_state_shim_targets_active_scene(tmp_path):
    app = create_app(preset_path=str(tmp_path / "preset.json"), root_dir=str(tmp_path))
    client = app.server.test_client()
    response = client.post("/api/v1/state", json={"display_mode": "cluster"})
    assert response.status_code == 200
    assert client.get("/api/v1/state").get_json()["display_mode"] == "cluster"
