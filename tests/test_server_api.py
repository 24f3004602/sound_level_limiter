from fastapi.testclient import TestClient

from server.app import app


def test_post_tasks_registers_custom_task() -> None:
    client = TestClient(app)

    payload = {
        "id": "task_api_custom",
        "name": "API Custom Task",
        "description": "Task inserted through POST /tasks",
        "difficulty": "custom",
        "initial_sound": 82.0,
        "noise_std": 3.3,
        "max_steps": 35,
        "success_threshold": 0.52,
    }

    response = client.post("/tasks?overwrite=true", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "task" in body
    assert body["task"]["id"] == payload["id"]

    task_response = client.get(f"/tasks/{payload['id']}")
    assert task_response.status_code == 200
    assert task_response.json()["difficulty"] == "custom"


def test_websocket_reset_and_step_roundtrip() -> None:
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        connected = websocket.receive_json()
        assert connected["type"] == "connected"
        assert "session_id" in connected

        websocket.send_json({"type": "reset", "task_id": "task_easy", "seed": 7})
        reset_payload = websocket.receive_json()
        assert reset_payload["type"] == "reset"
        assert "observation" in reset_payload

        websocket.send_json({"type": "step", "action": 2})
        step_payload = websocket.receive_json()
        assert step_payload["type"] == "step"
        assert "reward" in step_payload
        assert "state" in step_payload
