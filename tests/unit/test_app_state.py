from pathlib import Path


def test_app_state_has_supervisor_and_port_allocator(client):
    app = client.app
    assert hasattr(app.state, "supervisor")
    assert hasattr(app.state, "port_allocator")
    assert app.state.port_allocator.allocate() >= 10000


def test_engine_driver_setting_defaults_local():
    # Settings lives in app.config (frozen dataclass with required fields),
    # not app.settings; construct it directly with the required args.
    from app.config import Settings

    s = Settings(
        data_dir=Path("/data"),
        hf_cache_dir=Path("/root/.cache/huggingface"),
        cookie_secret="x" * 32,
        container_gpu_count=0,
    )
    assert s.engine_driver == "local"
