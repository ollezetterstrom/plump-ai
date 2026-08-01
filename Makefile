.PHONY: test test-rust test-py build py-build fmt clippy setup

PY ?= .venv/bin/python
MATURIN ?= maturin

# Rust engine (default workspace member, no external deps needed).
test-rust:
	cargo test

# Python extension via maturin. Requires a venv (see `make setup`);
# VIRTUAL_ENV lets maturin find it even when the `maturin` binary lives outside.
py-build:
	VIRTUAL_ENV="$$(pwd)/.venv" $(MATURIN) develop --release --manifest-path crates/plump-py/Cargo.toml

# Python tests (requires `make py-build` first).
test-py:
	$(PY) -m pytest -q

test: test-rust test-py

fmt:
	cargo fmt

clippy:
	cargo clippy --all-targets --all-features -- -D warnings

# One-time dev setup: venv (reusing system torch) + maturin + pytest.
setup:
	python3 -m venv --system-site-packages .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install maturin pytest
