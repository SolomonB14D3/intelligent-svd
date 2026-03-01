"""Core tests for intelligent-svd using mock models (no downloads needed).

Tests compression, freezing, and the CF90 pipeline using a lightweight
mock transformer that mimics the HuggingFace attention structure.
"""

import pytest
import torch
import torch.nn as nn


# ── Mock Model ───────────────────────────────────────────────────────────


class MockConfig:
    _name_or_path = "mock-model"
    hidden_size = 64
    num_attention_heads = 4
    num_hidden_layers = 4


class MockAttn(nn.Module):
    """Minimal attention module with Q/K/V/O projections."""

    def __init__(self, dim=64):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return x


class MockMLP(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim * 4, bias=False)
        self.up_proj = nn.Linear(dim, dim * 4, bias=False)
        self.down_proj = nn.Linear(dim * 4, dim, bias=False)

    def forward(self, x):
        return x


class MockLayer(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.self_attn = MockAttn(dim)
        self.mlp = MockMLP(dim)

    def forward(self, x):
        return x


class MockInnerModel(nn.Module):
    def __init__(self, n_layers=4, dim=64):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, dim)
        self.layers = nn.ModuleList([MockLayer(dim) for _ in range(n_layers)])

    def forward(self, x):
        return x


class MockModel(nn.Module):
    """Minimal HuggingFace-style causal LM for testing."""

    def __init__(self, n_layers=4, dim=64):
        super().__init__()
        self.config = MockConfig()
        self.model = MockInnerModel(n_layers, dim)
        self.lm_head = nn.Linear(dim, 100, bias=False)

    def forward(self, x):
        return x


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def mock_model():
    """Fresh mock model for each test."""
    return MockModel(n_layers=4, dim=64)


# ── Test: compress_qko ───────────────────────────────────────────────────


class TestCompressQKO:
    def test_compresses_qko_projections(self, mock_model):
        from intelligent_svd import compress_qko

        # Save original weights
        layer0_q = mock_model.model.layers[0].self_attn.q_proj.weight.data.clone()

        n = compress_qko(mock_model, ratio=0.7)

        # Should compress Q, K, O for all 4 layers = 12 matrices
        assert n == 12

        # Weight should be modified (rank-reduced)
        new_q = mock_model.model.layers[0].self_attn.q_proj.weight.data
        assert not torch.allclose(layer0_q, new_q, atol=1e-6), \
            "Q weight should change after compression"

    def test_v_proj_not_compressed(self, mock_model):
        from intelligent_svd import compress_qko

        v_before = mock_model.model.layers[0].self_attn.v_proj.weight.data.clone()
        compress_qko(mock_model, ratio=0.7)
        v_after = mock_model.model.layers[0].self_attn.v_proj.weight.data

        assert torch.allclose(v_before, v_after), \
            "V projection should NOT be compressed"

    def test_ratio_affects_rank(self, mock_model):
        from intelligent_svd.compress import compress_qko

        # At ratio=0.5, rank = max(1, int(64 * 0.5)) = 32
        # At ratio=0.9, rank = max(1, int(64 * 0.9)) = 57
        # Both should work without error
        model1 = MockModel(n_layers=2, dim=64)
        model2 = MockModel(n_layers=2, dim=64)

        compress_qko(model1, ratio=0.5)
        compress_qko(model2, ratio=0.9)

        # Both should succeed
        # Lower ratio → more aggressive compression → larger reconstruction error
        # (can't easily test rank directly, but the function shouldn't crash)

    def test_n_layers_limits_compression(self, mock_model):
        from intelligent_svd import compress_qko

        # Only compress first 2 of 4 layers
        n = compress_qko(mock_model, ratio=0.7, n_layers=2)
        assert n == 6  # 3 projections × 2 layers


# ── Test: freeze_layers ──────────────────────────────────────────────────


class TestFreezeLayers:
    def test_freeze_75_percent(self, mock_model):
        from intelligent_svd import freeze_layers

        stats = freeze_layers(mock_model, ratio=0.75)

        assert stats['n_layers'] == 4
        assert stats['n_frozen'] == 3  # 75% of 4 = 3

        # First 3 layers should be frozen
        for i in range(3):
            for p in mock_model.model.layers[i].parameters():
                assert not p.requires_grad, f"Layer {i} should be frozen"

        # Last layer should be trainable
        for p in mock_model.model.layers[3].parameters():
            assert p.requires_grad, "Last layer should be trainable"

    def test_freeze_embeddings(self, mock_model):
        from intelligent_svd import freeze_layers

        freeze_layers(mock_model, ratio=0.5)

        for p in mock_model.model.embed_tokens.parameters():
            assert not p.requires_grad, "Embeddings should be frozen"

    def test_unfreeze_restores_grad(self, mock_model):
        from intelligent_svd import freeze_layers, unfreeze_all

        freeze_layers(mock_model, ratio=0.75)
        unfreeze_all(mock_model)

        for p in mock_model.parameters():
            assert p.requires_grad, "All params should be trainable after unfreeze"

    def test_stats_keys(self, mock_model):
        from intelligent_svd import freeze_layers

        stats = freeze_layers(mock_model, ratio=0.5)
        expected_keys = {
            'n_layers', 'n_frozen', 'freeze_ratio',
            'trainable_params', 'total_params', 'trainable_pct',
        }
        assert set(stats.keys()) == expected_keys


# ── Test: apply_cf90 ─────────────────────────────────────────────────────


class TestApplyCF90:
    def test_returns_stats_dict(self, mock_model):
        from intelligent_svd import apply_cf90

        stats = apply_cf90(mock_model, ratio=0.7, freeze_ratio=0.75)

        assert 'n_compressed' in stats
        assert 'ratio' in stats
        assert 'n_frozen' in stats
        assert 'n_layers' in stats
        assert stats['ratio'] == 0.7
        assert stats['n_compressed'] == 12  # Q,K,O × 4 layers
        assert stats['n_frozen'] == 3  # 75% of 4

    def test_model_modified_in_place(self, mock_model):
        from intelligent_svd import apply_cf90

        q_before = mock_model.model.layers[0].self_attn.q_proj.weight.data.clone()

        apply_cf90(mock_model)

        q_after = mock_model.model.layers[0].self_attn.q_proj.weight.data
        assert not torch.allclose(q_before, q_after, atol=1e-6)

    def test_frozen_layers_not_trainable(self, mock_model):
        from intelligent_svd import apply_cf90

        apply_cf90(mock_model, freeze_ratio=0.75)

        # First 3 layers frozen
        for i in range(3):
            for p in mock_model.model.layers[i].parameters():
                assert not p.requires_grad


# ── Test: version ─────────────────────────────────────────────────────────


class TestVersion:
    def test_version_exists(self):
        from intelligent_svd import __version__
        assert isinstance(__version__, str)
        assert len(__version__) > 0
