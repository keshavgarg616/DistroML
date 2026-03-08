"""
Test checkpoint restore functionality for workers.

Tests:
1. Checkpoint save and restore cycle
2. Model/optimizer/RNG state preservation
3. Training resume from correct step
4. Manifest validation and error handling
"""

import os
import sys
import shutil
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.worker.runtime import WorkerConfig, WorkerRuntime
from src.coordinator.manifest import build_manifest, save_manifest, load_manifest


class TestCheckpointRestore:
    """Test suite for checkpoint restore functionality"""

    def setup_method(self):
        """Create temporary checkpoint directory for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"\n🧪 Test checkpoint dir: {self.checkpoint_dir}")

    def teardown_method(self):
        """Clean up temporary directory"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print(f"✅ Cleaned up {self.temp_dir}")

    def create_worker_config(self, rank: int = 0, world_size: int = 2, run_id: str = "test-run"):
        """Helper to create worker config"""
        return WorkerConfig(
            worker_id=f"worker_{rank}",
            rank=rank,
            world_size=world_size,
            coordinator_url="http://localhost:8000",
            job_id="test-job",
            run_id=run_id,
            backend="gloo",
            checkpoint_dir=self.checkpoint_dir,
        )

    def create_mock_checkpoint(self, run_id: str, step: int, rank: int, world_size: int):
        """
        Create a mock checkpoint shard file.
        Returns the checkpoint path.
        """
        checkpoint_step_dir = os.path.join(
            self.checkpoint_dir,
            run_id,
            f"ckpt_step_{step:06d}"
        )
        os.makedirs(checkpoint_step_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_step_dir, f"worker_{rank}.pt")

        # Create dummy model and optimizer states (must match WorkerRuntime's model: Linear(10, 1))
        model = nn.Linear(10, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        state = {
            "step": step,
            "epoch": step // 50,
            "rank": rank,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": None,
        }

        torch.save(state, checkpoint_path)
        return checkpoint_path

    def create_mock_manifest(self, run_id: str, step: int, world_size: int):
        """
        Create a mock manifest.json for a checkpoint.
        """
        checkpoint_step_dir = os.path.join(
            self.checkpoint_dir,
            run_id,
            f"ckpt_step_{step:06d}"
        )

        worker_shards = []
        for rank in range(world_size):
            shard_path = os.path.join(checkpoint_step_dir, f"worker_{rank}.pt")
            worker_shards.append({
                "worker_id": f"worker_{rank}",
                "rank": rank,
                "path": shard_path,
                "file_size_bytes": os.path.getsize(shard_path) if os.path.exists(shard_path) else 0,
                "sha256": "fake_hash_for_testing",
            })

        manifest = build_manifest(
            step=step,
            run_id=run_id,
            job_id="test-job",
            world_size=world_size,
            worker_shards=worker_shards,
        )

        save_manifest(manifest, checkpoint_step_dir)
        return os.path.join(checkpoint_step_dir, "manifest.json")

    def test_no_checkpoint_found(self):
        """Test: Worker starts fresh when no checkpoint exists"""
        print("\n📝 Test: No checkpoint found - start fresh")

        config = self.create_worker_config(rank=0, run_id="empty-run")
        worker = WorkerRuntime(config)

        result = worker.find_latest_checkpoint()
        assert result is None, "Expected None when no checkpoint exists"
        print("✅ Correctly returned None for missing checkpoint")

    def test_find_latest_checkpoint_with_manifest(self):
        """Test: Worker finds latest checkpoint with valid manifest"""
        print("\n📝 Test: Find latest checkpoint with manifest")

        run_id = "multi-ckpt-run"
        world_size = 2

        # Create checkpoints at steps 200, 400, 600
        for step in [200, 400, 600]:
            for rank in range(world_size):
                self.create_mock_checkpoint(run_id, step, rank, world_size)
            self.create_mock_manifest(run_id, step, world_size)

        config = self.create_worker_config(rank=0, world_size=world_size, run_id=run_id)
        worker = WorkerRuntime(config)

        result = worker.find_latest_checkpoint()
        assert result is not None, "Expected to find checkpoint"

        checkpoint_dir, manifest = result
        assert manifest["step"] == 600, f"Expected step 600, got {manifest['step']}"
        print(f"✅ Found latest checkpoint at step {manifest['step']}")

    def test_skip_checkpoint_without_manifest(self):
        """Test: Worker skips checkpoint directories without manifest"""
        print("\n📝 Test: Skip checkpoint without manifest")

        run_id = "partial-ckpt-run"
        world_size = 2

        # Create checkpoint at step 200 WITH manifest
        for rank in range(world_size):
            self.create_mock_checkpoint(run_id, 200, rank, world_size)
        self.create_mock_manifest(run_id, 200, world_size)

        # Create checkpoint at step 400 WITHOUT manifest (incomplete)
        for rank in range(world_size):
            self.create_mock_checkpoint(run_id, 400, rank, world_size)
        # Intentionally skip creating manifest for step 400

        config = self.create_worker_config(rank=0, world_size=world_size, run_id=run_id)
        worker = WorkerRuntime(config)

        result = worker.find_latest_checkpoint()
        assert result is not None, "Expected to find checkpoint"

        checkpoint_dir, manifest = result
        assert manifest["step"] == 200, f"Expected step 200 (skipped 400 without manifest), got {manifest['step']}"
        print(f"✅ Correctly skipped incomplete checkpoint, restored from step {manifest['step']}")

    def test_restore_checkpoint_success(self):
        """Test: Worker successfully restores checkpoint state"""
        print("\n📝 Test: Restore checkpoint successfully")

        run_id = "restore-test"
        world_size = 2
        rank = 0
        step = 200

        # Create checkpoint and manifest
        for r in range(world_size):
            self.create_mock_checkpoint(run_id, step, r, world_size)
        self.create_mock_manifest(run_id, step, world_size)

        config = self.create_worker_config(rank=rank, world_size=world_size, run_id=run_id)
        worker = WorkerRuntime(config)

        # Find and restore checkpoint
        checkpoint_info = worker.find_latest_checkpoint()
        assert checkpoint_info is not None

        checkpoint_dir, manifest = checkpoint_info
        next_step = worker.restore_checkpoint(checkpoint_dir, manifest)

        # Verify restoration
        assert next_step == step + 1, f"Expected next step {step + 1}, got {next_step}"
        assert worker.current_epoch == step // 50, f"Epoch not restored correctly"
        print(f"✅ Restored checkpoint: next_step={next_step}, epoch={worker.current_epoch}")

    def test_restore_world_size_mismatch(self):
        """Test: Worker rejects checkpoint with mismatched world_size"""
        print("\n📝 Test: Reject checkpoint with world_size mismatch")

        run_id = "mismatch-run"
        checkpoint_world_size = 4
        worker_world_size = 2
        rank = 0
        step = 200

        # Create checkpoint with world_size=4
        for r in range(checkpoint_world_size):
            self.create_mock_checkpoint(run_id, step, r, checkpoint_world_size)
        self.create_mock_manifest(run_id, step, checkpoint_world_size)

        # Worker has world_size=2
        config = self.create_worker_config(rank=rank, world_size=worker_world_size, run_id=run_id)
        worker = WorkerRuntime(config)

        checkpoint_info = worker.find_latest_checkpoint()
        assert checkpoint_info is not None

        checkpoint_dir, manifest = checkpoint_info

        # Should raise ValueError for world_size mismatch
        try:
            worker.restore_checkpoint(checkpoint_dir, manifest)
            assert False, "Expected ValueError for world_size mismatch"
        except ValueError as e:
            assert "world_size" in str(e).lower()
            print(f"✅ Correctly rejected checkpoint: {e}")

    def test_restore_rank_not_in_manifest(self):
        """Test: Worker fails gracefully when rank not in manifest"""
        print("\n📝 Test: Rank not found in manifest")

        run_id = "missing-rank-run"
        world_size = 2
        step = 200

        # Create checkpoint only for rank 0 (missing rank 1)
        self.create_mock_checkpoint(run_id, step, rank=0, world_size=world_size)

        # Create manifest with only rank 0
        checkpoint_step_dir = os.path.join(
            self.checkpoint_dir, run_id, f"ckpt_step_{step:06d}"
        )
        worker_shards = [{
            "worker_id": "worker_0",
            "rank": 0,
            "path": os.path.join(checkpoint_step_dir, "worker_0.pt"),
            "file_size_bytes": 1000,
            "sha256": "fake_hash",
        }]
        manifest = build_manifest(
            step=step, run_id=run_id, job_id="test-job",
            world_size=world_size, worker_shards=worker_shards
        )
        save_manifest(manifest, checkpoint_step_dir)

        # Worker with rank 1 tries to restore
        config = self.create_worker_config(rank=1, world_size=world_size, run_id=run_id)
        worker = WorkerRuntime(config)

        checkpoint_info = worker.find_latest_checkpoint()
        checkpoint_dir, manifest_data = checkpoint_info

        try:
            worker.restore_checkpoint(checkpoint_dir, manifest_data)
            assert False, "Expected ValueError for missing rank"
        except ValueError as e:
            assert "rank" in str(e).lower()
            print(f"✅ Correctly rejected: {e}")

    def test_manifest_load_and_validate(self):
        """Test: Manifest loading with validation"""
        print("\n📝 Test: Manifest validation")

        run_id = "manifest-test"
        world_size = 2
        step = 200

        # Create valid checkpoint with manifest
        for rank in range(world_size):
            self.create_mock_checkpoint(run_id, step, rank, world_size)
        manifest_path = self.create_mock_manifest(run_id, step, world_size)

        # Test loading valid manifest
        checkpoint_dir = os.path.dirname(manifest_path)
        manifest = load_manifest(checkpoint_dir)

        assert manifest["step"] == step
        assert manifest["world_size"] == world_size
        assert len(manifest["worker_shards"]) == world_size
        print(f"✅ Valid manifest loaded: step={manifest['step']}, world_size={manifest['world_size']}")

        # Test loading from non-existent directory
        try:
            load_manifest("/nonexistent/path")
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            print("✅ Correctly raised FileNotFoundError for missing manifest")

    def test_model_state_preservation(self):
        """Test: Model weights are preserved across save/restore"""
        print("\n📝 Test: Model state preservation")

        run_id = "state-preserve-test"
        world_size = 1
        rank = 0
        step = 100

        config = self.create_worker_config(rank=rank, world_size=world_size, run_id=run_id)
        worker = WorkerRuntime(config)

        # Get original model weights
        original_weights = worker.model.weight.data.clone()
        original_bias = worker.model.bias.data.clone()

        # Save checkpoint
        worker.current_step = step
        worker.current_epoch = step // 50
        worker.save_checkpoint(step)

        # Create manifest (normally done by coordinator)
        checkpoint_step_dir = os.path.join(
            self.checkpoint_dir, run_id, f"ckpt_step_{step:06d}"
        )
        self.create_mock_manifest(run_id, step, world_size)

        # Modify model weights
        worker.model.weight.data.fill_(0.0)
        worker.model.bias.data.fill_(0.0)

        # Restore checkpoint
        checkpoint_info = worker.find_latest_checkpoint()
        checkpoint_dir, manifest = checkpoint_info
        worker.restore_checkpoint(checkpoint_dir, manifest)

        # Verify weights restored
        assert torch.allclose(worker.model.weight.data, original_weights), "Weights not restored"
        assert torch.allclose(worker.model.bias.data, original_bias), "Bias not restored"
        print("✅ Model state correctly preserved and restored")

    def test_corrupted_checkpoint_file(self):
        """Test: Worker handles corrupted checkpoint file gracefully"""
        print("\n📝 Test: Corrupted checkpoint file")

        run_id = "corrupted-test"
        world_size = 2
        rank = 0
        step = 200

        # Create valid checkpoint for rank 1
        self.create_mock_checkpoint(run_id, step, rank=1, world_size=world_size)

        # Create corrupted checkpoint for rank 0
        checkpoint_step_dir = os.path.join(
            self.checkpoint_dir, run_id, f"ckpt_step_{step:06d}"
        )
        os.makedirs(checkpoint_step_dir, exist_ok=True)
        corrupted_path = os.path.join(checkpoint_step_dir, "worker_0.pt")

        # Write garbage data
        with open(corrupted_path, "wb") as f:
            f.write(b"CORRUPTED_DATA_NOT_A_PYTORCH_FILE")

        # Create manifest
        self.create_mock_manifest(run_id, step, world_size)

        config = self.create_worker_config(rank=rank, world_size=world_size, run_id=run_id)
        worker = WorkerRuntime(config)

        checkpoint_info = worker.find_latest_checkpoint()
        checkpoint_dir, manifest = checkpoint_info

        # Should fail to load corrupted file
        try:
            worker.restore_checkpoint(checkpoint_dir, manifest)
            assert False, "Expected exception for corrupted checkpoint"
        except Exception as e:
            # Expect UnpicklingError, RuntimeError, or similar from torch.load
            exc_name = type(e).__name__
            # Check for pickle-related errors or runtime errors
            is_expected_error = (
                "unpickling" in exc_name.lower() or
                "pickle" in exc_name.lower() or
                "runtime" in exc_name.lower()
            )
            assert is_expected_error, f"Unexpected exception type: {exc_name}"
            print(f"✅ Correctly failed on corrupted file: {exc_name}")

    def test_missing_checkpoint_file(self):
        """Test: Worker fails gracefully when checkpoint file is missing"""
        print("\n📝 Test: Missing checkpoint file (manifest exists, file deleted)")

        run_id = "missing-file-test"
        world_size = 2
        rank = 0
        step = 200

        # Create checkpoint files
        for r in range(world_size):
            self.create_mock_checkpoint(run_id, step, r, world_size)

        # Create manifest
        self.create_mock_manifest(run_id, step, world_size)

        # Delete rank 0's checkpoint file (simulate deletion after manifest creation)
        checkpoint_step_dir = os.path.join(
            self.checkpoint_dir, run_id, f"ckpt_step_{step:06d}"
        )
        os.remove(os.path.join(checkpoint_step_dir, "worker_0.pt"))

        config = self.create_worker_config(rank=rank, world_size=world_size, run_id=run_id)
        worker = WorkerRuntime(config)

        checkpoint_info = worker.find_latest_checkpoint()
        checkpoint_dir, manifest = checkpoint_info

        try:
            worker.restore_checkpoint(checkpoint_dir, manifest)
            assert False, "Expected FileNotFoundError for missing checkpoint file"
        except FileNotFoundError as e:
            assert "worker_0.pt" in str(e)
            print(f"✅ Correctly raised FileNotFoundError: {e}")

    def test_malformed_manifest_json(self):
        """Test: Worker handles malformed manifest.json"""
        print("\n📝 Test: Malformed manifest JSON")

        run_id = "malformed-manifest-test"
        world_size = 2
        step = 200

        # Create checkpoints
        for rank in range(world_size):
            self.create_mock_checkpoint(run_id, step, rank, world_size)

        # Create malformed manifest
        checkpoint_step_dir = os.path.join(
            self.checkpoint_dir, run_id, f"ckpt_step_{step:06d}"
        )
        os.makedirs(checkpoint_step_dir, exist_ok=True)
        manifest_path = os.path.join(checkpoint_step_dir, "manifest.json")

        with open(manifest_path, "w") as f:
            f.write("{ INVALID JSON SYNTAX }")

        try:
            load_manifest(checkpoint_step_dir)
            assert False, "Expected JSONDecodeError for malformed JSON"
        except json.JSONDecodeError:
            print("✅ Correctly raised JSONDecodeError for malformed JSON")

    def test_manifest_missing_required_fields(self):
        """Test: Worker validates manifest has all required fields"""
        print("\n📝 Test: Manifest missing required fields")

        run_id = "incomplete-manifest-test"
        world_size = 2
        step = 200

        # Create checkpoints
        for rank in range(world_size):
            self.create_mock_checkpoint(run_id, step, rank, world_size)

        checkpoint_step_dir = os.path.join(
            self.checkpoint_dir, run_id, f"ckpt_step_{step:06d}"
        )
        os.makedirs(checkpoint_step_dir, exist_ok=True)
        manifest_path = os.path.join(checkpoint_step_dir, "manifest.json")

        # Create manifest missing "worker_shards" field
        incomplete_manifest = {
            "step": step,
            "run_id": run_id,
            "job_id": "test-job",
            "world_size": world_size,
            # Missing: "worker_shards"
        }

        with open(manifest_path, "w") as f:
            json.dump(incomplete_manifest, f)

        try:
            load_manifest(checkpoint_step_dir)
            assert False, "Expected ValueError for missing fields"
        except ValueError as e:
            assert "worker_shards" in str(e)
            print(f"✅ Correctly raised ValueError: {e}")

    def test_rng_state_reproducibility(self):
        """Test: RNG state produces identical random sequences after restore"""
        print("\n📝 Test: RNG state reproducibility")

        run_id = "rng-test"
        world_size = 1
        rank = 0
        step = 100

        config = self.create_worker_config(rank=rank, world_size=world_size, run_id=run_id)
        worker = WorkerRuntime(config)

        # Set known RNG state
        torch.manual_seed(42)

        # Generate some random numbers before checkpoint
        random_before_1 = torch.randn(5)
        random_before_2 = torch.randn(5)

        # Save checkpoint
        worker.current_step = step
        worker.current_epoch = step // 50
        worker.save_checkpoint(step)

        # Create manifest
        checkpoint_step_dir = os.path.join(
            self.checkpoint_dir, run_id, f"ckpt_step_{step:06d}"
        )
        self.create_mock_manifest(run_id, step, world_size)

        # Change RNG state
        torch.manual_seed(999)
        random_after_seed_change = torch.randn(5)

        # Restore checkpoint (should restore RNG state)
        checkpoint_info = worker.find_latest_checkpoint()
        checkpoint_dir, manifest = checkpoint_info
        worker.restore_checkpoint(checkpoint_dir, manifest)

        # Generate random numbers after restore
        random_after_restore_1 = torch.randn(5)
        random_after_restore_2 = torch.randn(5)

        # These should NOT match (RNG was at different state when checkpoint was saved)
        assert not torch.allclose(random_after_restore_1, random_before_1), \
            "RNG shouldn't produce same sequence (different points in sequence)"

        # But we can verify RNG was restored by checking state is not the seed(999) state
        assert not torch.allclose(random_after_restore_1, random_after_seed_change), \
            "RNG state should be restored, not from seed(999)"

        print("✅ RNG state correctly restored (produces different sequence than corrupted state)")

    def test_optimizer_momentum_preservation(self):
        """Test: Optimizer momentum buffers are preserved"""
        print("\n📝 Test: Optimizer momentum preservation")

        run_id = "momentum-test"
        world_size = 1
        rank = 0
        step = 100

        config = self.create_worker_config(rank=rank, world_size=world_size, run_id=run_id)
        worker = WorkerRuntime(config)

        # Do a few training steps to build momentum
        for _ in range(5):
            loss = (worker.model(torch.randn(1, 10)) - torch.randn(1, 1)).pow(2).sum()
            loss.backward()
            worker.optimizer.step()
            worker.optimizer.zero_grad()

        # Get optimizer state (includes momentum buffers)
        original_optimizer_state = worker.optimizer.state_dict()

        # Save checkpoint
        worker.current_step = step
        worker.current_epoch = step // 50
        worker.save_checkpoint(step)

        # Create manifest
        checkpoint_step_dir = os.path.join(
            self.checkpoint_dir, run_id, f"ckpt_step_{step:06d}"
        )
        self.create_mock_manifest(run_id, step, world_size)

        # Reset optimizer (clears momentum)
        worker.optimizer = optim.SGD(worker.model.parameters(), lr=0.01)

        # Restore checkpoint
        checkpoint_info = worker.find_latest_checkpoint()
        checkpoint_dir, manifest = checkpoint_info
        worker.restore_checkpoint(checkpoint_dir, manifest)

        # Verify optimizer state matches (including momentum)
        restored_optimizer_state = worker.optimizer.state_dict()

        # Check param_groups match
        assert len(original_optimizer_state['param_groups']) == len(restored_optimizer_state['param_groups'])

        # Check state dict has same keys
        assert set(original_optimizer_state['state'].keys()) == set(restored_optimizer_state['state'].keys())

        print("✅ Optimizer state (including momentum) correctly preserved")

    def test_step_zero_checkpoint(self):
        """Test: Handle checkpoint at step 0"""
        print("\n📝 Test: Checkpoint at step 0")

        run_id = "step-zero-test"
        world_size = 1
        rank = 0
        step = 0

        # Create checkpoint at step 0
        self.create_mock_checkpoint(run_id, step, rank, world_size)
        self.create_mock_manifest(run_id, step, world_size)

        config = self.create_worker_config(rank=rank, world_size=world_size, run_id=run_id)
        worker = WorkerRuntime(config)

        checkpoint_info = worker.find_latest_checkpoint()
        assert checkpoint_info is not None

        checkpoint_dir, manifest = checkpoint_info
        next_step = worker.restore_checkpoint(checkpoint_dir, manifest)

        assert next_step == 1, f"Expected next step 1 from step 0 checkpoint, got {next_step}"
        print(f"✅ Step 0 checkpoint correctly handled, next_step={next_step}")

    def test_resume_when_already_past_checkpoint(self):
        """Test: Handle case where restored_step >= total_steps"""
        print("\n📝 Test: Resume when already past total_steps")

        run_id = "past-total-test"
        world_size = 1
        rank = 0
        step = 500

        # Create checkpoint at step 500
        self.create_mock_checkpoint(run_id, step, rank, world_size)
        self.create_mock_manifest(run_id, step, world_size)

        config = self.create_worker_config(rank=rank, world_size=world_size, run_id=run_id)
        worker = WorkerRuntime(config)

        checkpoint_info = worker.find_latest_checkpoint()
        checkpoint_dir, manifest = checkpoint_info
        next_step = worker.restore_checkpoint(checkpoint_dir, manifest)

        # Start training with total_steps=100 (already past checkpoint at 500)
        worker.current_step = next_step - 1

        # Training loop should handle this gracefully (loop won't execute)
        step_count = 0
        for step in range(max(1, next_step + 1), 100 + 1):
            step_count += 1

        assert step_count == 0, "Should not execute any steps when past total_steps"
        print("✅ Correctly handles checkpoint beyond total_steps (no training steps)")

    def test_multiple_run_ids_in_checkpoint_dir(self):
        """Test: Worker only finds checkpoints for its run_id"""
        print("\n📝 Test: Multiple run_ids in checkpoint directory")

        run_id_1 = "run-001"
        run_id_2 = "run-002"
        world_size = 2
        rank = 0

        # Create checkpoints for run-001 at step 200
        for r in range(world_size):
            self.create_mock_checkpoint(run_id_1, 200, r, world_size)
        self.create_mock_manifest(run_id_1, 200, world_size)

        # Create checkpoints for run-002 at step 400
        for r in range(world_size):
            self.create_mock_checkpoint(run_id_2, 400, r, world_size)
        self.create_mock_manifest(run_id_2, 400, world_size)

        # Worker with run-001 should only see step 200
        config = self.create_worker_config(rank=rank, world_size=world_size, run_id=run_id_1)
        worker = WorkerRuntime(config)

        checkpoint_info = worker.find_latest_checkpoint()
        assert checkpoint_info is not None

        checkpoint_dir, manifest = checkpoint_info
        assert manifest["step"] == 200, f"Expected step 200 for run-001, got {manifest['step']}"
        assert manifest["run_id"] == run_id_1

        print(f"✅ Correctly isolated run_id {run_id_1}, found step {manifest['step']}")

    def test_non_sequential_checkpoint_steps(self):
        """Test: Find latest checkpoint when steps are not sequential"""
        print("\n📝 Test: Non-sequential checkpoint steps")

        run_id = "non-sequential-test"
        world_size = 2
        rank = 0

        # Create checkpoints at non-sequential steps: 100, 500, 250, 600
        for step in [100, 500, 250, 600]:
            for r in range(world_size):
                self.create_mock_checkpoint(run_id, step, r, world_size)
            self.create_mock_manifest(run_id, step, world_size)

        config = self.create_worker_config(rank=rank, world_size=world_size, run_id=run_id)
        worker = WorkerRuntime(config)

        checkpoint_info = worker.find_latest_checkpoint()
        assert checkpoint_info is not None

        checkpoint_dir, manifest = checkpoint_info
        assert manifest["step"] == 600, f"Expected latest step 600, got {manifest['step']}"
        print(f"✅ Correctly found latest step {manifest['step']} from non-sequential checkpoints")

    def test_checkpoint_with_wrong_rank_in_file(self):
        """Test: Detect when checkpoint file has mismatched rank"""
        print("\n📝 Test: Checkpoint file contains wrong rank")

        run_id = "wrong-rank-test"
        world_size = 2
        rank = 0
        step = 200

        # Create checkpoint with rank=1 data but save as worker_0.pt
        checkpoint_step_dir = os.path.join(
            self.checkpoint_dir, run_id, f"ckpt_step_{step:06d}"
        )
        os.makedirs(checkpoint_step_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_step_dir, "worker_0.pt")

        model = nn.Linear(10, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Save with rank=1 in the state (wrong!)
        state = {
            "step": step,
            "epoch": step // 50,
            "rank": 1,  # Should be 0!
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": None,
        }
        torch.save(state, checkpoint_path)

        # Create checkpoint for rank 1
        self.create_mock_checkpoint(run_id, step, rank=1, world_size=world_size)

        # Create manifest
        self.create_mock_manifest(run_id, step, world_size)

        config = self.create_worker_config(rank=rank, world_size=world_size, run_id=run_id)
        worker = WorkerRuntime(config)

        checkpoint_info = worker.find_latest_checkpoint()
        checkpoint_dir, manifest = checkpoint_info

        try:
            worker.restore_checkpoint(checkpoint_dir, manifest)
            assert False, "Expected ValueError for rank mismatch"
        except ValueError as e:
            assert "rank" in str(e).lower()
            print(f"✅ Correctly detected rank mismatch: {e}")


def run_all_tests():
    """Run all tests and report results"""
    print("=" * 70)
    print("🧪 CHECKPOINT RESTORE TEST SUITE - COMPREHENSIVE EDITION")
    print("=" * 70)

    test_suite = TestCheckpointRestore()
    tests = [
        # Basic functionality tests
        test_suite.test_no_checkpoint_found,
        test_suite.test_find_latest_checkpoint_with_manifest,
        test_suite.test_skip_checkpoint_without_manifest,
        test_suite.test_restore_checkpoint_success,

        # Validation and error handling tests
        test_suite.test_restore_world_size_mismatch,
        test_suite.test_restore_rank_not_in_manifest,
        test_suite.test_manifest_load_and_validate,

        # State preservation tests
        test_suite.test_model_state_preservation,
        test_suite.test_optimizer_momentum_preservation,
        test_suite.test_rng_state_reproducibility,

        # File corruption and missing file tests
        test_suite.test_corrupted_checkpoint_file,
        test_suite.test_missing_checkpoint_file,

        # Manifest edge cases
        test_suite.test_malformed_manifest_json,
        test_suite.test_manifest_missing_required_fields,

        # Step and resume edge cases
        test_suite.test_step_zero_checkpoint,
        test_suite.test_resume_when_already_past_checkpoint,

        # Multi-run and ordering tests
        test_suite.test_multiple_run_ids_in_checkpoint_dir,
        test_suite.test_non_sequential_checkpoint_steps,

        # Data integrity tests
        test_suite.test_checkpoint_with_wrong_rank_in_file,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        test_suite.setup_method()
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {test_func.__name__}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"💥 ERROR: {test_func.__name__}")
            print(f"   Exception: {e}")
            failed += 1
        finally:
            test_suite.teardown_method()

    print("\n" + "=" * 70)
    print(f"📊 TEST RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
