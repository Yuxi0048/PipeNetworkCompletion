# ISARC Low-Quality Node-Attribute Ablation

This note documents a controlled ablation of the original ISARC-style
anchor-based GNN pipeline. It is separate from the newer anchor-free road-support
experiments.

## Research Question

How much of the original manhole-to-manhole topology prediction performance is
retained if the graph keeps only node locations and removes most node
attributes?

This mimics a lower-quality asset database where manhole and road coordinates
are known, but categorical node attributes are missing or unreliable.

## What Is Preserved

The original model treats the first two columns of both node feature matrices as
spatial coordinates:

- `data["MH"].x[:, :2]`
- `data["Road"].x[:, :2]`

The low-quality ablation keeps those two columns unchanged.

The graph structure, edge labels, and train/validation/test graph split remain
unchanged. By default, edge attributes are also retained because the requested
ablation removes node attributes. A stricter context-poor variant can also zero
edge attributes with `--zero-edge-attrs`.

## What Is Removed

For `MH` and `Road` nodes, all columns after the first two are set to zero:

- `data["MH"].x[:, 2:] = 0`
- `data["Road"].x[:, 2:] = 0`

The tensor shapes are preserved so existing checkpoints can still be evaluated.
This is important for a checkpoint diagnostic, but a paper-quality low-quality
claim should retrain the model under the same ablated feature setting.

## Added Files

- `pipe_network_completion/low_quality.py` implements the reusable
  location-only graph transform.
- `scripts/train_isarc_low_quality.py` retrains the original anchor-based GNN
  with full or location-only node features.
- `scripts/evaluate_checkpoint.py` now supports
  `--node-feature-mode location-only` for checkpoint diagnostics.
- `tests/test_low_quality_ablation.py` checks that coordinates are preserved,
  non-location node attributes are removed, and optional edge-attribute removal
  works.

## Recommended Runs

Quick diagnostic using the existing full-feature checkpoint:

```powershell
& '.\.conda\pipe-network-completion-cuda\python.exe' scripts\evaluate_checkpoint.py `
  --split test `
  --device cuda `
  --node-feature-mode location-only `
  --output-json outputs\isarc_low_quality_checkpoint_eval\test_location_only.json
```

This answers: "How robust is the already-trained full-feature checkpoint if node
attributes disappear at test time?" It is not a fair retrained low-quality model.

Retrained low-quality model:

```powershell
& '.\.conda\pipe-network-completion-cuda\python.exe' scripts\train_isarc_low_quality.py `
  --device cuda `
  --node-feature-mode location-only `
  --architecture 1212 `
  --hidden-channels 128 `
  --dropout 0.0 `
  --epochs 20 `
  --output-dir outputs\isarc_low_quality_location_only
```

Stricter low-quality variant that also removes edge attributes:

```powershell
& '.\.conda\pipe-network-completion-cuda\python.exe' scripts\train_isarc_low_quality.py `
  --device cuda `
  --node-feature-mode location-only `
  --zero-edge-attrs `
  --epochs 20 `
  --output-dir outputs\isarc_low_quality_location_only_no_edge_attrs
```

## Interpretation

If location-only retraining remains strong, the ISARC task is likely dominated
by spatial proximity, graph structure, and split geometry. If performance drops
substantially, the removed manhole and road node attributes provide material
predictive value.

Because this is still the original anchor-based setup, manhole coordinates are
used as graph nodes. It should not be described as anchor-free.

## Local Run Results

Environment checked before training:

- CUDA available: `True`
- GPU: `NVIDIA GeForce GTX 1080 Ti`

Baseline full-feature checkpoint evaluation on the test split:

- AUC: `0.949688`
- F1: `0.878724`
- Precision: `0.882331`
- Recall: `0.875147`
- Accuracy: `0.879218`
- MCC: `0.758461`

Location-only diagnostic with the full-feature checkpoint, without retraining:

- AUC: `0.725384`
- F1: `0.776925`
- Precision: `0.639876`
- Recall: `0.988683`
- Accuracy: `0.716124`
- MCC: `0.515588`

This diagnostic is a distribution-shift test, not a fair low-quality model.

Location-only retraining, 5 epochs, architecture `1212`, hidden size `128`,
dropout `0.0`, edge attributes retained:

- Best validation AUC: `0.947581` at epoch 4
- Test AUC: `0.949500`
- Test F1: `0.881133`
- Test precision: `0.888458`
- Test recall: `0.873928`
- Test accuracy: `0.882105`
- Test MCC: `0.764312`

Interpretation: in the current split/task, removing non-location node
attributes does not materially reduce performance after retraining. The
published-style result appears to be recoverable from manhole/road locations,
graph structure, and retained edge attributes. This should be reported as a
low-quality anchor-based ablation, not as evidence of anchor-free utility
mapping.
