# gemma-asr: Vietnamese speech-to-text from EmbeddingGemma 2

Turns [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2), a multimodal *embedding*
model, into a fast non-autoregressive Vietnamese ASR model. The pretrained audio encoder stays frozen; only a
small CTC "reader" (~22M params) is trained on top of it.

```
16 kHz wav ─► log-mel (GPU, identical to Gemma4AudioFeatureExtractor)
          ─► EmbeddingGemma 2 audio encoder (USM-style conformer, 300M, 12 layers, 25 frames/s)  [frozen]
          ─► learned softmax mix of its 13 layer outputs
          ─► reader: Linear + depthwise conv + 6 × transformer (d=512)
          ─► head over ~10-20k units = Gemma tokenizer tokens seen in Vietnamese text
             (rows initialised from Gemma's own token embeddings, i.e. "the model's text space")
          ─► CTC, greedy decoding
```

Why Gemma tokens as output units: Gemma tokenizes Vietnamese at roughly one token per syllable
(`▁xin ▁chào ▁các ▁bạn`), so this is effectively a syllable vocabulary with sub-word fallback for rare
syllables and foreign words. Unit ids: 0 = blank, 1 = `<unk>`, 2.. = Gemma tokens.

## Layout

```
gemma_asr/
  text.py      Vietnamese normalization (NFC, lowercase, punctuation, tone-mark placement hoà→hòa, digit filter)
  features.py  batched GPU log-mel matching the HF feature extractor (verified to ~1e-5)
  units.py     CTC unit inventory (subset of Gemma tokens)
  model.py     GemmaCTC: frozen encoder + reader + head, checkpoint save/load, greedy CTC decoding
  data.py      sharded streaming train loader with duration-bucketed dynamic batches; eval sets
  scoring.py   WER / CER / tone-insensitive WER, summed across GPUs
scripts/
  prepare_data.py  download HF datasets → uniform FLAC parquet shards (parallel, resumable)
  build_vocab.py   build units.json from training transcripts
  train.py         training (single GPU or torchrun DDP)
  evaluate.py      WER table on prepared test/valid sets
  transcribe.py    transcribe audio files of any length
```

## 0. Setup

```bash
git clone/copy this directory to the machine, then:
python -m venv .venv && source .venv/bin/activate      # or: uv venv -p 3.11 .venv
pip install torch                                       # pick the wheel matching your CUDA driver
pip install -r requirements.txt
hf auth login                                           # needed for the gated datasets
```

`transformers>=5.19` is required (EmbeddingGemma 2 support). Check that torch sees the GPUs:
`python -c "import torch; print(torch.cuda.device_count())"`. If torch says the driver is too old, install a
wheel built for an older CUDA, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu126`.

**Gated datasets.** Click "agree / request access" on the HF page while logged in with the same account:
- [linhtran92/viet_bud500](https://huggingface.co/datasets/linhtran92/viet_bud500) (auto-approved)
- [capleaf/viVoice](https://huggingface.co/datasets/capleaf/viVoice) (manual approval, can take days)
- optional: [thivux/phoaudiobook](https://huggingface.co/datasets/thivux/phoaudiobook) (manual, off by default)

If a download fails because access is missing, `prepare_data.py` prints a warning and continues with the rest.

## 1. Download + prepare data

```bash
python scripts/prepare_data.py --out_dir data/prepared --raw_dir data/raw --workers $(nproc)
```

Default sources (~2,300 h of training audio before filtering, ~370 GB to download):

| source | hours (approx.) | splits used |
|---|---|---|
| viVoice (YouTube) | ~1000 | train |
| Bud500 | ~500 | train / valid / test |
| InfoRe 2 audiobooks | ~415 | train |
| VLSP2020 (VinAI) | ~100 | train |
| LSVSC | ~100 | train / valid / test |
| ViMD (63 dialects) | ~100 | train / valid / test |
| FPT FOSD | ~30 | train |
| InfoRe 1 | ~25 | train |
| FLEURS vi_vn | ~10 | train / valid / test |
| Common Voice 17 vi | ~5 | train / valid / test |
| GigaSpeech 2 vi (dev/test only) | ~20 | valid / test |
| PhoAudiobook (opt-in: `--sources ... phoaudiobook`) | ~940 | train / valid / test |

What it does per utterance: decode → mono → resample to 16 kHz (soxr) → normalize transcript → drop if it
contains digits or non-Vietnamese/Latin characters, or is shorter than 0.5 s / longer than 30 s → write FLAC
bytes into `data/prepared/<split>/<source>/*.parquet`. A summary table (hours kept / dropped per source)
is printed at the end and saved to `data/prepared/manifest.json`.

- Re-running skips finished shards, so it's safe to restart after an interruption.
- Disk: raw ≈ 370 GB, prepared ≈ 150 GB. Add `--delete_raw` to delete each raw file once it's converted.
- Pick sources with `--sources bud500 vlsp2020 fleurs ...`.
- Test the whole thing on a tiny subset first: `--sources fleurs cv17 fpt_fosd --out_dir data/smoke`.

## 2. Build the unit vocabulary

```bash
python scripts/build_vocab.py --data_dir data/prepared        # writes data/prepared/units.json
```

Prints the vocab size and token coverage. Tokens seen fewer than `--min_count 2` times map to `<unk>`.

## 3. Train (8×H100)

```bash
torchrun --nproc_per_node 8 scripts/train.py \
    --data_dir data/prepared --out_dir runs/vi \
    --max_steps 20000 --max_batch_seconds 1200 --num_workers 10
```

- `--max_batch_seconds` is padded audio *per GPU* per step, so the global batch is 8 × 1200 s ≈ 2.7 h of audio.
- Every `--eval_every` (2000) steps it prints WER / CER / WER-without-tones on each `valid` set plus an
  example ref/hyp, and saves `runs/vi/best.pt` (best average valid WER). `runs/vi/last.pt` is saved every
  `--save_every` steps. Metrics are also appended to `runs/vi/metrics.jsonl`. Add `--wandb <project>` for W&B.
- Resume after an interruption with `--resume runs/vi/last.pt`.
- Every 500 steps the log prints the learned encoder layer weights, which show which encoder layers the reader uses.

**Expected speed.** Measured on one RTX A6000: ~1,300× real time (0.42 s/step at 600 s of audio per batch),
GPU at 99%, 9.5 GB memory. An H100 should be roughly 2.5-4× faster, so 8×H100 should process about
25-40k seconds of audio per second. That makes the default 20k steps (~53k hours of audio, ~20 passes over
the data) take roughly **1.5-2.5 h**. These are estimates; check the `x realtime` value in the log.
Watch the valid WER: if it stops improving early, stop the run or lower `--max_steps`.

**Data-loader throughput.** At these speeds the loaders must decode ~30k seconds of FLAC per second.
10 workers per GPU (80 total) is fine on a typical 100+ core H100 node. If the log's `x realtime` is low and
the GPUs aren't at ~100%, raise `--num_workers`; on machines with fewer cores, lower it.

## 4. Evaluate

```bash
torchrun --nproc_per_node 8 scripts/evaluate.py --ckpt runs/vi/best.pt --data_dir data/prepared --split test
# single GPU also works: python scripts/evaluate.py --ckpt runs/vi/best.pt --split test
```

Prints WER / CER / tone-insensitive WER per test set. WER is computed on space-separated syllables, the
standard measure for Vietnamese. `--dump hyps.jsonl` writes ref/hyp pairs for error analysis.

## 5. Transcribe

```bash
python scripts/transcribe.py --ckpt runs/vi/best.pt some_audio.wav podcast.mp3
```

Any length: long audio is split into ≤30 s chunks at the quietest point near each boundary and decoded in
batches. Output is lowercase without punctuation, matching the training normalization.

## Smoke-test result (one A6000)

Trained on only 39 h (FLEURS + Common Voice + FPT FOSD), 3000 steps × 600 s batches, ~20 minutes:

| valid set | step 1000 | step 2000 | step 3000 |
|---|---|---|---|
| FLEURS: WER / WER without tones | 28.3% / 26.1% | 23.7% / 22.4% | 21.0% / 19.6% |
| Common Voice: WER / WER without tones | 33.2% / 29.8% | 26.6% / 24.7% | 25.1% / 23.4% |

Tone-only errors are a small part of the total, so the frozen embedding encoder keeps tone information. The
remaining errors mostly come from having very little training data, which the full ~2,300 h run fixes.

## Experiments worth trying (flags on `train.py`)

| flag | what it tests |
|---|---|
| `--unfreeze_top_k 2 --encoder_lr 5e-5` | fine-tune the top encoder layers (slower; can't keep the encoder "untouched") |
| `--head_init random` | is initializing the head from Gemma's text embeddings actually helping? |
| `--freeze_head` | pure "nearest token in the model's text space" (only the reader learns) |
| `--no_layer_mix` | use only the last encoder layer |
| `--n_layers 8 --d_model 768` | bigger reader |
| `--source_weights '{"vimd": 3, "fleurs": 3}'` | upsample small or dialect-heavy sources |
| `--no_specaug` | SpecAugment ablation |

## Notes / limitations

- Utterances containing digits are dropped (no reliable way to know how a number was read aloud), so the
  model outputs numbers as words. Common Voice / FLEURS lose 0-25% of utterances to this filter.
- Tone-mark placement is normalized to the old style (hòa, khỏe, thủy) for both training and scoring.
- Decoding is greedy CTC with no language model. Adding an n-gram LM with beam search (e.g. pyctcdecode) is an
  easy improvement for later.
- The encoder looks only backward (no right context), so streaming is possible in principle. The reader as written
  is bidirectional; make its attention causal or chunked if you want true streaming.
