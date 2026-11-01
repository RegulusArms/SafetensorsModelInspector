# Safetensors Model Inspector

Inspect `.safetensors` models from a desktop GUI and CLI.

<img width="2537" height="1283" alt="image" src="https://github.com/user-attachments/assets/27ce9f06-4c5f-4b32-aee9-bda85ff576b0" />

## What It Does

- Detects architecture families and variants (Flux, SDXL/SD3, Wan, Hunyuan, Qwen, HiDream, LTX, Z-Image, Chroma, and more)
- Detects adapter type (`LoRA`, `LyCORIS`, `LoHa`, `LoKr`, `DoRA`, `GLoRA`)
- Extracts training metadata when present (steps, epochs, images, resolution, software, and related fields)
- Supports file or folder workflows (including recursive folder scanning)
- Supports `.modelinfo` key dumps for debugging and sharing
- Compares a fine-tuned or merged model against its base model, showing how much each component changed

## Repository Layout

- `gui.py`: GUI only
- `inspect_model.py`: model parsing, detection logic, data extraction, CLI
- `compare_model.py`: tuned-vs-base weight comparison, CLI
- `requirements.txt`: dependencies
- `venv_create.bat`: virtual environment bootstrap helper
- `venv_activate.bat`: activate helper

## Setup

1. Create the virtual environment:

```bat
venv_create.bat
```

2. Activate:

```bat
venv_activate.bat
```

3. Run GUI:

```bat
py gui.py
```

4. Run CLI help:

```bat
py inspect_model.py --help
```

## CLI Usage

### Inspect one or more files

```bat
py inspect_model.py path\to\model1.safetensors path\to\model2.safetensors
```

### Inspect folders

```bat
py inspect_model.py path\to\folder
py inspect_model.py path\to\folder --recursive
```

### JSON output

```bat
py inspect_model.py path\to\folder --recursive --json
```

### Write `.modelinfo` files

```bat
py inspect_model.py path\to\folder --recursive --write-modelinfo
```

### Dump key/debug report text to console

```bat
py inspect_model.py path\to\folder --recursive --dump-keys
```

### Optional alias fallback (filename tokens)

```bat
py inspect_model.py path\to\folder --recursive --allow-filename-alias-detection
```

### Compare a tuned model against its base model

```bat
py compare_model.py path\to\my-finetune.safetensors path\to\base-model.safetensors
py compare_model.py path\to\my-finetune.safetensors path\to\base-model.safetensors --json
```

The two models must share the same architecture. Tensors are read one at a time via memory mapping, so this
works on large checkpoints without loading them into RAM. The report shows:

- Per-component relative change (`||tuned - base|| / ||base||`) and cosine similarity, broken down by
  UNet block, text encoder, and VAE, with a verdict (`unchanged`, `changed`, `partially changed`,
  `rescaled copy`, `different variant`, `replaced / heavily changed`)
- Keys added or removed, shape mismatches, and dtype changes (for example a VAE upcast from FP16 to FP32)
- Tensors containing NaN/Inf values in either model
- Rescaled tensors (same direction, very different magnitude), which usually mean a different VAE variant
  (such as the fp16-fix VAE) was swapped in rather than trained
- The most-changed individual tensors

In the GUI, select two models and click `Compare...` (you will be asked which one is the base), or select
one model and pick the base file in the browse dialog.

## GUI Walkthrough

### Top Area (Input + Controls)

- Drag and drop files or folders into the drop zone
- Use `Browse...` or `Browse Folder...`
- `Analyze` processes queued inputs
- `Settings` controls visibility and behavior
- `Minimize` / `Restore` collapses or expands the top area for more workspace
 
<img width="2547" height="373" alt="image" src="https://github.com/user-attachments/assets/419e5d42-e3f2-469e-8850-633720ac7782" />


### Tab: Simple Cards

- Lightweight model cards
- Supports card selection, multi-select, and context menu actions

<img width="1323" height="369" alt="image" src="https://github.com/user-attachments/assets/09cae186-42b5-4c57-be21-611ff8a11396" />

### Tab: Detailed Cards

- Full card details with configured metadata visibility
- Supports card selection, multi-select, and context menu actions

<img width="1708" height="1076" alt="image" src="https://github.com/user-attachments/assets/a146a5a7-3a9f-422f-8eee-64efb36af715" />

- Supports specific LoRA formats like LoHa, LoKr, GLoRa
- Some fail sometimes (lycoris)

<img width="2526" height="953" alt="image" src="https://github.com/user-attachments/assets/1ef32b95-868a-4407-8569-8207d68eac3a" />



### Tab: Data

- Sortable/resizable table
- Multi-select cells and copy via `Ctrl+C`
- Right-click actions (`View Raw`, `Copy Selected Entries`)
- Column visibility can be configured in settings

<img width="2385" height="257" alt="image" src="https://github.com/user-attachments/assets/1dcd1a23-ca36-433e-8e77-9252cfcc0208" />



### Tab: Raw

- Per-model raw `.modelinfo` text view
- `View Raw` context action jumps here for the selected model
- `Ctrl+C` copies the selected text, or the full raw content when no selection exists

<img width="2442" height="726" alt="image" src="https://github.com/user-attachments/assets/4c2f9d4d-1476-4348-b872-06c282a80007" />


## Notes

- Folder drag/drop and folder browse both support recursive discovery of `.safetensors`.
- Filtering in the UI affects visibility and copy behavior (hidden rows are excluded from table copy).
- `.modelinfo` output is generated by shared backend logic in `inspect_model.py`.
- Filename alias detection is opt-in in Settings and can map filename tokens to fallback labels.
- `Pony7` is treated as distinct from `PDXL`. The alias tokens `pony7`, `ponyv7`, and `pony v7` map to `Pony7`.

## Settings (Current)

### General

- `Filename Alias Detection`: optional filename-token fallback for special labels
- `Auto-minimize top section on Analyze`
- `Auto-analyze when files are added`
- `File add behavior`:
  - `Replace current input list`
  - `Append to current input list`
- `Default tab`: `Simple Cards`, `Detailed Cards`, `Data`, or `Raw`

### Visibility Groups

- `Simple Cards`: choose which data fields are shown
- `Detailed Cards`: choose which data fields are shown
- `Data Columns`: choose visible columns in the Data tab
