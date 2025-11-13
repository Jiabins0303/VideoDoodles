# Setup Instructions for 3D Canvas Placement Prototype

## Prerequisites

1. Python 3.8 or higher
2. pip package manager
3. Git (for cloning Video-Depth-Anything)

## Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Setup Video-Depth-Anything

### Clone the Repository

```bash
git clone https://github.com/DepthAnything/Video-Depth-Anything
cd Video-Depth-Anything
```

### Install Video-Depth-Anything Dependencies

```bash
pip install -r requirements.txt
```

### Download Model Weights

The repository provides a script to download model weights:

```bash
bash get_weights.sh
```

Alternatively, you can manually download the model weights:

**For Small Model (recommended for speed):**
```bash
mkdir -p checkpoints
cd checkpoints
wget https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
cd ..
```

**For Base Model:**
```bash
cd checkpoints
wget https://huggingface.co/depth-anything/Video-Depth-Anything-Base/blob/main/video_depth_anything_vitb.pth
cd ..
```

**For Large Model (best quality, slower):**
```bash
cd checkpoints
wget https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth
cd ..
```

### Verify Installation

The model weights should be in the `Video-Depth-Anything/checkpoints/` directory:
- `video_depth_anything_vits.pth` (Small model)
- `video_depth_anything_vitb.pth` (Base model)
- `video_depth_anything_vitl.pth` (Large model)

## Step 3: Enable Jupyter Widgets

If using Jupyter Notebook (not JupyterLab):

```bash
jupyter nbextension enable --py widgetsnbextension
```

If using JupyterLab:

```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

## Step 4: Verify Setup

1. Open the notebook: `prototype_frame_canvas.ipynb`
2. Run the first cell (Setup & Imports) to verify all imports work
3. Check that Video-Depth-Anything is accessible:
   ```python
   from pathlib import Path
   vda_path = Path("Video-Depth-Anything")
   print(f"Video-Depth-Anything found: {vda_path.exists()}")
   ```

## Troubleshooting

### Issue: Video-Depth-Anything import fails

**Solution:** Make sure you've cloned the repository and it's in the same directory as the notebook:
```bash
ls Video-Depth-Anything  # Should show the repository contents
```

### Issue: Model weights not found

**Solution:** Verify the checkpoint file exists:
```bash
ls Video-Depth-Anything/checkpoints/
```

If missing, download the weights using the instructions above.

### Issue: CUDA/GPU not available

**Solution:** The code will automatically fall back to CPU if CUDA is not available. For faster processing, install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Jupyter widgets not working

**Solution:** 
- For Jupyter Notebook: `jupyter nbextension enable --py widgetsnbextension --sys-prefix`
- For JupyterLab: `jupyter labextension install @jupyter-widgets/jupyterlab-manager`
- Restart Jupyter after installation

## Notes

- The Small model (vits) is recommended for faster processing during prototyping
- The Large model (vitl) provides better quality but is slower
- Processing time depends on video length and resolution
- GPU acceleration significantly speeds up depth estimation

