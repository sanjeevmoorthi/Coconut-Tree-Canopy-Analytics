# GitHub Push Checklist

Use this checklist before publishing the project to GitHub.

## 1. Keep the Repository Lightweight

Commit these:

- Source code in `scripts/`
- Configuration files in `config/`
- Documentation in `README.md` and `docs/`
- Demo assets in `assets/`
- Demo image `test_run.jpg`
- Custom model `weights/best.pt`

Do not commit these:

- `.venv/`
- `data/raw/images/`
- `data/raw/annotations/`
- `data/processed/`
- `data/processed_seg/`
- `runs/`
- `outputs/`
- Base model downloads such as `yolov8n.pt` and `yolo11n-seg.pt`
- Local runtime files such as `settings.yaml` and `Ultralytics/`

## 2. If Large Files Were Already Added

`.gitignore` prevents future accidental adds, but it does not remove files that are already tracked. If Git shows large generated folders as tracked, remove them from the Git index without deleting local files:

```bash
git rm --cached -r data/processed data/processed_seg runs outputs .venv
git rm --cached yolov8n.pt yolo11n-seg.pt settings.yaml
```

Then verify:

```bash
git status
```

## 3. Create the GitHub Repository

Create an empty repository on GitHub. Do not initialize it with another README if this local repository already has one.

## 4. Push From Your Machine

Run these commands from the project root:

```bash
git init
git add .
git commit -m "Initial coconut tree detection project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

If the repository already has Git history, skip `git init` and only add the remote if it is missing.

## 5. Verify on GitHub

After pushing, check:

- README images render correctly.
- `weights/best.pt` is present.
- Large folders such as `data/`, `runs/`, `.venv/`, and `outputs/` are not uploaded.
- The Quick Start commands are visible and easy to follow.

## 6. Optional Improvements

- Add a `LICENSE` file.
- Add a small sample dataset or a download link if others need to retrain.
- Add a release with `weights/best.pt` if the model grows too large for normal Git.
- Add tests for tiling, NMS, and CSV output.
