# ASTevolve Linux Migration

This is the migration checklist for moving the current ASTevolve repo to a
Linux bastion/GPU machine.

## 1. Clone And Prepare Directories

Recommended layout:

```bash
/srv/astevolve/ast
  cases/
  astevolve/
  engine/
  data/
  model_weights/
  artifacts/
```

After cloning:

```bash
cd /srv/astevolve/ast
mkdir -p model_weights/progen2-small
mkdir -p model_weights/protenix
mkdir -p model_weights/biohub/ESMFold2-Fast
mkdir -p artifacts/tmp
```

`model_weights/` and `artifacts/` are ignored by Git. Only `.gitkeep` files are
kept so the expected folder shape exists in a fresh clone.

## 2. Conda Environment

Create the main ASTevolve runtime environment:

```bash
conda env create -f environments/pytorch-linux.yml
conda activate pytorch
```

If the server already has a compatible CUDA/PyTorch stack, create the
environment manually and then install the pip layer:

```bash
conda create -n pytorch python=3.10 pip -y
conda activate pytorch
pip install -r requirements-gpu.txt
```

Protenix is sensitive to CUDA/PyTorch compatibility. If `pip install protenix`
fails or installs incompatible dependencies, install the server-compatible
Protenix package first, then run:

```bash
pip install -r requirements-gpu.txt --no-deps
```

## 3. Environment Variables

Edit and source the Linux env example:

```bash
cp configs/env.linux.example configs/env.linux.local
vim configs/env.linux.local
set -a
source configs/env.linux.local
set +a
```

Required values:

```bash
ASTEVOLVE_PROJECT_ROOT=/srv/astevolve/ast
ASTEVOLVE_LLM_API_BASE=<OpenAI-compatible endpoint>
ASTEVOLVE_LLM_API_KEY=<key>
```

Model paths default to:

```bash
ASTEVOLVE_MODEL_ROOT=${ASTEVOLVE_PROJECT_ROOT}/model_weights
ASTEVOLVE_PROGEN_MODEL_DIR=${ASTEVOLVE_MODEL_ROOT}/progen2-small
ASTEVOLVE_PROTENIX_ROOT=${ASTEVOLVE_MODEL_ROOT}/protenix
```

The two case OpenEvolve configs now reference `${ASTEVOLVE_LLM_API_BASE}` and
`${ASTEVOLVE_LLM_API_KEY}`; no key should be committed in case configs.

## 4. Download Weights And Assets

Use `model_weights/WEIGHTS_MANIFEST.txt` as the authoritative download and
placement checklist.

At minimum for current TetR/scFv runs:

```text
model_weights/progen2-small/
model_weights/protenix/
data/atf_kb/
data/antibody_kb/
```

Optional ESMFold2 local backend:

```text
model_weights/biohub/ESMFold2-Fast/
```

## 5. Validate

From the project root:

```bash
conda activate pytorch
python scripts/check_assets.py --case tetr_dopamine
python scripts/check_assets.py --case cd25_scfv
python cases/tetr_dopamine/initial_program.py
python cases/cd25_scfv/initial_program.py
```

Run a cheap smoke without Protenix:

```bash
bash scripts/submit_ast_run.sh \
  --case tetr_dopamine \
  --profile smoke \
  --stage inner-smoke \
  --no-conda
```

Run a Protenix/ProGen/KB smoke:

```bash
bash scripts/submit_ast_run.sh \
  --case tetr_dopamine \
  --profile formal \
  --stage inner-smoke \
  --inner-iterations 1 \
  --use-protenix on \
  --external-kb on \
  --external-retrieval on \
  --progen-weight 0.5 \
  --no-conda
```

Run a minimal OpenEvolve outer loop:

```bash
bash scripts/submit_ast_run.sh \
  --case tetr_dopamine \
  --profile formal \
  --stage outer \
  --outer-iterations 2 \
  --inner-iterations 1 \
  --use-protenix on \
  --external-kb on \
  --external-retrieval on \
  --progen-weight 0.5 \
  --no-conda
```

## 6. Git Hygiene

Commit source/config/docs:

```text
astevolve/
engine/
cases/
configs/
docs/
scripts/
environments/
model_weights/**/.gitkeep
model_weights/WEIGHTS_MANIFEST.txt
```

Do not commit:

```text
artifacts/
model_weights/**/*.safetensors
model_weights/**/*.pt
model_weights/**/*.pth
model_weights/**/*.ckpt
.env
configs/*.local
```
