# If We May De-Presuppose: Robustly Verifying Claims through Presupposition-Free Question Decomposition

## Setup

```bash
uv venv --python 3.12
uv pip sync requirements.txt
```

## Run

change the variables in `scripts/grid_search_reasoner.py` to run different models and datasets.

```bash
python scripts/grid_search_reasoner.py
```

## Results

Results are saved in `outputs/` directory.

## Evaluate Questions

change the `ROOT_DIRS` in `scripts/eval_wice_ques.py` to evaluate different models and datasets.
```bash
python src/eval_wice_ques.py
```
