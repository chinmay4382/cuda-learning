# Commit & PR Rules

Rules for creating commits and pull requests in this repository.

---

## Commit Messages

### Format

```
<type>(<scope>): <short summary>

[optional body]
```

- **Subject line:** 50 chars or fewer, no period at the end
- **Body:** wrap at 72 chars, explain *why* not *what*

### Types

| Type | When to use |
|------|-------------|
| `add` | New notebook, new section, new concept |
| `fix` | Corrects wrong code, broken kernel, incorrect output |
| `update` | Improves existing notebook content or explanation |
| `refactor` | Restructures code without changing behavior |
| `docs` | Changes to `.md` files only |
| `chore` | Dependency updates, config, tooling |

### Scopes

Use the notebook number or area being changed:

| Scope | Meaning |
|-------|---------|
| `nb01` … `nb09` | Specific notebook (e.g. `nb03`) |
| `memory` | VRAM / memory management content |
| `kernels` | Numba / CUDA kernel code |
| `llm` | LLM inference notebook |
| `docs` | Markdown reference files |
| `global` | Repo-wide change |

### Examples

```
add(nb05): add CuPy FFT benchmark section

fix(nb03): correct out-of-bounds guard in 2D matrix kernel

update(nb04): clarify AMP GradScaler usage for RTX 3050

docs(memory): add RAM vs VRAM bandwidth comparison table

chore(global): add CLAUDE.md with repo guidance
```

### Rules

- Never use `feat:` or `style:` — not used in this repo
- Do not commit `.ipynb` checkpoints (`.ipynb_checkpoints/` is gitignored)
- Each commit should touch one notebook or one concern — avoid mixing notebook edits with doc edits in a single commit
- If a cell output is wrong or stale, fix the output before committing

---

## Pull Requests

### When to Open a PR

- Adding a new notebook
- Making changes across multiple notebooks at once
- Any change to `CLAUDE.md` or reference `.md` files

Single-notebook fixes can be committed directly to `main` without a PR.

### PR Title

Same format as a commit subject line:

```
add(nb10): CUDA streams and async execution
update(nb04): gradient checkpointing example for 4GB VRAM
```

### PR Body Template

```markdown
## What
One sentence describing what changed.

## Why
Why this change is needed — concept gap, bug, new topic.

## Notebooks affected
- `03_numba_cuda_kernels.ipynb` — added shared memory tiling example
- `CLAUDE.md` — updated architecture patterns section

## Tested on
- GPU: RTX 3050 Laptop (4GB VRAM)
- CUDA: 12.8 / PyTorch 2.11.0+cu128
- All cells run top-to-bottom without error
```

### PR Rules

- All cells in affected notebooks must execute top-to-bottom cleanly before opening a PR
- Output cells should be present (not stripped) so reviewers can verify results
- PRs that add a notebook must also update the sequence table in `CLAUDE.md`
- Keep PRs focused — one topic per PR; don't bundle unrelated notebook edits
