# Week 03 — Bug Hunt and Test-Driven Repair of an AI-Generated Physics Script

PHYS 3500/7440 · Thanh Ly

## Files

| file | purpose |
|---|---|
| `ai_generated_oscillator.py` | the original buggy script from the course repo (`docs/exercises/week03/`), unmodified |
| `ai_generated_oscillator_fixed.py` | corrected script; each fix is commented with its bug number |
| `test_quantum_oscillator.py` | pytest suite of 7 physics tests (15 test cases); module under test chosen by `QHO_MODULE` |
| `ai_prompt.md` | the exact prompt pasted into claude.ai; it embeds the test suite as it stood at prompting time (tests 1–6). Test 7 was added afterwards and Claude's unchanged output passes it too (notebook §5.4) |
| `ai_regenerated.py` | Claude's output from that prompt, byte-for-byte as downloaded; passes all 15 cases on the first attempt |
| `week03_Ly.ipynb` | notebook: test suite, pytest output for the buggy / fixed / AI-regenerated scripts, mutation check, the prompt and strategy annotation, reflection |
| `week03_Ly_bugreport.md` | observations and the structured five-bug report with Griffiths citations |
| `requirements.txt` / `environment.yml` | pinned package versions (identical pins in both) |

## Reproducing the results

From a fresh clone, create an environment and install the pinned packages with either `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` or `conda env create -f environment.yml && conda activate phys7440-week03` (Python 3.11 or newer is required by the pinned numpy and scipy; the notebook outputs in this submission were produced with Python 3.14.7 and exactly the pinned package versions, on Linux). Change into this directory — every path in the notebook and tests is relative, so it must be the working directory. To run the tests from the command line, select the module with an environment variable: `QHO_MODULE=ai_generated_oscillator pytest -v test_quantum_oscillator.py` reproduces the failing run against the buggy script (15 failed), `QHO_MODULE=ai_generated_oscillator_fixed pytest -v test_quantum_oscillator.py` reproduces the passing run (15 passed), and `QHO_MODULE=ai_regenerated pytest -v test_quantum_oscillator.py` runs the suite against Claude's regenerated script (15 passed). To reproduce the notebook, launch `jupyter lab`, open `week03_Ly.ipynb` and choose *Kernel → Restart Kernel and Run All Cells*; it loads the same test file in-process via `ipytest`, so no cell needs to be edited, and the mutation-check cell rebuilds its single-bug variants in a temporary directory each time. Nothing here is random, so no seeds are needed, and no API keys are used — the AI step was carried out interactively at claude.ai and the prompt and response are reproduced verbatim in the notebook and in the two `ai_*` files.
