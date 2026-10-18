# CSCI 566: AI-Assisted Image Classification

Read assignment.html and complete the three functions in src/models.py and src/training.py. Review the inherited workflow and handover notes. AI assistance is allowed and optional; you remain responsible for the results and conclusions.

Submit report.pdf of at most two pages and code.zip to the single Gradescope assignment AI-Assisted Image Classification. Both files are required; grading focuses on the report. Use report-template.md; submission-format.md explains upload and failed-run handling. Keep downloaded data, checkpoints, logs, and environments locally.

For assignment help, use a private Piazza post to instructors. Keep assignment questions, code, results, and report excerpts out of public posts. Staff will share general clarifications with the class.

## Setup

Use Python 3.12 and install the supplied dependencies. The code runs on CPU; no GPU is needed. Run all commands from the extracted package directory.

```bash
python -m pip install -r requirements.txt
python prepare_data.py --data-dir ./data
```

Fashion-MNIST downloads automatically. Supplied manifests fix the course subsets: 10,000 training, 2,000 validation, and 2,000 test examples. No additional dataset or API key is needed.

## Implement and Run

Complete SmallCNN.forward, train_one_epoch, and evaluate according to their docstrings. Then run:

```bash
python -m pytest tests/test_interfaces.py
python run_experiment.py --condition baseline --seed 17 --data-dir ./data --output-dir . --overwrite
python run_experiment.py --condition baseline --seed 29 --data-dir ./data --output-dir . --overwrite
python run_experiment.py --condition flip --seed 17 --data-dir ./data --output-dir . --overwrite
python run_experiment.py --condition flip --seed 29 --data-dir ./data --output-dir . 
python select_model.py --output-dir .
python evaluate_test.py --data-dir ./data --output-dir .
```

The files in configs/ record the four settings; the commands above already apply them. Selection compares condition means, breaks ties in favor of baseline, and uses the selected condition's seed-17 checkpoint. Inspect the pipeline and outputs before making claims.

The unfinished starter reports two tests passed and four skipped. After completing all three functions, all six tests should pass without skips. Passing these tests checks the functions, not the validity of the full experiment.

## Retry After a Repair

To repeat a completed experiment, add --overwrite to its run_experiment.py command. After a repair affecting the comparison, rerun all four experiments with --overwrite, then run select_model.py again. Finally rerun evaluate_test.py with --overwrite. Replace the report's values using these new outputs, and briefly explain the repair. Do not change model selection based on test performance.

If any required step fails, leave the affected report entries as N/A and explain the error. Do not reuse results from an earlier attempt.

## Write and Upload

Use results/summary.csv for the table's epochs, n_examples, and accuracy. The final_test row contains test measurements. Consult results/selection.json for the selected model and handover/ for the inherited evidence.

Describe one check and its observed result in the report. No custom check script, figures, separate decision log, or chat transcript is required. Export the report to report.pdf using any editor. Package your working source files with this command from the extracted package directory:

```bash
python -m zipfile -c code.zip src configs manifests tests prepare_data.py run_experiment.py select_model.py evaluate_test.py requirements.txt
```

Upload report.pdf to Report and code.zip to Code in the same Gradescope assignment. Click Submit Answer for each upload. Preview the PDF and confirm both files are present.

If execution fails, enter N/A for missing results and briefly explain the attempted step and error. Submit your current code along with the report.
