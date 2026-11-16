# Instructions for Claude in this folder

Read `README.md` first: it says what this folder is, how data gets in and how to run the models.

- Run everything from this folder with `PYTHONPATH=.` (Windows: `set PYTHONPATH=.`), Python 3.12, the venv in `.venv`.
- This machine may have no internet. New data arrives as a `data_pull_*.zip` made elsewhere with `python pull_data.py`
  and laid in with `python pull_data.py --unpack <zip>`. Do not try to download anything; a fetch fails and the
  code falls back to the files already in the folder.
- Numbers come from the code, never typed by hand: change a script, rerun it, then read the output.
- House CPI paths, UK RPI and the `uk_flows/output`, `projects/`, `flow/` files are fixed inputs (README, *Fixed
  inputs*); use them as they are.
- Never write credentials into files. The only optional key is `BDF_WEBSTAT_API_KEY` (environment variable).
- The spread fair-value model has its own README in `sovereign_cash/debt/note/spread_fv_audit/`.
