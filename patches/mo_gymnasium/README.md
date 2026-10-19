# mo-gymnasium patch

The symmetric Deep Sea Treasure maps used in this work (`DSTS_MAP` in
`morl_baselines/common/env.py`) are stored as dictionaries `{(row, col): value}`.
The stock `mo-gymnasium` 1.0.1 environment only supports 2D-array maps and
crashes with a `KeyError` on them. The files in this directory are the patched
versions of the corresponding `mo-gymnasium` environments and **must be copied
over the installed package** after `pip install mo-gymnasium==1.0.1`:

```bash
DST_DIR=$(python -c "import mo_gymnasium, os; print(os.path.join(os.path.dirname(mo_gymnasium.__file__), 'envs'))")
cp deep_sea_treasure.py $DST_DIR/deep_sea_treasure/
cp four_room.py        $DST_DIR/four_room/
```

Changed files (vs mo-gymnasium 1.0.1):

| File | Change |
|---|---|
| `deep_sea_treasure.py` | support dict-style `sea_map` (symmetric DST variants) |
| `four_room.py` | minor adaptation used by our experiments |
