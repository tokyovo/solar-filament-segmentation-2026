# Source of the template files

`main.bib` and `preamble.tex` come from the official report template, the read-only Overleaf
project `kaggle_comp_report_fi_segm_2026` (https://www.overleaf.com/read/vjztpvrnbrdh#43fbbe),
linked from the competition's Final Submission page. The supervisor session copied them from
the Overleaf viewer on 2026-10-31 (handoff folder
`~/workspace/kaggle/lab/supervisor/handoff/filament-template/`).

- `main.bib`: verbatim. Our own entries are added at the end, below a comment line.
- `preamble.tex`: verbatim; only the two `% ====` separator lines may differ in length.
- The Introduction footnote in `main.tex` is verbatim from the template.
- `template_main.tex`: the template's `main.tex`, reassembled from the Overleaf viewer by the
  supervisor (7,092 bytes with the final newline; the original is 7,091 characters).
- `main.tex` is built from `template_main.tex`: all black text, the subtitle and the
  Acknowledgment are unchanged; only the gray `\guide{}` text, the title, the author and the
  email are replaced, `\guideStyletrue` becomes `\guideStylefalse`, and two `tikz` lines are
  added after `\input{preamble}`. Check with `diff template_main.tex main.tex`.
- Author block: Sensen.ai, Melbourne, Australia (Nhat, 2026-11-12).
