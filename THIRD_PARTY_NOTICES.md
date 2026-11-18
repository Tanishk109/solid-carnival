# Third-party notices

SpamScope includes third-party data and can download third-party model weights. These materials keep
their original licenses; they are not covered by any license that may later be
applied to the original SpamScope source code.

## UCI SMS Spam Collection

- Creators: Tiago A. Almeida and José María Gómez Hidalgo
- Source: [UCI Machine Learning Repository](https://doi.org/10.24432/C5CC84)
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

`data/SMSSpamCollection` is the source corpus. `data/splits.csv` and the files
under `results/` are derived or annotated outputs created by SpamScope. The
project removes normalized duplicate messages and adds fixed train,
validation, and test assignments; those changes are documented in
`data/dataset_manifest.json`.

## sentence-transformers/all-MiniLM-L6-v2

- Authors: Sentence Transformers maintainers and model contributors
- Source: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- License: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

The optional model configuration, tokenizer, documentation, and weights are
downloaded from the upstream package on first use and cached locally. They are
not committed to this repository.
