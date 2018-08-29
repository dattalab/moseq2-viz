# moseq2-viz
Visualization toolbox for MoSeq2

so you need somewhere in your directory (will search recursively), `pca_scores.h5` and the `results.h5` files

`pca_sores.h5` actually you specify via `moseq2-viz generate-index --pca-file _pca/pca_scores.h5`, e.g.
that stores the path to everything you need
then `moseq2-viz make-crowd-movies moseq2-index.yaml testing.p` if `testing.p` contains a model fit

## Installing the bash completion script
If you want to add bash completion for `moseq2-viz`, add the completion script in your `.bash_profile` or `.bashrc`.
For instance if you want to add it to your `.bash_profile`:

```bash
echo "source /path/to/moseq2-viz/scripts/moseq2-viz-completion.bash" >> ~/.bash_profile`
```

Where `/path/to/moseq2-viz/` is the path to this repo on your computer.
