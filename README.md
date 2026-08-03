# Experiment Inference Notes

A [Quarto](https://quarto.org/) book on asymptotic inference for online experiments.

Home page: https://dmalison.github.io/experiment-inference-notes/

## Render locally

The book is built with Quarto 1.9.37. Render it from the repository root:

```sh
quarto render
```

The rendered book is written to `_book/`. Use `quarto preview` for a local preview.

## Publish

Pushing to `main` runs the GitHub Pages workflow in `.github/workflows/publish.yml`. Before the first workflow deployment, set the repository's Pages source to **GitHub Actions** under **Settings > Pages**.