# DevBench: A multimodal developmental benchmark for language learning

This is the project repository for DevBench ([preprint](https://doi.org/10.48550/arXiv.2406.10215)), a multimodal benchmark intended to assess Vision–Language Models in terms of their similarities with human responses across development.

Evaluation of an implemented model against DevBench can be conducted as follows:
```
python eval.py model_name
```
where `model_name` is one of `clip_base`, `clip_large`, `blip`, `flava`, `bridgetower`, `vilt`, or `cvcl`.

You can add additional models by constructing a subclass with methods for obtaining image features, text features, and image--text similarity scores.

## Obtaining assets and data
For attribution and licensing reasons, not all assets and data are hosted within this repo. Assets and data can be obtained via the following means:

**(Lexical) Looking While Listening:** Assets are available in this repo. Assets from Adams et al. (2018) and Frank et al. (2016) were directly obtained from the original papers, while assets from Donnelly & Kidd (2021) were reconstructed to ensure licensing at least as permissible as CC-BY-NC-SA. Data are available in this repo; they were aggregated from data in the original papers.

**(Lexical) Visual Vocabulary:** Assets are available from [OSF](https://osf.io/j3mn2/) (these are the same images as in the THINGS similarity task). Data are available in this repo.

**(Grammatical) Test of Reception of Grammar:** Assets can be downloaded from the [LEVANTE repo](https://github.com/levante-framework/core-tasks/tree/main/assets/TROG/original) by running `sh assets/gram-trog/trog_dl.sh`. Data are available in this repo.

**(Grammatical) Winoground:** Assets can be downloaded from [Hugging Face](https://huggingface.co/datasets/facebook/winoground/tree/main/data); download and unzip `images.zip` into `assets/gram-winoground/images`. Data can be downloaded from [Hugging Face](https://huggingface.co/datasets/facebook/winoground/blob/main/statistics/model_scores/human.jsonl); this should go into `evals/gram-winoground`. 

**(Semantic) Free Word Association Task:** Assets and data from children are available in this repo. These were transcribed from Entwisle (1966), but thresholded to remove idiosyncratic responses. Assets and data from adults can be downloaded from the [Florida Free Association Norms](http://w3.usf.edu/FreeAssociation/) by running `sh assets/sem-wat/wat_adult_dl.sh`.

**(Semantic) Visual Object Categorisation:** Assets are available in this repo. These were obtained either from Kiani et al. (2007) via Spriet et al. (2021), or reconstructed to ensure licensing at least as permissible as CC-BY-NC-SA. Data are also available in this repo, converted from SPSS files from the original paper.

**(Semantic) THINGS Similarity:** Assets are available from [OSF](https://osf.io/j3mn2/). Data can be downloaded from [OSF](https://osf.io/w75eu); this should go into `evals/sem-things`.
