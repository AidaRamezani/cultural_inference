# Knowledge of cultural moral norms in large language models

Replicating code and data for submission "Knowledge of cultural moral norms in large language models".



Citation: Aida Ramezani and Yang Xu. 2023. Knowledge of cultural moral norms in large language models. In Proceedings of the 61st
Annual Meeting of the Association for Computational
Linguistics (ACL 2023).

## Requirements
```
requirements.txt
jupyter
python >= 3.8.0
```


## Data
* Download World Values Survey from [here](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp),
  and store it in ```data/WVS``` directory.

* Download PEW global views on morality survey from [here](https://www.pewresearch.org/global/2014/04/15/global-morality/country/united-states/),
  and store it in ```data/PEW_2013``` directory.

* Download globalAMT dataset and the respective moral norms
from [here](https://github.com/ml-research/MoRT_NMI/blob/master/MoRT/data/correlation/userstudy/userStudy_scores_globalAMT.csv), 
and [here](https://github.com/ml-research/MoRT_NMI/blob/master/Supplemental_Material/MoralScore/actions_for_subspace/list.csv), 
and store them in ```data/MoRT_actions```.

### Citation
* Christian Haerpfer, Ronald Inglehart, Alejandro
  Moreno, Christian Welzel, Kseniya Kizilova,
  Jaime Diez-Medrano, Marta Lagos, Pippa Norris,
  E Ponarin, and B Puranen. 2021. World Values
  Survey: Round Seven – Country-Pooled Datafile.
  Madrid, Spain & Vienna, Austria: 
JD Systems Institute & WVSA Secretariat. Data File Version, 2(0).
[URL](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp).

* Global Attitudes survey. 
PEW Research Center, 2014, Washington, D.C.,[URL](https://www.pewresearch.org/global/interactives/global-morality/).

* Patrick Schramowski, Cigdem Turan, Nico Andersen,
  Constantin A Rothkopf, and Kristian Kersting. 2022.
  Large pre-trained language models contain human-like biases of what is right and wrong to do.
Nature Machine Intelligence, 4(3):258–268.

## Scripts
### Probing 
To replicate the probing experiments, run the following scripts:
```
python3 src/probing_experiments/compare_prompt_responses/compare_sbert.py
python3 src/probing_experiments/compare_prompt_responses/compare_gpt2.py
python3 src/probing_experiments/compare_prompt_responses/compare_gpt3.py
```

### Fine-tuning
To create the fine-tuning data set run
```angular2html
python3 src/fine_tuning/creating_finetuning_data.py
```
To fine-tune GPT2 models on WVS and PEW run:
```
python3 src/fine_tuning/finetuning.py --model gpt2 --train wvs --test pew
python3 src/fine_tuning/finetuning.py --model gpt2 --train pew --test wvs
```
Follow the example below to store the evaluation results on a fine-tuned model:
```angular2html
python3 src/fine_tuning/eval_for_finetuned.py --model gpt2 --train wvs --strategy random
```

## Notebooks
The display items and results of experiments are shown in notebooks folder.
To replicate the experiments in the paper, run the notebook by the following order:
```
src/notebooks/display_item_1.ipynb
src/notebooks/homogenous_inference_results.ipynb
src/notebooks/fine_grained_analysis.ipynb
src/notebooks/cluster_experiment.ipynb
src/notebooks/cultural_diversities_analysis.ipynb
src/notebooks/clustering_score_difference.ipynb
src/notebooks/finetuned_on_PEW.ipynb
src/notebooks/finetuned_on_WVS.ipynb
src/notebooks/eval_pretrained_on_finetuning_data.ipynb
src/notebooks/homogeneous_inference_fine_tuned.ipynb
```
