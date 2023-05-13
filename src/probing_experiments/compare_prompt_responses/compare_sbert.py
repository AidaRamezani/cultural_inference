from probing_experiments.prompts import *
from probing_experiments.wvs_prompts import \
    COUNTRIES_WVS_W7
from src.probing_experiments.prompts import get_pew_moral_df,get_wvs_df
from src.probing_experiments.compare_prompt_responses.compare_gpt3 import wvs_gpt3_prompts_ratings
import pickle
import pandas as pd


def compare_mcm_pew(rating_path: str = '', saving_path: str = '', save_prompts = True):
    list_rows = []
    if len(rating_path) > 0:
        pew_mcms = pickle.load(open(rating_path, 'rb'))
    else:
        pew_mcms = pickle.load(open('data/PEW_2013/pew_mcm_all.p', 'rb'))
    aggregate_pew_mcm = pew_mcms['universal']
    pew_df = get_pew_moral_df()
    culture_prompts = {}
    pew_topics = list(PEW_MAPPING.values())
    for culture in PEW_COUNTRIES:
        prompts = pew_gpt3_prompts_ratings(pew_df, culture)
        culture_prompts[culture] = prompts
        mcm_scores = []
        rating_scores = []
        text_questions = []
        countries = []

        for question,ratings in prompts.items():
            pew_score = ratings[2] #rating_answers['Morally acceptable'] - rating_answers['Morally unacceptable']
            q_index = pew_topics.index(question)
            culture_pew_mcm = pew_mcms[culture]
            mcm_score = float(culture_pew_mcm[PEW_MCM_PROMPTS[q_index]][0].numpy())
            mcm_scores.append(mcm_score)
            universal_mcm_score = float(aggregate_pew_mcm[PEW_MCM_PROMPTS[q_index]][0].numpy())
            rating_scores.append(pew_score)
            text_questions.append(question)
            countries.append(culture)

            row = {'country': culture, 'question':question, 'mcm_score': mcm_score,
                   'universal_mcm_score' : universal_mcm_score, 'pew_rating':ratings[0], 'pew_score': pew_score}
            list_rows.append(row)

    if save_prompts:
        pickle.dump(culture_prompts, open('data/PEW_2013/compare_to_mcm.p', 'wb'))
    df = pd.DataFrame(list_rows)
    saving_path = 'data/pew_mcm.csv' if saving_path == '' else saving_path
    df.to_csv(saving_path, index = False)

def compare_wvs_mcm(rating_path: str = '',saving_path : str = '',save_prompts = True, wave = 7):
    list_rows = []
    wvs_mcms = pickle.load(open(rating_path, 'rb'))
    aggregate_pew_mcm = wvs_mcms['universal']
    wvs_df = get_wvs_df(wave)
    culture_prompts = {}
    for culture in COUNTRIES_WVS_W7:
        prompts = wvs_gpt3_prompts_ratings(wvs_df, culture, wave=wave)
        culture_prompts[culture] = prompts
        mcm_scores = []
        rating_scores = []
        text_questions = []
        countries = []

        for question,ratings in prompts.items():
            wvs_score = ratings[0]
            culture_pew_mcm = wvs_mcms[culture]
            mcm_score = float(culture_pew_mcm[question + '.'][0].numpy())
            mcm_scores.append(mcm_score)
            universal_mcm_score = float(aggregate_pew_mcm[question + '.'][0].numpy())
            rating_scores.append(wvs_score)
            text_questions.append(question)
            countries.append(culture)

            row = {'country': culture, 'question':question, 'mcm_score': mcm_score,
                   'universal_mcm_score' : universal_mcm_score,  'wvs_score': wvs_score}
            list_rows.append(row)


    if save_prompts:
        pickle.dump(culture_prompts, open(f'data/WVS/compare_to_mcm_w{wave}.p', 'wb'))

    saving_path = f'data/wvs_w{wave}_mcm.csv' if saving_path == '' else saving_path

    df = pd.DataFrame(list_rows)
    df.to_csv(saving_path, index = False)


def compare_mcm():
    compare_mcm_pew('data/PEW_2013/pew_mcm_all.p')
    compare_wvs_mcm('data/WVS/wvs_w7_mcm_all.p')

