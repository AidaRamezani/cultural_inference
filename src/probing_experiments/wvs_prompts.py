import pandas as pd
from src.probing_experiments.prompts import get_culture_prompting
MINUS = 5.5
DIVIDE = 4.5
from src.probing_experiments.prompt_utils import *

def get_wvs_ratings(wvs_df, culture, q):
    df = wvs_df[['StateNme', q]]
    df = df.loc[df['StateNme'] == culture]
    if len(df) == 0:
        return None

    ratings = df.loc[df[q] > 0][q]
    ratings = ((ratings - MINUS) / DIVIDE).mean()
    return ratings


def wvs_gpt3_prompts_ratings(wvs_df:pd.DataFrame, culture ='', prompt_mode ='in', wave = 7, extended_rating = False):

    rating = RATING_OPTIONS_EXTENDED if extended_rating else RATING_OPTIONS_WVS
    prompts = {}
    if len(culture) > 0:
        culture_prompt = get_culture_prompting(culture, prompt_mode)
    else:
        culture_prompt = 'people'
    question_start = f'Do {culture_prompt} believe that '

    questions, questions_text = QUESTION_WAVES[wave]

    for q, q_text in zip(questions, questions_text):
        rating_answers = get_wvs_ratings(wvs_df, culture, q) #is the average ratings from the countru
        actions = GPT3_QUESTION_HEAD + question_start + q_text + ' is:\n'

        actions += '\n'.join(rating)

        prompts[q_text] = (rating_answers, actions)
    return prompts



def wvs_gpt2_prompts_ratings_multiple_tokens(wvs_df:pd.DataFrame, culture ='', prompt_mode ='in', wave = 7,
                                             ):
    prompts = {}
    culture_prompt = ''
    if len(culture) > 0:
        if prompt_mode == 'in':
            culture_prompt = f'In {culture} '
        elif prompt_mode == 'people':
            culture_prompt = f'People in {culture} believe '


    questions, questions_text = QUESTION_WAVES[wave]

    for q,q_text in zip(questions, questions_text):
        rating_answers = get_wvs_ratings(wvs_df, culture, q)
        prompts[q_text] = []
        for (prompt_head_moral, prompt_head_nonmoral) in GPT2_TOKENS:
            prompt_moral = f'{culture_prompt}{q_text} is {prompt_head_moral}.'
            prompt_nonmoral = f'{culture_prompt}{q_text} is {prompt_head_nonmoral}.'

            prompts[q_text].append((prompt_moral, prompt_nonmoral,rating_answers))

    return prompts





