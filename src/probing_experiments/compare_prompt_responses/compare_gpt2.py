from probing_experiments.prompts import *
from probing_experiments.wvs_prompts import \
    COUNTRIES_WVS_W7,wvs_gpt2_prompts_ratings_multiple_tokens
from src.probing_experiments.prompts import get_pew_moral_df,get_wvs_df, get_user_study_scores
from model_utilities import get_lines_log_prob_last_token,get_lines_log_prob_matrix
from src.my_models import *
import pickle
import pandas as pd
import numpy as np


def compare_moral_non_moral_probs(prompts, modelname, use_cuda = True):
    from src.my_models import get_gpt2_model

    tokenizer, model = get_gpt2_model(modelname,use_cuda=use_cuda)
    for i , row in enumerate(prompts):
        moral_prob, non_moral_prob = get_lines_log_prob_matrix([row['moral prompt'], row['nonmoral prompt']],
                                                               model,tokenizer, use_cuda)
        row['moral log prob'] = moral_prob
        row['nonmoral log prob'] = non_moral_prob
        row['log prob difference'] = (moral_prob - non_moral_prob)

    return prompts


def compare_gpt2_pew_token_pairs(cultures : list = None,
                                 model_name = 'gpt2',
                                 excluding_topics : list = [],
                                 model = None,
                                 tokenizer = None,use_cuda = False,
                                 prompt_mode = 'in'):

    if model == None:
        tokenizer, model = get_gpt2_model(model_name, use_cuda)

    model.eval()
    pew_df = get_pew_moral_df()
    if cultures == None:
        cultures = list(pew_df['COUNTRY'].unique())
    cultures.append('')
    gpt2_all = []
    for culture in cultures:
        prompts = pew_gpt2_prompts_ratings_multiple_tokens(pew_df, culture, prompt_mode=prompt_mode)
        culture = culture if culture != '' else 'universal'

        country_rows = []

        for question,rating_pairs in prompts.items():

            if any([x in question for x in excluding_topics]):
                print(question)
                continue
            lm_score = get_log_prob_difference(rating_pairs, 0, 1, model, tokenizer, use_cuda)
            pew_score = rating_pairs[0][2]

            row = {'country': culture,
                   'topic': question,  'pew_score': pew_score,
                   'log prob difference' : lm_score}
            gpt2_all.append(row)
            country_rows.append(row)

    df = pd.DataFrame(gpt2_all)

    save_dir = f'data/pew_{model_name}_token_pairs.csv'
    df.to_csv(save_dir, index = False )


def compare_gpt2_wvs_token_pairs(cultures : list = None, wave = 7,model_name = 'gpt2',
                                 excluding_topics = [], excluding_cultures = [], model = None, tokenizer =None,
                                 use_cuda = False,prompt_mode = 'in'):
    if model == None:
        tokenizer, model = get_gpt2_model(model_name,use_cuda)

    wvs_df = get_wvs_df(wave)
    if cultures == None:
        cultures = COUNTRIES_WVS_W7
    cultures.append('')

    gpt2_all = []

    for culture in cultures:
        if culture in excluding_cultures:
            continue
        prompts = wvs_gpt2_prompts_ratings_multiple_tokens(wvs_df, culture, prompt_mode)
        culture = culture if culture != '' else 'universal'
        rating_scores = []
        text_questions = []
        country_rows = []
        for question,rating_pairs in prompts.items():
            if any([x in question for x in excluding_topics]):
                continue
            moral_log_probs = []
            nonmoral_log_probs = []
            lm_score = get_log_prob_difference(rating_pairs, 0, 1, model, tokenizer, use_cuda)
            wvs_score = rating_pairs[0][2]
            rating_scores.append(wvs_score)
            text_questions.append(question)
            row = {'country': culture, 'topic': question,  'wvs_score': wvs_score,
                   'moral log prob' : np.mean(moral_log_probs),
                   'non moral log probs': np.mean(nonmoral_log_probs),
                   'log prob difference' : lm_score}

            gpt2_all.append(row)
            country_rows.append(row)


    df = pd.DataFrame(gpt2_all)
    save_dir = f'data/wvs_w{wave}_{model_name}_token_pairs_{prompt_mode}.csv'
    df.to_csv(save_dir, index = False )


def get_log_prob_difference(pairs, moral_index, nonmoral_index, model, tokenizer, use_cuda):

    question_average_lm_score = []
    for rating in pairs:
        moral_prompt = rating[moral_index]
        nonmoral_promopt = rating[nonmoral_index]

        logprobs = \
            get_lines_log_prob_last_token([moral_prompt, nonmoral_promopt], model, tokenizer, use_cuda)

        lm_score = logprobs[0] - logprobs[1]
        question_average_lm_score.append(lm_score)
    lm_score = np.mean(question_average_lm_score)
    return lm_score


def compare_paired_moral_non_moral_probs(prompts, modelname, use_cuda = True,
                                         model = None, tokenizer = None):

    from src.my_models import get_gpt2_model
    if model == None:
        tokenizer, model = get_gpt2_model(modelname,use_cuda)

    for row in prompts:
        moral_prompts = row['moral prompt']
        nonmoral_prompts = row['nonmoral prompt']
        pairs = [(m, nonm) for m , nonm in zip(moral_prompts,nonmoral_prompts)]
        q_lm_scores = get_log_prob_difference(pairs, 0, 1 , model, tokenizer, use_cuda)
        row['log prob difference'] = q_lm_scores

    return prompts

def compare_gpt2_pair_prompts_mort_user_study(style = 'mv_at_end',modelname = 'gpt2',
                                              model = None, tokenizer = None, use_cuda = True):

    prompts = gpt2_mort_prompts_multiple_tokens(include_atoms= True)

    prompts = get_user_study_scores(prompts, user_study = 'globalAMT')
    new_prompts = compare_paired_moral_non_moral_probs(prompts, modelname,
                                                       use_cuda, model, tokenizer)
    modelname = modelname.split('/')[0]
    pickle.dump(new_prompts, open(f'data/MoRT_actions/'
                                  f'prompts_{modelname}_'
                                  f'pair_logprob_userstudy_globalAMT_style_'
                                  f'{style}_use_last_token_True.p', 'wb'))

def compare_gpt2s():
    store_small_pew()
    model_names = ['gpt2', 'gpt2-medium','gpt2-large']
    for mn in model_names:
        compare_gpt2_pair_prompts_mort_user_study(style='mv_at_end', modelname=mn)
        compare_gpt2_wvs_token_pairs(COUNTRIES_WVS_W7,model_name= mn, use_cuda=True, prompt_mode='in')
        compare_gpt2_pew_token_pairs(cultures=PEW_COUNTRIES, model_name=mn, use_cuda=True, prompt_mode = 'in')

if __name__ =='__main__':
    compare_gpt2s()