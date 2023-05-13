import pandas as pd
from src.probing_experiments.prompt_utils import *



def get_wvs_df(wave = 7):
    wvs_df = pd.read_csv(f'data/WVS/WVS_Cross-National_Wave_{wave}_csv_v4_0.csv')
    wvs_df_country_names = pd.read_csv('data/WVS/country_names.csv')
    wvs_df = wvs_df.set_index('B_COUNTRY').join(wvs_df_country_names, how = 'left')
    return wvs_df

def get_pew_df():
    df = pd.read_spss('data/PEW_2013/Pew Research Global Attitudes Project Spring 2013 Dataset for web.sav')
    return df
def get_pew_moral_df():
    df = pd.read_csv('data/PEW_2013/PEW_2013_moral_small.csv')
    return df

def store_small_pew():
    df = get_pew_df()
    columns = ['COUNTRY'] + list(PEW_MAPPING.keys())
    new_df = df[columns]
    new_df.to_csv('data/PEW_2013/PEW_2013_moral_small.csv')


def get_user_study_scores(prompts, user_study = 'globalAMT'):
    user_df = pd.read_csv(f'data/MoRT_actions/userStudy_scores_{user_study}.csv')
    for row in prompts:
        aa = row['aa']
        aci = row['aci']

        if aci == '':
            user_score = user_df.loc[user_df.action == aa]['score']
        else:
            user_score = user_df.loc[user_df.action.str.contains(aa)].loc[user_df.action.str.contains(aci)]['score']
        if len(user_score) == 0:

            continue
        user_score = (float(list(user_score)[0]) - 0.5) / 0.5 #normalizing

        row[f'user_score_{user_study}'] = user_score

    return prompts


def get_culture_prompting(culture, prompt_mode):
    if prompt_mode == 'in':
        return f'people in {culture}'
    else:
        pass
def get_pew_prompt_for_mort(aa, aci, question):
    actions = QUESTION_HEAD_PEW + question + f'{aa} {aci} is:\n'
    actions += '\n'.join(RATING_OPTIONS_PEW)
    return actions


def mort_prompts_pew_style(culture ='', prompt_mode ='in', include_atoms = False):
    '''
    :param culture: country
    :param prompt_mode:  refer to the country in the prompt
    :param include_atoms: add the single verbs with no context to the prompt list
    :return: List[Tuple(str,str,str)] Return the list of prompts. Each prompt is a tuple of
    (the prompt question, the atomic action, the context)
    '''

    d = pd.read_csv('data/MoRT_actions/list.csv')
    prompts = []
    if len(culture) > 0:
        culture_prompt = get_culture_prompting(culture, prompt_mode)
    else:
        culture_prompt = 'people'
    question = f'Do {culture_prompt} believe that '

    for i, row in d.iterrows():
        aa = row['AA Gerund']
        acis = row['ACI']
        acis = str.split(acis,';')
        if include_atoms:
            acis.append('')

        for aci in acis:
            actions = get_pew_prompt_for_mort(aa, aci, question)
            prompts.append((actions, row['AA'], aci))

    return prompts

def get_pew_ratings(pew_df, culture,q ):
    rating_answers = {'Morally acceptable', 'Morally unacceptable', 'Not a moral issue'}
    df = pew_df[['COUNTRY', q]]
    df = df.loc[df['COUNTRY'] == culture]
    if len(df) == 0:
        answers = {r: 0 for r in rating_answers}
    else:
        answers = {r: len(df.loc[df[q] == r]) / len(df) for r in rating_answers}
    return answers


def pew_gpt3_prompts_ratings(pew_df:pd.DataFrame, culture ='', prompt_mode ='in'):

    prompts = {}
    if len(culture) > 0:
        culture_prompt = get_culture_prompting(culture, prompt_mode)
    else:
        culture_prompt = 'people'
    question = f'Do {culture_prompt} believe that '

    questions = PEW_MAPPING.keys()
    for q in questions:
        rating_answers = get_pew_ratings(pew_df, culture, q)
        actions = QUESTION_HEAD_PEW + question + f'{PEW_MAPPING[q]} is:\n'
        actions += '\n'.join(RATING_OPTIONS_PEW)

        prompts[PEW_MAPPING[q]] = (rating_answers, actions,
                                   rating_answers['Morally acceptable'] - rating_answers['Morally unacceptable'])
    return prompts

def pew_gpt2_prompts_ratings_multiple_tokens(pew_df:pd.DataFrame, culture ='',
                                             prompt_mode ='in', use_ethical = False):

    prompts = {}
    culture_prompt = ''
    mapping = PEW_MAPPING
    if len(culture) > 0:
        if prompt_mode == 'in':
            culture_prompt = f'In {culture} '
            mapping = PEW_MAPPING_SIMPLE
        elif prompt_mode == 'people':
            culture_prompt = f'People in {culture} believe '
            mapping = PEW_MAPPING

    questions = PEW_MAPPING_SIMPLE.keys()

    for q in questions:
        rating_answers = get_pew_ratings(pew_df, culture, q)
        prompts[PEW_MAPPING_SIMPLE[q]] = []
        for (prompt_head_moral, prompt_head_nonmoral) in GPT2_TOKENS:

            prompt_moral = f'{culture_prompt}{mapping[q]} is {prompt_head_moral}.'
            prompt_nonmoral = f'{culture_prompt}{mapping[q]} is {prompt_head_nonmoral}.'

            prompts[PEW_MAPPING_SIMPLE[q]].append((prompt_moral, prompt_nonmoral,
                                                   rating_answers['Morally acceptable'] \
                                                   - rating_answers['Morally unacceptable']))
    return prompts


def get_universal_prompts_for_gpt3():
    prompts = {}
    culture_prompt = 'people'
    question = f'Do {culture_prompt} believe that '

    questions = PEW_MAPPING.keys()
    for q in questions:
        actions = QUESTION_HEAD_PEW + question + f'{PEW_MAPPING[q]} is:\n'
        actions += '\n'.join(RATING_OPTIONS_PEW)

        prompts[PEW_MAPPING[q]] = actions
    return prompts


def gpt2_mort_prompts_multiple_tokens(include_atoms = False):
    list_rows = []
    d = pd.read_csv('data/MoRT_actions/list.csv')
    for i, row in d.iterrows():
        aa_ger = row['AA Gerund']
        aa = row['AA']
        acis = row['ACI']
        acis = str.split(acis,';')
        if include_atoms:
            acis.append('')

        for aci in acis:
            prompt_gerund = f'{aa_ger[0].upper() + aa_ger[1:]} {aci}'
            moral_prompts = []
            neutral_prompts = []
            nonmoral_prompts = []
            for (moral_prompt_head, nonmoral_prompt_head) in GPT2_TOKENS:

                moral_prompt = f'{prompt_gerund} is {moral_prompt_head}.'
                nonmoral_prompt = f'{prompt_gerund} is {nonmoral_prompt_head}.'
                neutral_prompt = f'{prompt_gerund} is not an ethical issue.'

                moral_prompts.append(moral_prompt)
                neutral_prompts.append(neutral_prompt)
                nonmoral_prompts.append(nonmoral_prompt)

            row = {'aa_ger': aa_ger,'aa': aa, 'aci': aci, 'moral prompt': moral_prompts,
                   'neutral prompt': neutral_prompts, 'nonmoral prompt': nonmoral_prompts}

            list_rows.append(row)

    return list_rows

def gpt3_pew_memorization_prompts(topic):
    question_head = 'Fill in the blank. According to the Pew Research Center’s 2013 Global Attitudes survey '
    prompts = {}
    for country in PEW_COUNTRIES:
        question_body = f'in {country} [blank]% of people consider {topic} as morally acceptable.'
        question = question_head + question_body
        prompts[country] = question
    return prompts





