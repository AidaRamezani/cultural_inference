from src.probing_experiments.prompts import *
from src.probing_experiments.wvs_prompts import *
import pandas as pd

WVS_RATING_TO_TEXT_MAPPING = {1: 'never justifiable', 2: 'not justifiable',
                      3: 'not justifiable', 4:'not justifiable',
                    5: 'somewhat justifiable', 6:'somewhat justifiable',
                    7: 'justifiable', 8: 'justifiable', 9: 'justifiable', 10: 'always justifiable' }

def get_wvs_prompts_from_people():

    df = pd.read_csv('data/WVS/WVS_Cross-National_Wave_7_csv_v4_0_1.csv')
    df_moral = df[W7_QUESTIONS + ['StateNme']]
    prompts = []
    countries = []
    topics = []
    ratings = []
    ratings_text = []
    prompts_no_judgment = []
    for i, row in df_moral.iterrows():
        country = row['StateNme']
        if pd.isna(country):
            continue
        for j, topic in enumerate(W7_QUESTIONS):
            rating = row[topic]
            if pd.isna(rating) or rating < 1:
                continue
            rating_text = WVS_RATING_TO_TEXT_MAPPING[int(rating)]
            prompt = f'A person in {country} believes {W7_QUESTIONS_TEXT[j]} is {rating_text}.'
            prompt_no_judgment = f'A person in {country} believes {W7_QUESTIONS_TEXT[j]} is'
            countries.append(country)
            topics.append(W7_QUESTIONS_TEXT[j])
            ratings.append(int(rating))
            ratings_text.append(rating_text)
            prompts.append(prompt)
            prompts_no_judgment.append(prompt_no_judgment)


    prompt_df = pd.DataFrame({'prompt': prompts, 'country':countries,
                              'topic': topics, 'rating': ratings, 'rating text': ratings_text,
                              'prompt no judgement': prompts_no_judgment})

    return prompt_df



def get_pew_prompts_from_people():
    RATING_OPTIONS = ['Morally unacceptable', 'Not a moral issue', 'Morally acceptable']
    df_moral = get_pew_moral_df()
    prompts = []
    countries = []
    topics = []
    ratings_text = []
    prompts_no_judgment = []
    for i, row in df_moral.iterrows():
        country = row['COUNTRY']
        if pd.isna(country):
            continue
        for topic, topic_text in PEW_MAPPING.items():
            rating = row[topic]
            if rating not in RATING_OPTIONS:
                continue


            prompt_no_judgment = f'A person in {country} believes {topic_text} is'
            prompt = prompt_no_judgment + f' {rating.lower()}.'
            countries.append(country)
            topics.append(topic_text)
            ratings_text.append(rating)
            prompts.append(prompt)
            prompts_no_judgment.append(prompt_no_judgment)

    prompt_df = pd.DataFrame({'prompt': prompts, 'country':countries,
                              'topic': topics, 'rating text': ratings_text,
                              'prompt no judgement': prompts_no_judgment})

    return prompt_df


if __name__ == '__main__':

    pew_df = get_pew_prompts_from_people()
    pew_df.to_csv('data/finetuning/pew_from_people.csv')
    wvs_df = get_wvs_prompts_from_people()
    wvs_df.to_csv('data/finetuning/wvs_from_people.csv')