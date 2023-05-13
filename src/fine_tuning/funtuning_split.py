from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
RANDOM_SEED = 113
TEST_RATIO = 0.10
DEV_RATIO = 0.10
MAX_LENGTH = 40
REMOVING_TOPICS = ['homosexuality', 'abortion','sex']


def get_country_topic_pairs_eval(finetuning_df, train_df):
    train_pairs = set([(c, t) for c, t in zip(list(train_df.country), list(train_df.topic))])
    finetuning_pairs = set([(c, t) for c, t in zip(list(finetuning_df.country), list(finetuning_df.topic))])
    eval_pairs = finetuning_pairs - train_pairs
    return eval_pairs


def get_counrty_test(df):
    countries = list(df.country.unique())
    train_size = 1 - (TEST_RATIO + DEV_RATIO)

    countries_train, countries_test = train_test_split(countries,test_size=TEST_RATIO + DEV_RATIO,
                                                       train_size=train_size, random_state=RANDOM_SEED)

    countries_test, countries_dev = train_test_split(countries_test,train_size=TEST_RATIO /  (TEST_RATIO + DEV_RATIO),
                                                     test_size=DEV_RATIO /  (TEST_RATIO + DEV_RATIO),
                                                     random_state=RANDOM_SEED)

    train_df = df.loc[df.country.isin(countries_train)]
    test_df = df.loc[df.country.isin(countries_test)]
    dev_df = df.loc[df.country.isin(countries_dev)]


    return train_df,dev_df,test_df

def get_topic_test(df):
    topics = list(df.topic.unique())
    train_size = 1 - (TEST_RATIO + DEV_RATIO)
    topics_train, topics_test = train_test_split(topics,test_size=TEST_RATIO + DEV_RATIO,
                                                       train_size=train_size, random_state=RANDOM_SEED)

    topics_test, topics_dev = train_test_split(topics_test,train_size=TEST_RATIO /  (TEST_RATIO + DEV_RATIO),
                                                     test_size=DEV_RATIO /  (TEST_RATIO + DEV_RATIO),
                                                     random_state=RANDOM_SEED)

    train_df = df.loc[df.topic.isin(topics_train)]
    test_df = df.loc[df.topic.isin(topics_test)]
    dev_df = df.loc[df.topic.isin(topics_dev)]

    return train_df,dev_df,test_df


def get_removed_topic_test(df: pd.DataFrame):

    df_copy = df.copy()
    df_copy['removing_topic'] = df_copy['topic'] \
        .apply(lambda t :any([key_word in t for key_word in REMOVING_TOPICS]))

    train_df = df_copy.loc[df_copy['removing_topic'] == False]
    test_df = df_copy.loc[df_copy['removing_topic'] == True]

    train_size = 1 - (DEV_RATIO)
    train_df, dev_df = train_test_split(train_df,test_size= DEV_RATIO,
                                                 train_size=train_size, random_state=RANDOM_SEED)


    return train_df,dev_df,test_df


def get_df_from_pairs(pairs):
    def included(row):
        country = row['country']
        topic = row['topic']
        return (country, topic) in pairs

    return included

def get_new_df(df, pairs):
    df_copied = df.copy()
    train_func = get_df_from_pairs(pairs)
    df_copied['included'] = df_copied.apply(train_func, axis = 1)
    new_df = df_copied.loc[df_copied.included == True]
    return new_df


def get_random_test(df):
    topic_pairs = list(set([(c, t) for c, t in zip(list(df.country), list(df.topic))]))

    train_size = 1 - (TEST_RATIO + DEV_RATIO)
    train_pairs, test_pairs = train_test_split(topic_pairs,test_size=TEST_RATIO + DEV_RATIO,
                                               train_size=train_size, random_state=RANDOM_SEED)

    test_pairs, dev_pairs = train_test_split(test_pairs,train_size=TEST_RATIO /  (TEST_RATIO + DEV_RATIO),
                                             test_size=DEV_RATIO /  (TEST_RATIO + DEV_RATIO),
                                             random_state=RANDOM_SEED)

    train_df = get_new_df(df, train_pairs)
    test_df = get_new_df(df, test_pairs)
    dev_df = get_new_df(df, dev_pairs)


    return train_df, dev_df, test_df

def get_balanced_df(df, sample_num = 100):
    sample_df = df.groupby(['country','topic']).sample(sample_num,random_state = RANDOM_SEED)
    return sample_df


def load_dataset(tokenizer,dataname_train, data_name_test,data_class, data_source_type = 'from_people', sample_num = 100,
                 ):
    df = pd.read_csv(f'data/finetuning/{dataname_train}_{data_source_type}.csv')

    cross_test_df = pd.read_csv(f'data/finetuning/{data_name_test}_{data_source_type}.csv')
    cross_test_df['removing_topic'] = cross_test_df['topic']\
        .apply(lambda t :any([key_word in t for key_word in REMOVING_TOPICS]))

    cross_test_df = cross_test_df.loc[~cross_test_df.removing_topic]
    balanced_df = get_balanced_df(df, sample_num)

    datasets = {}
    eval_pairs = {}
    datasets['country_based'] = get_counrty_test(balanced_df)
    eval_pairs['country_based'] = get_country_topic_pairs_eval(balanced_df, datasets['country_based'][0])

    datasets['topic based'] = get_topic_test(balanced_df)
    eval_pairs['topic based'] = get_country_topic_pairs_eval(balanced_df, datasets['topic based'][0])

    datasets['random'] = get_random_test(balanced_df)
    eval_pairs['random'] = get_country_topic_pairs_eval(balanced_df, datasets['random'][0])


    for strategy, (train_df, dev_df, test_df) in datasets.items():
        datasets[strategy] = (data_class(train_df, max_length=MAX_LENGTH, tokenizer= tokenizer),
                                  data_class(dev_df, max_length=MAX_LENGTH, tokenizer= tokenizer),
                                  data_class(test_df, max_length=MAX_LENGTH, tokenizer= tokenizer),
                                  test_df)


    cross_test_dataset = data_class(cross_test_df, max_length=MAX_LENGTH, tokenizer= tokenizer)
    pickle.dump(eval_pairs,open(f'data/{dataname_train}_eval_pairs.p', 'wb'))

    return datasets, cross_test_dataset



