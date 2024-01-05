from typing import List
import pandas as pd
import random
import math
import numpy
import pickle


def compute_f(id, df_1: pd.DataFrame, df_2: pd.DataFrame) -> float:
    # df must have 3 columns: id_1, i_d2, contact_duration

    # day 1
    contacts_1_of_id = df_1[(df_1["id_1"] == id) | (df_1["id_2"] == id)]
    V_1 = set(contacts_1_of_id['id_1']).union(set(contacts_1_of_id['id_2']))
    # all individuals that id meets excluding itself
    V_1.discard(id)
    if len(V_1) == 0:
        return 0

    # day 2
    contacts_2_of_id = df_2[(df_2["id_1"] == id) | (df_2["id_2"] == id)]
    V_2 = set(contacts_2_of_id['id_1']).union(set(contacts_2_of_id['id_2']))
    # all individuals that id meets excluding itself
    V_2.discard(id)

    f = len(V_1.intersection(V_2))/len(V_1)
    return f


def compute_avg_f(df_1: pd.DataFrame, df_2: pd.DataFrame) -> float:
    unique_ids = get_unique_ids(df_1=df_1, df_2=df_2)

    f_scores = [compute_f(id, df_1=df_1, df_2=df_2) for id in unique_ids]
    return sum(f_scores)/len(f_scores)


def get_unique_ids(df_1: pd.DataFrame, df_2: pd.DataFrame) -> numpy.ndarray:
    all_ids = pd.concat([df_1["id_1"], df_1["id_2"],
                        df_2["id_1"], df_2["id_2"]])
    unique_ids = pd.unique(all_ids)
    return unique_ids


def update_time(df: pd.DataFrame, iter: int):
    df_copy = df.copy()
    # update time for each synthetic 2-day data
    current_day_in_second = 2*24*60*60*iter
    df_copy["time"] = df_copy["time"]+current_day_in_second
    return df_copy


def replace_i_with_j(df: pd.DataFrame, tag_i, tag_j) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy.loc[df_copy['id_2'] != tag_j,
                'id_1'] = df_copy.loc[df_copy['id_2'] != tag_j, 'id_1'].replace(tag_i, tag_j)
    df_copy.loc[df_copy['id_1'] != tag_j,
                'id_2'] = df_copy.loc[df_copy['id_1'] != tag_j, 'id_2'].replace(tag_i, tag_j)

    return df_copy


def shuffle_i_and_j(df: pd.DataFrame, tag_i, tag_j) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['id_1'] = df_copy['id_1'].replace(
        {tag_i: tag_j, tag_j: tag_i})
    df_copy['id_2'] = df_copy['id_2'].replace(
        {tag_i: tag_j, tag_j: tag_i})

    return df_copy


def data_processing(df: pd.DataFrame):
    df_copy = df.copy()
    # Switch id_1 and id_2 where id_1 > id_2
    mask = df_copy['id_1'] > df_copy['id_2']
    df_copy.loc[mask, ['id_1', 'id_2']
                ] = df_copy.loc[mask, ['id_2', 'id_1']].values

    # Sort DataFrame by 'id_1', 'id_2', and 'time'
    df_copy.sort_values(['id_1', 'id_2', 'time'], inplace=True)
    df_copy.reset_index(drop=True, inplace=True)

    # Identify and aggregate continuous contacts
    df_copy['group'] = -((df_copy['id_1'] == df_copy['id_1'].shift()) &
                         (df_copy['id_2'] == df_copy['id_2'].shift()) &
                         (df_copy['time'] == df_copy['time'].shift() + df_copy['contact_duration']))
    df_copy['group'] = df_copy['group'].cumsum()

    df_copy = df_copy.groupby(['id_1', 'id_2', 'group']).agg(
        {'time': 'last', 'contact_duration': 'sum'}).reset_index()

    # Sort the result by 'time'
    df_copy.sort_values('time', inplace=True)
    df_copy = df_copy.drop('group', axis=1)

    df_copy.rename(columns={'time': 'end_moment'}, inplace=True)
    df_copy["start_moment"] = df_copy["end_moment"] - \
        df_copy["contact_duration"]

    return df_copy


def repetitive_generating(no_iteration: int, df_1: pd.DataFrame, df_2: pd.DataFrame) -> List[pd.DataFrame]:
    dfs = [df_1.copy(), df_2.copy()]
    iter = 1
    while iter <= no_iteration:
        df_1_copy = df_1.copy()
        df_2_copy = df_2.copy()

        df_1_copy = update_time(df_1_copy, iter)
        df_2_copy = update_time(df_2_copy, iter)

        dfs.append(df_1_copy)
        dfs.append(df_2_copy)

        iter += 1

    return [data_processing(df=df) for df in dfs]


def random_shuffle_generating(no_iteration: int, df_1: pd.DataFrame, df_2: pd.DataFrame) -> List[pd.DataFrame]:
    dfs = [df_1.copy(), df_2.copy()]

    unique_ids = get_unique_ids(df_1=df_1, df_2=df_2)
    iter = 1
    while iter <= no_iteration:
        # Choose two tag Ids at random
        tag_i, tag_j = random.sample(list(unique_ids), k=2)

        transformed_df_1 = shuffle_i_and_j(df=df_1, tag_i=tag_i, tag_j=tag_j)
        transformed_df_2 = shuffle_i_and_j(df=df_2, tag_i=tag_i, tag_j=tag_j)

        transformed_df_1 = update_time(transformed_df_1, iter)
        transformed_df_2 = update_time(transformed_df_2, iter)

        dfs.append(transformed_df_1)
        dfs.append(transformed_df_2)

        iter += 1

    return [data_processing(df=df) for df in dfs]


def constrained_shuffle_generating(no_iteration: int, df_1: pd.DataFrame, df_2: pd.DataFrame) -> List[pd.DataFrame]:
    # see Stehlé et al. BMC Medicine 2011 on how to extrapolate data

    f_emp = compute_avg_f(df_1=df_1, df_2=df_2)
    # initialize b*(f-femp)^2
    squared_deviation = math.inf

    dfs = [df_1.copy(), df_2.copy()]
    unique_ids = get_unique_ids(df_1=df_1, df_2=df_2)

    iter = 1
    loop_count = 1

    while iter <= no_iteration:
        # Choose two tag Ids at random
        tag_i, tag_j = random.sample(list(unique_ids), k=2)

        transformed_df_1 = replace_i_with_j(df=df_1, tag_i=tag_i, tag_j=tag_j)
        transformed_df_2 = replace_i_with_j(df=df_2, tag_i=tag_i, tag_j=tag_j)

        # compute f
        f = compute_avg_f(df_1=transformed_df_1, df_2=transformed_df_2)

        new_squared_deviation = ((f-f_emp)**2)

        if new_squared_deviation <= squared_deviation:
            print("2-day data: {} is created at the loop: {} with squared_deviation: {} by replace {} with {}".format(
                iter, loop_count, new_squared_deviation, tag_i, tag_j))

            transformed_df_1 = update_time(transformed_df_1, iter)
            transformed_df_2 = update_time(transformed_df_2, iter)

            dfs.append(transformed_df_1)
            dfs.append(transformed_df_2)

            iter += 1
            squared_deviation = new_squared_deviation
        else:
            pass

        loop_count += 1

    return [data_processing(df=df) for df in dfs]


if __name__ == "__main__":
    no_iteration = 49

    input_file = "conference_attendance_data.csv"
    output_file = "synthetic_conference_attendance_data.pkl"

    # input_file = "test_data.csv"
    # output_file = "synthetic_conference_attendance_data.pkl"

    df = pd.read_csv(input_file)

    split_index = (df["time"] == 83400).idxmax()
    df_1 = df.iloc[:split_index]
    df_2 = df.iloc[split_index:]
    # make the time for day 2 starts with 86400 = 24*60*60
    df_2["time"] = df_2["time"] + 86400-83400

    print("repetitive_generating...")
    repetitive_generating_dfs = repetitive_generating(
        no_iteration=no_iteration, df_1=df_1, df_2=df_2)

    print("random_shuffle_generating...")
    random_shuffle_generating_dfs = random_shuffle_generating(
        no_iteration=no_iteration, df_1=df_1, df_2=df_2)

    # print("constrained_shuffle_generating...")
    # constrained_shuffle_dfs = constrained_shuffle_generating(
    #     no_iteration=no_iteration, df_1=df_1, df_2=df_2)

    with open(output_file, 'wb') as file:
        pickle.dump(
            {"repetitive_generating_dfs": repetitive_generating_dfs}, file)
        pickle.dump(
            {"random_shuffle_generating_dfs": random_shuffle_generating_dfs}, file)
    # pickle.dump(
    #     {"constrained_shuffle_dfs": constrained_shuffle_dfs}, file)
