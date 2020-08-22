from sklearn.model_selection import train_test_split



def space_replacer(df):
    new_column_list = []
    for column_name in df.columns:
        new_column_name = column_name.replace(" ", "_")
        new_column_list.append(new_column_name)
    df.columns = new_column_list

def index_replacer(df):
    df.index += 1
    index_as_string = df.index.astype('str')
    df.index = "wine_" + index_as_string



def dataset_split(df):
    df = df.sample(frac=1)
    intermediate_df, valid_df = train_test_split(df, test_size=0.15)
    train_df, test_df = train_test_split(intermediate_df, test_size=0.15)
    train_y = train_df.pop('quality')
    valid_y = valid_df.pop('quality')
    test_y = test_df.pop('quality')
    dfs = train_df, valid_df, test_df
    targets = train_y, valid_y, test_y
    return dfs, targets


# for serving input function 

def max_min_finder(dataframe, parameters):
    my_dict = {}
    for columns in parameters['feature_names']:
        max_value = dataframe[columns].max()
        min_value = dataframe[columns].min()
        my_dict[columns] = [max_value,min_value]
    return my_dict

