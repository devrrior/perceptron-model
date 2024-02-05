import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, delimiter=";")

    num_columnas = len(df.columns)

    columns = ["x{}".format(i) for i in range(1, num_columnas)]
    columns.append("y")

    df.columns = columns

    return df
