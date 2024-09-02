import numpy as np
import pandas as pd
import os

import shtReader_py.shtRipper as shtRipper


def clear_space(line) -> None:
    """
    Function to clear tabs & spaces from lines of data
    :param line: line of data
    :return: line w/o needless spaces
    """
    len_l = len(line)
    line = line.replace("\t", " ")
    line = line.replace("  ", " ")
    while len_l > len(line):
        len_l = len(line)
        line = line.replace("  ", " ")
    return line


def read_dataFile(file_path: str, id_file="") -> pd.DataFrame:
    """
    Function to read exported files. It's clearing spaces & unnecessary formatting
    :param file_path: full path w/ filename
    :param id_file: id of reading file (5 digits)
    :return: pd.DataFrame
    """
    new_filename = f'fil_{id_file}.dat'

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # удаляем первую строку
    lines.pop(0)
    # удаляем последние 4 строк (с запасом, чтобы не было мусора)
    lines = lines[:-4]
    # удаляем пробелы в начале и конце каждой строки
    lines = [line.strip() + "\n" for line in lines]
    # чистим пробелы
    lines = list(map(clear_space, lines))

    with open(new_filename, 'w') as f:
        f.writelines(lines)

    # Загрузка всех столбцов из файла
    data = pd.read_table(new_filename, sep=" ", names=["t"] + ["ch{}".format(i) for i in range(1, 10)])

    os.remove(new_filename)

    return data.dropna(axis=1)


def load_sht(filepath='D:/Edu/Lab/D-alpha-instability-search/data/sht/', filename="sht44168") -> pd.DataFrame:
    if ".sht" in filename.lower():
        res = shtRipper.ripper.read(filepath + filename)
    else:
        res = shtRipper.ripper.read(filepath + filename + ".SHT")

    data = np.array([res[res.keys()[0]]["x"]] + [res[ch]["y"] for ch in res.keys()])
    df = pd.DataFrame(data.transpose(), columns=["t"] + res.keys())
    return df


def read_sht_data(filename: str, filepath: str, data_name="D-alfa  хорда R=50 cm") -> pd.DataFrame:
    if ".sht" in filename.lower():
        res = shtRipper.ripper.read(filepath + filename)
    else:
        res = shtRipper.ripper.read(filepath + filename + ".SHT")

    data = np.array([res[data_name]["x"], res[data_name]["y"]])
    dalpha_df = pd.DataFrame(data.transpose(), columns=["t", "ch1"])
    return dalpha_df


def sht_rewrite(filepath='D:/Edu/Lab/D-alpha-instability-search/data/sht/', filename="sht44168",
                result_path="../data/d-alpha/", result_format="txt") -> str:
    """
    Function to export D-alpha data from SHT files to txt/csv/dat/... files
    :param filepath: Full path to file
    :param filename: Filename w/o format
    :param result_path: Path to save the result file
    :param result_format: Format of the result file
    :return: ok/Exception
    """
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    try:
        if ".sht" in filename.lower():
            res = shtRipper.ripper.read(filepath + filename)
            filename = filename[:-4]
        else:
            res = shtRipper.ripper.read(filepath + filename + ".SHT")

        data = np.array([res["D-alfa  хорда R=50 cm"]["x"], res["D-alfa  хорда R=50 cm"]["y"]])
        sht_df = pd.DataFrame(data.transpose(), columns=["t", "D-alpha_h50"])
        sht_df.to_csv(result_path + filename + "." + result_format, index=False)
        return "ok"
    except Exception as e:
        return e


def save_toSHT(data_dict: dict, result_filename="default_data.SHT",
               result_path="D:/Edu/Lab/D-alpha-instability-search/data/sht/marked/") -> str:
    """

    :param data_dict: keys: df - data to pack w/ ch1, ch1_marked & ch1_ai_marked columns, meta - list of dicts (3 items w/ keys: comment, unit, yRes)
    :param result_filename:
    :param result_path:
    :return: ok/Exception
    """
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    try:
        df = data_dict["df"]
        meta_data = data_dict["meta"]

        to_pack = {
            "D-alpha, chord=50 cm": {
                'comment': meta_data[0]["comment"],
                'unit': meta_data[0]["unit"],
                'tMin': df.t.min(),  # minimum time
                'tMax': df.t.max(),  # maximum time
                'offset': 0.0,  # ADC zero level offset
                'yRes': meta_data[0]["yRes"],  # ADC resolution: 0.0001 Volt per adc bit
                'y': df.ch1.to_list()
            },
            "Mark": {
                'comment': meta_data[1]["comment"],
                'unit': meta_data[1]["unit"],
                'tMin': df.t.min(),  # minimum time
                'tMax': df.t.max(),  # maximum time
                'offset': 0.0,  # ADC zero level offset
                'yRes': meta_data[1]["yRes"],  # ADC resolution: 0.0001 Volt per adc bit
                'y': df.ch1_marked.to_list()
            },
            "AI prediction": {
                'comment': meta_data[2]["comment"],
                'unit': meta_data[2]["unit"],
                'tMin': df.t.min(),  # minimum time
                'tMax': df.t.max(),  # maximum time
                'offset': 0.0,  # ADC zero level offset
                'yRes': meta_data[2]["yRes"],  # ADC resolution: 0.0001 Volt per adc bit
                'y': df.ch1_ai_marked.to_list()
            },
        }

        packed = shtRipper.ripper.write(path="D:/Edu/Lab/D-alpha-instability-search/data/sht/marked/",
                                        filename=result_filename, data=to_pack)
        if len(packed) != 0:
            raise Exception(f'Packed error = "{packed}"')

        return "ok"
    except Exception as e:
        return str(e)


def export_toSHT(filepath='../data/d-alpha/df/', filename="sht44184.scv",
                 result_path="D:/Edu/Lab/D-alpha-instability-search/data/sht/marked/") -> str:
    """
    Function to export D-alpha data from SHT files to txt/csv/dat/... files
    :param filepath: Full path to file
    :param filename: Filename w/o format
    :param result_path: Path to save the result file
    :param result_format: Format of the result file
    :return: ok/Exception
    """
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    try:
        df = pd.read_csv(filepath + filename, sep=",")

        to_pack = {}

        for column_name in df.columns[1:]:
            to_pack[column_name] = {
                'comment': '',
                'unit': 'U(V)',
                'tMin': df.t.min(),  # minimum time
                'tMax': df.t.max(),  # maximum time
                'offset': 0.0,  # ADC zero level offset
                'yRes': 0.0001,  # ADC resolution: 0.0001 Volt per adc bit
                'y': df[column_name]

            }

        packed = shtRipper.ripper.write(path=result_path, filename=filename[:-4] + '.SHT', data=to_pack)

        if len(packed) != 0:
            raise Exception(f'Packed error = "{packed}"')

        return "ok"
    except Exception as e:
        return str(e)


def rewrite_sht_fromDir(dir_path, result_path="../data/d-alpha/", result_format="txt") -> None:
    """
    Function to export data from entire dir
    :param dir_path:
    :param result_path:
    :param result_format:
    :return:
    """
    print("|", end="")

    for name in os.listdir(dir_path):
        report = sht_rewrite(filepath=dir_path, filename=name, result_path=result_path, result_format=result_format)

        if report == "ok":
            print(".", end="")
        else:
            print("-", end="")

    print("|")
