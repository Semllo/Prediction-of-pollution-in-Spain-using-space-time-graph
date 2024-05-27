import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def apply_stl_to_group_adjusted(group, lat, lon, period=24, columns=["C6H6", "CO", "NO2", "NOx", "O3", "PM10", "PM25", "SO2"]):
    try:
        print(f"Comenzando procesamiento: Latitud {lat}, Longitud {lon}")
        for column in columns:
            print(f"Procesando columna {column} para latitud {lat} y longitud {lon}")
            n = len(group)
            train_end_idx = int(n * 0.7)
            valid_end_idx = int(n * 0.85)
            
            group[f'atrend_{column}'] = np.nan
            group[f'aseasonality_{column}'] = np.nan
            
            if train_end_idx > period:
                stl_train = STL(group[column][:train_end_idx], period=period, robust=True).fit()
                group.iloc[:train_end_idx, group.columns.get_loc(f'atrend_{column}')] = stl_train.trend
                group.iloc[:train_end_idx, group.columns.get_loc(f'aseasonality_{column}')] = stl_train.seasonal

            if valid_end_idx > period:
                stl_valid = STL(group[column][:valid_end_idx], period=period, robust=True).fit()
                group.iloc[train_end_idx:valid_end_idx, group.columns.get_loc(f'atrend_{column}')] = stl_valid.trend[train_end_idx:valid_end_idx]
                group.iloc[train_end_idx:valid_end_idx, group.columns.get_loc(f'aseasonality_{column}')] = stl_valid.seasonal[train_end_idx:valid_end_idx]

            if n > period:
                stl_test = STL(group[column], period=period, robust=True).fit()
                group.iloc[valid_end_idx:, group.columns.get_loc(f'atrend_{column}')] = stl_test.trend[valid_end_idx:]
                group.iloc[valid_end_idx:, group.columns.get_loc(f'aseasonality_{column}')] = stl_test.seasonal[valid_end_idx:]

        print(f"Finalizando procesamiento: Latitud {lat}, Longitud {lon}")
        return lat, lon, group
    except Exception as e:
        print(f"Error procesando grupo en Latitud {lat}, Longitud {lon}: {e}")
        return lat, lon, group

def process_group(args):
    return apply_stl_to_group_adjusted(*args)

def main():
    ruta_base = 'Dades/'
    ruta_archivo = ruta_base + 'dataset_combinado_final.parquet'

    df = pd.read_parquet(ruta_archivo)
    df = df.dropna(subset=['LATITUD_G', 'LONGITUD_G'])
    df.sort_values(by=['FECHA', 'LATITUD_G', 'LONGITUD_G'], inplace=True)

    args_list = [(group, lat, lon) for (lat, lon), group in df.groupby(['LATITUD_G', 'LONGITUD_G'])]

    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_group, args) for args in args_list]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                lat, lon, group = future.result()
                indices = group.index
                for contaminante in ["C6H6", "CO", "NO2", "NOx", "O3", "PM10", "PM25", "SO2"]:
                    df.loc[indices, f'atrend_{contaminante}'] = group[f'atrend_{contaminante}']
                    df.loc[indices, f'aseasonality_{contaminante}'] = group[f'aseasonality_{contaminante}']
            except Exception as e:
                print(f"Error durante el procesamiento paralelo: {e}")

    ruta_salida_parquet = ruta_base + 'dataset_procesado.parquet'
    df.to_parquet(ruta_salida_parquet)
    print(f"DataFrame guardado en {ruta_salida_parquet}")

if __name__ == '__main__':
    main()
