import pandas as pd
from scipy.spatial import KDTree
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

def combinar_datos_fecha_hora(df_estaciones_filtrado, df_climaticos_filtrado):
    if df_estaciones_filtrado.empty or df_climaticos_filtrado.empty:
        return pd.DataFrame()
    
    kdtree_temp = KDTree(df_climaticos_filtrado[['longitude', 'latitude']])
    combined_data = []
    for _, row in df_estaciones_filtrado.iterrows():
        distance, index_closest = kdtree_temp.query([row['LONGITUD_G'], row['LATITUD_G']])
        closest_match = df_climaticos_filtrado.iloc[index_closest].to_dict()
        
        combined_row = {**row.to_dict(), **closest_match}
        combined_data.append(combined_row)
    print('Estacion añadida.')
    return pd.DataFrame(combined_data)

def main(df_estaciones, df_climaticos):
    df_combinado_final = pd.DataFrame()
    
    
    print('Empieza.')
    grupos_estaciones = df_estaciones.groupby(df_estaciones['FECHA'].dt.floor('h'))
    segmentos_estaciones = [grupo for _, grupo in grupos_estaciones]
    print('final estaciones.')
    grupos_climaticos = df_climaticos.groupby(df_climaticos['time'].dt.floor('h'))
    segmentos_climaticos = [grupo for _, grupo in grupos_climaticos]
    all_results = [] 
    print('final clima.')
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(combinar_datos_fecha_hora, seg_est, seg_clim) for seg_est, seg_clim in zip(segmentos_estaciones, segmentos_climaticos)]
        
        for future in as_completed(futures):
            df_resultado = future.result()
            if not df_resultado.empty:
                all_results.append(df_resultado) 

    if all_results: 
        print('Antes de concatenar.')
        df_combinado_final = pd.concat(all_results, ignore_index=True)
        df_combinado_final.to_parquet('Dades/dataset_combinado_final.parquet', engine='pyarrow')
        print("Procesamiento completado y guardado en 'Dades/dataset_combinado_final.parquet'")
    else:
        print("No se procesaron datos.")

if __name__ == '__main__':
    # Suponiendo que df_estaciones y df_climaticos ya están cargados y listos para usar
    df_estaciones = pd.read_parquet('Dades/estaciones.parquet')
    df_climaticos = pd.read_parquet('Dades/ERA5spain.parquet')
    df_estaciones['FECHA'] = pd.to_datetime(df_estaciones['FECHA'])
    df_climaticos['time'] = pd.to_datetime(df_climaticos['time'])
    
    main(df_estaciones, df_climaticos)
