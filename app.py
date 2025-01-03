import streamlit as st
import pandas as pd
import plotly.express as px
import scipy.stats as stats
import plotly.graph_objects as go
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
import arviz as az
import numpy as np
import requests
# from deltalake import DeltaTable
# import os
# from dotenv import load_dotenv

# # Cargar variables de entorno desde .env
# load_dotenv()

# # Configurar opciones de almacenamiento usando las variables de entorno
# storage_options = {
#     "key": os.getenv("AWS_ACCESS_KEY_ID"),
#     "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
#     "client_kwargs": {
#         "region_name": os.getenv("AWS_DEFAULT_REGION")
#     }
# }


# Configuración de la página
st.set_page_config(page_title="AB test - MELI", page_icon=":bar_chart:", layout="wide")

# URL del logo
logo_url = "https://http2.mlstatic.com/frontend-assets/ml-web-navigation/ui-navigation/6.6.73/mercadolibre/logo_large_25years@2x.png?width=300"

# Variables para la selección de secciones
if 'selected_main' not in st.session_state:
    st.session_state.selected_main = None

# # Rutas a las tablas Delta en S3
# bucket_name = "abtest-meli"
# bronze_path = f"s3://{bucket_name}/delta-table-bronze/"
# silver_path = f"s3://{bucket_name}/delta-table-silver-v1/"
# gold_aggregate_path = f"s3://{bucket_name}/delta-table-gold-aggregate/"
# gold_tunnel_path = f"s3://{bucket_name}/delta-table-gold-tunnel-v1/"


# Función para cargar datos con deltalake
@st.cache_data
def load_bronze_data():
    st.info("Cargando datos de Bronze...")
    df = pd.read_parquet("bronze")
    return df

@st.cache_data
def load_silver_data():
    st.info("Cargando datos de Silver...")
    df = pd.read_parquet("silver")
    return df

@st.cache_data
def load_gold_data():
    st.info("Cargando datos de Gold Aggregate...")
    df = pd.read_parquet("gold_aggregated")
    return df

@st.cache_data
def load_tunnel_data():
    st.info("Cargando datos de Gold Tunnel...")
    df = pd.read_parquet("gold_tunnel")
    return df


# Barra lateral con el logo y menú
with st.sidebar:
    st.image(logo_url, use_container_width=True)  # Muestra el logo desde la URL
    st.title("Menú Principal")
      

    # Botones principales para secciones
    if st.button("Nivel 1: ETL"):
        st.session_state.selected_main = "Nivel 1: ETL"

    if st.button("Nivel 2: Analytics"):
        st.session_state.selected_main = "Nivel 2: Analytics"

    if st.button("Nivel 3: API"):
        st.session_state.selected_main = "Nivel 3: API"

    if st.button("Nivel 4: Llamadas API"):
        st.session_state.selected_main = "Nivel 4: Llamadas API"

# Menú horizontal a la derecha basado en la selección
st.title("Data Science Technical Challenge - AB Test")
st.subheader("Presentado por: Daniel Grass")
if st.session_state.selected_main:
    if st.session_state.selected_main == "Nivel 1: ETL":
        menu_options = st.radio(
            "Opciones de Nivel 1",
            options=["Introducción", "Bronze", "Silver", "Gold"],
            horizontal=True
        )

        if menu_options == "Introducción":
            st.header("Nivel 1 - Transformación y Validación de Datos")
            st.markdown("""
                ## Introducción 
                Este código se centra en la limpieza, transformación y validación de un conjunto de datos relacionado con pruebas A/B. Se divide en tres niveles principales:

                1. **Nivel Bronze**: Limpieza inicial y validación de integridad.
                2. **Nivel Silver**: Enriquecimiento de los datos con marcas específicas como `flag_purchase`.
                3. **Nivel Gold**: Cálculo de métricas agregadas a nivel de experimento y variante.""")
            st.image("images/ETL.png")
            st.markdown("""
                ## Validaciones Realizadas

                1. **Datos Nulos**:
                - Se verificó que no existan valores nulos en las columnas clave.

                2. **Timestamps Inválidos**:
                - Se validó que todas las marcas de tiempo sigan un formato válido.

                3. **Duplicados**:
                - Se eliminaron filas duplicadas en base a todas las columnas clave relevantes.

                4. **Formato de Experimentos**:
                - Se aseguró que los datos de experimentos sigan la estructura esperada (`key=value`).
            """)
            st.image("images/ETLProceso1.png")
            st.image("images/ETLProceso2.png")
            st.markdown("""    
                ## Posibles Siguientes Pasos:
                1. **Validaciones Adicionales**:
                    - Validación de consistencia entre `event_name` y otros campos (e.g., `item_id`).

                2. **Análisis Exploratorio**:
                - Generación de gráficos para evaluar tendencias por experimento y variante.
                        
                ### El script usado para el ETL se encuentra en este proyecto en: models/ETL.py:
                ```python
                    from pyspark.sql import SparkSession
                    from pyspark.sql.functions import regexp_replace, regexp_extract, split, explode, col, when, lit, row_number, sum as F_sum, last, countDistinct, to_date, to_timestamp
                    from pyspark.sql import Window
                    import pandas as pd

                    spark = SparkSession.builder \
                        .appName("DeltaTableTest") \
                        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0,org.apache.hadoop:hadoop-aws:3.3.4") \
                        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
                        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
                        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
                        .config("spark.local.dir", "C:/tmp/spark") \
                        .getOrCreate()

                    # Activar Arrow
                    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

                    # Configuración de logs para evitar mensajes excesivos
                    spark.sparkContext.setLogLevel("ERROR")

                    ########################################
                    ###### 1. Crear tabla bronze (raw data)
                    ########################################

                    # Ruta al archivo CSV (raw data) en S3
                    bucket_name = "abtest-meli"
                    file_key = "data.csv"
                    s3_path = f"s3a://{bucket_name}/{file_key}"
                    # Ruta donde se guardará la tabla Delta
                    delta_table_bronze_path = f"s3a://{bucket_name}/delta-table-bronze/"

                    # Leer el archivo CSV desde S3
                    df = spark.read.csv(s3_path, header=True, inferSchema=True)

                    # 1. Detectar valores nulos o faltantes
                    columns_to_check = ["user_id", "event_name", "timestamp", "experiments"]
                    df_nulls = df.select(
                        *[F_sum(col(column).isNull().cast("int")).alias(column) for column in columns_to_check]
                    )

                    print("Valores nulos por columna:")
                    df_nulls.show()

                    # 2. Validar el formato de los datos
                    df_invalid_timestamps = df.filter(to_timestamp("timestamp").isNull())
                    print(f"Filas con timestamp inválido: {df_invalid_timestamps.count()}")
                    df_invalid_timestamps.show()

                    # 3. Verificar si hay filas duplicadas y eliminarlas
                    df_duplicates_before = df.groupBy("user_id", "timestamp", "event_name", "item_id", "site", "experiments").count().filter(col("count") > 1)
                    print(f"Filas duplicadas encontradas antes de eliminarlas: {df_duplicates_before.count()}")
                    df_duplicates_before.show()

                    df_cleaned = df.dropDuplicates(["user_id", "timestamp", "event_name", "item_id", "site", "experiments"])

                    # Confirmar que los duplicados han sido eliminados
                    df_duplicates_after = df_cleaned.groupBy("user_id", "timestamp", "event_name", "item_id", "site", "experiments").count().filter(col("count") > 1)
                    print(f"Filas duplicadas restantes: {df_duplicates_after.count()}")
                    df_duplicates_after.show()

                    # 4. Validar que experiments tenga el formato esperado
                    df_valid_experiments = df_cleaned.withColumn("is_valid_experiment", regexp_extract("experiments", r"(\w+=\w+)", 0) != "")
                    invalid_experiments = df_valid_experiments.filter(~col("is_valid_experiment"))
                    print(f"Filas con 'experiments' inválidos: {invalid_experiments.count()}")
                    invalid_experiments.show()

                    # Guardar como Delta Table
                    df_cleaned.orderBy("user_id", "timestamp").write.format("delta").mode("overwrite").save(delta_table_bronze_path)
                    print(f"Delta Table guardada en: {delta_table_bronze_path}")

                    df_cleaned.orderBy("user_id", "timestamp").write.format("overwrite").parquet("bronze")

                    ########################################
                    ###### 2. Crear tabla silver (transform data)
                    ########################################

                    # Ruta donde se guardará la tabla Delta
                    delta_table_silver_path = f"s3a://{bucket_name}/delta-table-silver-v1/"

                    # Cargar la tabla Delta Bronze
                    df_delta = spark.read.format("delta").load(delta_table_bronze_path)

                    # Marcar eventos de compra
                    df_delta = df_delta.withColumn(
                        "is_buy", 
                        when(col("event_name") == "BUY", lit(1)).otherwise(lit(0))
                    )

                    # Crear una ventana para propagar el grupo de compras hacia atrás (orden descendente)
                    user_window_desc = Window.partitionBy("user_id").orderBy(col("timestamp").desc())

                    # Propagar los grupos de compras hacia atrás
                    df_delta = df_delta.withColumn(
                        "buy_group",
                        last(when(col("is_buy") == 1, row_number().over(user_window_desc)), ignorenulls=True).over(user_window_desc)
                    )

                    df_delta = df_delta.withColumn(
                        "flag_purchase",
                        when(col("buy_group").isNull(), lit(False)).otherwise(lit(True))
                    )

                    df_delta = df_delta.orderBy("user_id", "timestamp")

                    # Limpiar la columna 'experiments' para quitar las llaves '{' y '}'
                    df_cleaned = df_delta.withColumn("experiments", regexp_replace(col("experiments"), "[{}]", ""))
                    df_array = df_cleaned.withColumn("experiments_array", split(col("experiments"), ", "))
                    df_exploded = df_array.withColumn("experiment", explode(col("experiments_array")))
                    df_final = df_exploded.withColumn("experiment_name", split(col("experiment"), "=")[0]) \
                                        .withColumn("variant_id", split(col("experiment"), "=")[1])
                    df_result = df_final.select("user_id", "timestamp", "event_name", "item_id", "site", "experiment_name", "variant_id", "flag_purchase") \
                                        .orderBy("user_id", "timestamp")

                    df_result.write.format("delta").mode("overwrite").save(delta_table_silver_path)

                    ########################################
                    ###### 3. Crear tabla Gold (resultados agregados)
                    ########################################

                    delta_table_gold_aggregate_path = f"s3a://{bucket_name}/delta-table-gold-aggregate/"
                    delta_table_gold_tunnel_path = f"s3a://{bucket_name}/delta-table-gold-tunnel-v1/"

                    df_delta_silver = spark.read.format("delta").load(delta_table_silver_path)

                    # Extraer el día de la columna de timestamp
                    df_delta_silver = df_delta_silver.withColumn("day", to_date(col("timestamp")))

                    # Agrupar por día, experimento y variante, y calcular métricas
                    df_aggregated = df_delta_silver.groupBy("day", "experiment_name", "variant_id").agg(
                        countDistinct("user_id").alias("users"),
                        countDistinct(when(col("flag_purchase") == True, col("user_id"))).alias("purchases")
                    ).orderBy("day", "experiment_name")

                    df_aggregated.write.format("delta").mode("overwrite").save(delta_table_gold_aggregate_path)

                    df_tunnel = df_delta_silver.groupBy("day", "event_name", "experiment_name", "variant_id").agg(
                        countDistinct("user_id").alias("users"),
                        countDistinct(when(col("flag_purchase") == True, col("user_id"))).alias("purchases")
                    ).orderBy("day", "experiment_name", "event_name")

                    df_tunnel.write.format("delta").mode("overwrite").save(delta_table_gold_tunnel_path)
 
                """)
            
                
        elif menu_options == "Bronze":
            st.header("Bronze - Limpieza Inicial")
            st.markdown("""
                ### Objetivo

                Garantizar que los datos cargados sean válidos y estén libres de errores o inconsistencias.

                ### Pasos Realizados

                1. **Carga de Datos**:
                - Los datos se cargan desde un archivo CSV alojado en S3.
                - Se especifican los tipos de datos y las columnas relevantes.

                2. **Validación de Datos**:
                - **Valores Nulos**: Identificación de columnas con valores nulos.
                - **Timestamps Inválidos**: Verificación de la validez de las marcas de tiempo.
                - **Duplicados**: Identificación y eliminación de filas duplicadas considerando todas las columnas clave.
                - **Formato de `experiments`**: Validación del formato de los experimentos para asegurar que sigan la estructura esperada.

                3. **Almacenamiento**:
                - Los datos limpios se guardan como una tabla Delta en la ruta `delta_table_bronze_path`.""")
            # Cargar y mostrar datos de la tabla Bronze
            
            st.write("**Transformaciones principales:** Limpieza inicial de datos, eliminación de duplicados y validaciones básicas.")
            pd_bronze_df = load_bronze_data()
            st.dataframe(pd_bronze_df)

        elif menu_options == "Silver":
            st.header("Silver - Enriquecimiento de Datos")
            st.markdown("""
                ### Objetivo

                Transformar los datos para enriquecerlos con información sobre compras y experimentos.

                ### Pasos Realizados

                1. **Marcas de Compra**:
                - Identificación de eventos de compra (`BUY`) y asignación de un grupo de compra para propagarlos hacia atrás en el tiempo.
                - Creación de la columna `flag_purchase` para indicar si un evento contribuyó a una compra.

                2. **Explosión de Datos**:
                - Limpieza de la columna `experiments` para eliminar caracteres innecesarios.
                - Separación de los experimentos en pares clave-valor (`experiment_name` y `variant_id`).

                3. **Almacenamiento**:
                - Los datos transformados se guardan como una tabla Delta en la ruta `delta_table_silver_path`.""")
            # Cargar y mostrar datos de la tabla Silver   
            pd_silver_df = load_silver_data()         
            st.markdown(" **Transformaciones principales:** Asignación de `flag_purchase` y separación de experimentos.")
            st.markdown(" **Usuario que tiene varias sesiones de compra y concreta algunas.** ")
            st.dataframe(pd_silver_df[pd_silver_df["user_id"]==687987])
            st.markdown(" **Usuario con usa sola sesion y con compra.** ")
            st.dataframe(pd_silver_df[pd_silver_df["user_id"]==1683])
            st.markdown(" **Usuario con varias compras.** ")
            st.dataframe(pd_silver_df[pd_silver_df["user_id"]==847572])
            st.markdown(" **Usuario con varias sesiones pero sin compras.** ")
            st.dataframe(pd_silver_df[pd_silver_df["user_id"]==1876])
            st.markdown(" **Todos las filas procesadas en silver.** ")
            st.dataframe(pd_silver_df)

        elif menu_options == "Gold":
            st.header("Gold - Métricas Agregadas")
            st.write("""
                ### Objetivo

                Generar métricas agregadas para evaluar el rendimiento de cada experimento y variante.

                ### Pasos Realizados

                1. **Agregación General**:
                - Agrupación por día, experimento y variante.
                - Cálculo de:
                    - Usuarios (únicos) que participaron.
                    - Compras realizadas (únicos usuarios con `flag_purchase=True`).

                2. **Túnel de Eventos**:
                - Agrupación por día, evento, experimento y variante.
                - Cálculo del número de usuarios (únicos) que realizaron cada evento.

                3. **Almacenamiento**:
                - Los datos agregados se guardan como tablas Delta en las rutas:
                    - `delta_table_gold_aggregate_path` (métricas agregadas).
                    - `delta_table_gold_tunnel_path` (túnel de eventos).
                     
                """)
            
            pd_gold_df = load_gold_data()         

            # Cargar y mostrar datos de la tabla Gold            
            st.write("**Transformaciones principales:** Cálculo de usuarios únicos y compras realizadas.")
            st.dataframe(pd_gold_df)

            # Leer los datos tunnel           
            st.write("**Datos Tunnel:** Número de usuarios (únicos) que realizaron cada evento.")
            pandas_tunnel_df = load_tunnel_data()
            st.dataframe(pandas_tunnel_df)
            # Filtrar experimentos únicos
            experiments = pandas_tunnel_df["experiment_name"].unique()
            selected_experiment = st.selectbox("Experimento", experiments)

            if selected_experiment:
                # Filtrar los datos por experimento seleccionado
                filtered_df = pandas_tunnel_df[pandas_tunnel_df["experiment_name"] == selected_experiment]

                # Iterar sobre cada variante dentro del experimento seleccionado
                for variant in filtered_df["variant_id"].unique():
                    # Filtrar los datos por variante
                    variant_data = filtered_df[filtered_df["variant_id"] == variant]

                    # Agrupar los datos por día y evento para segmentar el túnel
                    funnel_data = (
                        variant_data.groupby(["day", "event_name"])["users"]
                        .sum()
                        .reset_index()
                        .sort_values(by="users", ascending=True)
                    )

                    # Crear una gráfica tipo funnel segmentada por día
                    fig = px.funnel(
                        funnel_data,
                        x="users",
                        y="event_name",
                        color="day",  # Segmentar por día
                        title=f"Túnel del Experimento '{selected_experiment}' - Variante {variant}",
                        labels={"users": "Cantidad de Usuarios", "event_name": "Eventos"},
                        color_discrete_sequence=px.colors.qualitative.Prism,  # Colores diferentes por día
                    )

                    # Mostrar la gráfica en Streamlit
                    st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.selected_main == "Nivel 2: Analytics":
        menu_options = st.radio(
            "Opciones de Analytics",
            options=["Análisis de Confianza y Correctitud", "Análisis Estadístico de Pruebas A/B", "Modelo Bayesiano - Binario"],
            horizontal=True
        )
        pd_gold_df = load_gold_data() 

        variant_count = (
            pd_gold_df.groupby("experiment_name")["variant_id"]
            .nunique()
            .reset_index()
            .rename(columns={"variant_id": "variant_count"})
            .sort_values(by="variant_count", ascending=False)
        )
        pd_gold_df["conversion_rate"] = pd_gold_df["purchases"] / pd_gold_df["users"]

        # Filtrar experimentos con más de una variante
        multi_variant_experiments = pd_gold_df[pd_gold_df["experiment_name"].isin(
            variant_count[variant_count["variant_count"] > 1]["experiment_name"]
        )]
        # Mostrar el contenido correspondiente
        if menu_options == "Análisis de Confianza y Correctitud":
            st.title("Análisis de Confianza y Correctitud - AB Testing")

            # Sección 1: Conteo de variantes por experimento
            st.markdown("""
            ### Conteo de Variantes por Experimento

            En esta sección, se analiza cuántas variantes tiene cada experimento y se ordenan de mayor a menor. 
            Los experimentos con una sola variante no pueden ser considerados válidos para pruebas A/B y serán marcados.
            """)

            # Identificar experimentos con solo una variante
            single_variant_experiments = variant_count[variant_count["variant_count"] == 1]

            st.dataframe(variant_count, use_container_width=True)

            if not single_variant_experiments.empty:
                st.warning(f"Se encontraron {len(single_variant_experiments)} experimentos con solo una variante. Estos no son válidos para pruebas A/B.")
                st.dataframe(single_variant_experiments)

            
            # Sección 2: Tasa de Conversión
            st.markdown("""
            ### Tasa de Conversión

            Calculamos la tasa de conversión por variante dentro de cada experimento, definida como el número de compras dividido entre el número total de usuarios.
            Esto nos permitirá identificar variantes que tengan un mejor desempeño.
            """)

            
            st.dataframe(multi_variant_experiments.sort_values(by="conversion_rate", ascending=False), use_container_width=True)

            selected_experiment = st.selectbox(
                "Seleccione un experimento:",
                multi_variant_experiments["experiment_name"].unique()
            )

            if selected_experiment:
                exp_data = multi_variant_experiments[multi_variant_experiments["experiment_name"] == selected_experiment]
                variants = exp_data["variant_id"].unique()

                # Crear figura para todas las variantes del experimento seleccionado
                fig = go.Figure()

                for variant in variants:
                    variant_data = exp_data[exp_data["variant_id"] == variant]

                    # Agregar barras
                    fig.add_trace(
                        go.Bar(
                            x=variant_data["day"],
                            y=variant_data["conversion_rate"],
                            name=f"Variante {variant}",
                            text=variant_data["conversion_rate"].round(2),
                            textposition="auto"
                        )
                    )

                    # Agregar línea de tendencia
                    fig.add_trace(
                        go.Scatter(
                            x=variant_data["day"],
                            y=variant_data["conversion_rate"],
                            mode="lines+markers",
                            name=f"Tendencia Variante {variant}",
                            line=dict(dash="dot")
                        )
                    )

                # Ajustar diseño
                fig.update_layout(
                    title_text=f"Tasa de Conversión por Día - {selected_experiment}",
                    barmode="group",
                    xaxis_title="Día",
                    yaxis_title="Tasa de Conversión",
                    height=600,
                    legend_title="Variante"
                )

                # Mostrar gráfico
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("""           
            Este análisis proporciona una visualización detallada de las tasas de conversión por día para las variantes de un experimento seleccionado.
            Esto permite identificar tendencias y patrones de comportamiento a lo largo del tiempo.
            """)
        elif menu_options == "Análisis Estadístico de Pruebas A/B":
            st.title("Análisis Estadístico de Pruebas A/B")

            # Inicializar listas para almacenar resultados
            chi2_results = []

            # Explicación de Chi-cuadrado con Latex
            latex_chi2 = r'''
            ## Prueba de Chi-cuadrado

            ### Hipótesis:
            - **Hipótesis nula (H₀):** No hay diferencia significativa en las tasas de conversión entre las variantes para un día específico.
            - **Hipótesis alternativa (H₁):** Existe una diferencia significativa en las tasas de conversión entre las variantes para ese día.

            ### Fórmula del estadístico Chi-cuadrado:
            $$
            \chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
            $$
            Donde:
            - $O_{ij}$ es la frecuencia observada en la celda $i,j$.
            - $E_{ij}$ es la frecuencia esperada en la celda $i,j$:
            $$
            E_{ij} = \frac{(\text{Total fila } i) \times (\text{Total columna } j)}{\text{Total general}}
            $$

            ### Consideraciones
            La prueba de Chi-cuadrado requiere que todas las celdas de la tabla de frecuencias esperadas tengan valores mayores a cero. Si alguna celda tiene un valor igual a cero, no es posible realizar la prueba, ya que esto resultaría en una división por cero en el cálculo del estadístico.
            ### Ejemplo de Tabla de Contingencia
            Supongamos que tenemos un experimento con dos variantes:

            | Variante | Compras | No Compras | Total |
            |----------|---------|------------|-------|
            | A        | 30      | 70         | 100   |
            | B        | 20      | 80         | 100   |
            | Total    | 50      | 150        | 200   |

            En este caso, calculamos las frecuencias esperadas para cada celda utilizando la fórmula:
            $$
            E_{ij} = \frac{(\text{Total fila } i) \times (\text{Total columna } j)}{\text{Total general}}
            $$
            
            '''

            st.write(latex_chi2)

            # Iterar sobre cada experimento
            for experiment in multi_variant_experiments['experiment_name'].unique():
                exp_data = multi_variant_experiments[multi_variant_experiments['experiment_name'] == experiment]

                # Iterar sobre cada día
                for day in exp_data['day'].unique():
                    day_data = exp_data[exp_data['day'] == day]
                    variants = day_data['variant_id'].unique()

                    # Preparar datos para la prueba de Chi-cuadrado
                    contingency_table = []
                    for variant in variants:
                        variant_data = day_data[day_data['variant_id'] == variant]
                        conversions = variant_data['purchases'].sum()
                        non_conversions = variant_data['users'].sum() - conversions
                        contingency_table.append([conversions, non_conversions])

                    # Verificar que no haya celdas vacías en las frecuencias observadas
                    contingency_table = pd.DataFrame(contingency_table)
                    if (contingency_table.sum(axis=1) == 0).any() or (contingency_table.sum(axis=0) == 0).any():
                        st.warning(f"Datos insuficientes para realizar la prueba de Chi-cuadrado en el experimento {experiment} el día {day}.")
                        continue

                    # Realizar prueba de Chi-cuadrado
                    try:
                        chi2, p_chi2, _, _ = stats.chi2_contingency(contingency_table)
                        conclusion = "Se acepta H₀: No hay diferencias significativas." if p_chi2 >= 0.05 else "Se rechaza H₀: Hay diferencias significativas."
                        description = (
                            "Las variantes parecen comportarse de manera similar." if p_chi2 >= 0.05 else
                            "Hay evidencia de que al menos una variante tiene un comportamiento diferente."
                        )
                        chi2_results.append({
                            'day': day,
                            'experiment_name': experiment,
                            'chi2_statistic': chi2,
                            'p_value': p_chi2,
                            'conclusion': conclusion,
                            'description': description
                        })
                    except ValueError as e:
                        st.error(f"Error al calcular Chi-cuadrado para el experimento {experiment} en el día {day}: {e}")

            # Mostrar resultados de Chi-cuadrado
            chi2_df = pd.DataFrame(chi2_results).sort_values(by=['day', 'experiment_name'])
            st.markdown("""
            #### Resultados de la Prueba de Chi-cuadrado

            Los resultados muestran el valor del estadístico Chi-cuadrado, el valor p, la conclusión de la prueba y una descripción interpretativa para cada día y experimento. Si \( p < 0.05 \), rechazamos la hipótesis nula y concluimos que existen diferencias significativas entre las variantes.
            """)
            st.dataframe(chi2_df)


        elif menu_options == "Modelo Bayesiano - Binario":
            st.markdown("""
                        # Modelo Bayesiano - Análisis Binario
                        ## Descripción
                        El modelo bayesiano implementado tiene como objetivo analizar el desempeño de variantes en un experimento A/B utilizando distribuciones Beta y Binomiales para modelar probabilidades de éxito en base a datos observados. Este enfoque nos permite calcular intervalos de confianza, analizar significancia estadística entre grupos y obtener trazas posteriores para una comparación visual.

                        ---

                        ## Estructura del Modelo
                        El modelo sigue la siguiente estructura:

                        1. **Prior:**  
                        - `alpha` y `beta` siguen una distribución exponencial con parámetro \(1/3\).  
                        - **Interpretación:** Priorizamos incertidumbre inicial antes de observar datos.

                        2. **Likelihood:**  
                        - La probabilidad de éxito (\(thetas\)) se modela con una distribución Beta utilizando \(alpha\) y \(beta\).  
                        - Los datos observados (\(y\)) se modelan con una distribución Binomial basada en \(thetas\).

                        3. **Posterior:**  
                        - A partir de las distribuciones anteriores, obtenemos la distribución posterior que nos proporciona información sobre los valores más probables de \(thetas\) y sus intervalos de confianza.

                        **Diagrama del modelo bayesiano**  """)
            st.image("images/Bayesiano.png")
            st.markdown("""
                        ---

                        ## Pasos Principales

                        1. **Selección de Experimento**  
                        El usuario selecciona un experimento de interés desde una lista desplegable. El análisis se aplica únicamente al subconjunto de datos correspondiente al experimento seleccionado.

                        2. **Configuración del Modelo**  
                        - Observaciones (\(ob\\_var\)): Datos de compras realizadas.
                        - Número de usuarios (\(n\\_var\)): Cantidad de participantes en cada variante.
                        - Reglas de variantes (\(output\\_rule\)): Identificadores únicos para las variantes.

                        3. **Muestreo Bayesiano con NumPyro**  
                        Se utiliza un kernel NUTS (No-U-Turn Sampler) para realizar un muestreo eficiente y generar la distribución posterior.

                        ---

                        ## Resultados Posteriores
                        Una vez generado el modelo, los resultados incluyen:
                        1. **Intervalos de Confianza (HDI):**  
                        Utilizamos el 95% del intervalo de densidad más alta para determinar si las variantes tienen intersección en sus intervalos de confianza.
                        2. **Significancia Estadística:**  
                        Se evalúa si las variantes tienen diferencias significativas respecto al grupo de control.

                        ## Selecciona un experimento para realizar el muestreo bayesiano:
                        """)

            # Selección de experimento
            selected_experiment = st.selectbox("Experimentos:", multi_variant_experiments['experiment_name'].unique())

            # Función de muestreo bayesiano con NumPyro
            def binary_model(data):
                # Preparar datos específicos para el modelo
                ob_var = data['purchases'].values
                n_var = data['users'].values
                output_rule = data['variant_id'].astype('category').cat.codes.values
                n_rules = len(data['variant_id'].unique())

                # Modelo NumPyro
                def ab_test_model(ob_var, n_var, output_rule):
                    alpha = numpyro.sample('alpha', dist.Exponential(1 / 3))
                    beta = numpyro.sample('beta', dist.Exponential(1 / 3))

                    # Distribución Beta para probabilidades de éxito
                    thetas = numpyro.sample('binary', dist.Beta(alpha, beta), sample_shape=(n_rules,))

                    # Observaciones utilizando una distribución Binomial
                    with numpyro.plate('observations', size=len(ob_var)):
                        numpyro.sample('obs', dist.Binomial(n_var, probs=thetas[output_rule]), obs=ob_var)

                # Configuración de MCMC
                nuts_kernel = NUTS(ab_test_model)
                mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=5000)

                # Ejecución de MCMC
                mcmc.run(jax.random.PRNGKey(0), ob_var, n_var, output_rule)

                # Conversión de resultados a formato InferenceData
                inference_data = az.from_numpyro(mcmc)
                posterior_samples = inference_data.posterior

                # Preparar los resultados
                def parse_results(inference_data, posterior_samples, key, identifiers, rules):
                    # Obtiene el resumen estadístico de las muestras posteriores para la variable de interés.
                    results = az.summary(inference_data, var_names=[key], hdi_prob=0.95, round_to=10)

                    results["ab_test_rule_name"] = rules

                    for k, v in identifiers.items():
                        # Añade los valores de identificación al DataFrame de resultados.
                        results[k] = v

                    results["class"] = key  # Añade al DataFrame final el tipo de variable Binary

                    # Añadir las trazas
                    thetas_samples = posterior_samples[key].values
                    trcs = np.split(thetas_samples, len(rules), axis=2)
                    trcs = [x.flatten() for x in trcs]
                    results["trace_lst"] = trcs

                    return results, inference_data

                identifiers = {"experiment_name": selected_experiment}
                rules = data['variant_id'].unique()
                results = parse_results(inference_data, posterior_samples, 'binary', identifiers, rules)

                return results

            if st.button("Realizar Muestreo Bayesiano"):
                experiment_data = pd_gold_df[pd_gold_df['experiment_name'] == selected_experiment]
                results, inference_data = binary_model(experiment_data)

                # Mostrar resultados resumidos
                st.markdown("### Resumen de los Resultados Posteriores")
                st.dataframe(results)

                 # Convertir trace_lst para uso en plot_posterior
                trace_data = {rule: results["trace_lst"][idx] for idx, rule in enumerate(results["ab_test_rule_name"])}

                # Crear un dataset compatible con ArviZ
                trace_ds = az.from_dict(posterior=trace_data)

                # Mostrar histogramas juntos
                st.markdown("### Histogramas Posteriores para las Variantes")
                az.plot_posterior(trace_ds)
                st.pyplot()
                # Grupo de control: primera variante en la fila
                control_trace = results['trace_lst'][0]
                control_name = results['ab_test_rule_name'][0]

                # Comparar todas las variantes con el grupo de control
                comparisons = []
                for idx, trace in enumerate(results['trace_lst']):
                    if idx == 0:  # Saltar el grupo de control
                        continue
                    variant_name = results['ab_test_rule_name'][idx]

                    # Calcular los intervalos de confianza
                    control_mean = np.mean(control_trace)
                    control_hdi = az.hdi(control_trace, hdi_prob=0.95)

                    test_mean = np.mean(trace)
                    test_hdi = az.hdi(trace, hdi_prob=0.95)

                    # Verificar si hay intersección entre los intervalos de confianza
                    significant = not (
                        (control_hdi[1] >= test_hdi[0]) and (control_hdi[0] <= test_hdi[1])
                    )

                    comparisons.append({
                        "variant": variant_name,
                        "control_mean": control_mean,
                        "control_hdi": control_hdi,
                        "test_mean": test_mean,
                        "test_hdi": test_hdi,
                        "significant": significant
                    })

                # Crear DataFrame para la comparación
                comparison_df = pd.DataFrame(comparisons)

                # Mostrar resultados
                                   
                st.image("images/bayesiano1.png")
                st.image("images/bayesiano2.png")
                
                
                st.dataframe(comparison_df)

                # Graficar distribuciones y marcar los intervalos
                fig = go.Figure()

                # Graficar grupo de control
                fig.add_trace(go.Scatter(
                    x=control_trace,
                    y=[0] * len(control_trace),
                    mode='markers',
                    name=f"Control ({control_name})",
                    marker=dict(color='red')
                ))

                for idx, row in comparison_df.iterrows():
                    variant_trace = results['trace_lst'][idx + 1]
                    variant_name = row["variant"]

                    # Agregar variante
                    fig.add_trace(go.Scatter(
                        x=variant_trace,
                        y=[0] * len(variant_trace),
                        mode='markers',
                        name=f"Variante {variant_name}",
                        marker=dict(color='blue')
                    ))

                    # Agregar intervalo de confianza
                    fig.add_shape(type="line",
                                x0=row["test_hdi"][0], y0=0, x1=row["test_hdi"][1], y1=0,
                                line=dict(color="blue", width=3))

                    fig.add_shape(type="line",
                                x0=row["control_hdi"][0], y0=0, x1=row["control_hdi"][1], y1=0,
                                line=dict(color="red", width=3))

                fig.update_layout(
                    title="Comparación de Intervalos de Confianza",
                    xaxis_title="Valores Simulados",
                    yaxis_title="Densidad",
                    legend_title="Grupos",
                    template="plotly_white"
                )

                # Mostrar la gráfica
                st.plotly_chart(fig)

                # Mostrar conclusiones sobre significancia
                st.markdown("### Conclusiones de la Comparación")
                for idx, row in comparison_df.iterrows():
                    if row["significant"]:
                        st.success(f"La variante {row['variant']} tiene diferencias significativas respecto al control.")
                    else:
                        st.info(f"La variante {row['variant']} no tiene diferencias significativas respecto al control.")
                st.markdown("""

                                Se grafican las trazas de las variantes con sus intervalos de confianza:
                                - Color rojo: Grupo de control.
                                - Color azul: Variantes de prueba.
                                - Las líneas representan los intervalos de confianza para cada grupo.

                                ---

                                ## Conclusiones
                                1. Las variantes con intervalos de confianza que no se cruzan con el grupo de control muestran diferencias estadísticamente significativas.
                                2. La visualización de las distribuciones posteriores permite identificar claramente las diferencias entre variantes.

                            """)
    elif st.session_state.selected_main == "Nivel 3: API":
        st.title("Documentación de la API A/B Testing - FastAPI")
        st.markdown("""
        ## Descripción General
        Esta API permite obtener resultados de experimentos A/B. Contiene información sobre experimentos, incluyendo el número de usuarios, variantes y compras realizadas.

        La API está implementada con **FastAPI** y está desplegada en Heroku para acceso público.

        ## Nota Importante
        En el parámetro `experiment_name`, es necesario reemplazar los caracteres `\` por `|` antes de realizar la solicitud.

        ## Repositorio del Proyecto
        El código fuente de esta API está disponible en el siguiente repositorio de GitHub:
        [fastapi-abtest-meli](https://github.com/DanielGrass/fastapi-abtest-meli)

        ## Estructura de la API
        ### Endpoint: `/experiment/{experiment_name}/result`
        Este endpoint permite consultar los resultados de un experimento en una fecha específica.

        #### Parámetros:
        1. **experiment_name (str)**: Nombre del experimento.  
        2. **day (str)**: Fecha en formato `YYYY-MM-DD`. Este parámetro es obligatorio.

        #### Respuesta Exitosa (200):
        Si el experimento y la fecha existen en los datos, la API devuelve:
        - **exp_name**: Nombre del experimento.
        - **day**: Fecha del experimento.
        - **number_of_participants**: Número total de participantes en el experimento.
        - **winner**: Variante ganadora basada en el mayor número de compras.
        - **variants**: Lista de variantes con el número de compras realizadas.

        #### Respuesta de Error (404):
        Si el experimento o la fecha no existen, se devuelve un mensaje indicando que no se encontraron datos.

        ---

        ## Ejemplo de Uso
        ### URL Base
        La API está desplegada en Heroku en la siguiente URL:
        ```
        https://abtest-fastapi-662c944e83d2.herokuapp.com
        ```

        ### Ejemplo 1: Resultado satisfactorio
        **URL**:
        ```
        https://abtest-fastapi-662c944e83d2.herokuapp.com/experiment/qadb|sa-on-vip/result?day=2021-08-01
        ```

        **Respuesta**:
        ```json
        {
            "results": {
                "exp_name": "qadb|sa-on-vip",
                "day": "2021-08-01",
                "number_of_participants": 3500,
                "winner": 1,
                "variants": [
                    {
                        "id": 1,
                        "number_of_purchases": 1500
                    },
                    {
                        "id": 2,
                        "number_of_purchases": 1200
                    },
                    {
                        "id": 3,
                        "number_of_purchases": 800
                    }
                ]
            }
        }
        ```

        ### Ejemplo 2: Fecha sin resultados
        **URL**:
        ```
        https://abtest-fastapi-662c944e83d2.herokuapp.com/experiment/qadb|sa-on-vip/result?day=2024-08-01
        ```

        **Respuesta**:
        ```json
        {
            "detail": "No data found for experiment 'qadb|sa-on-vip' on day '2024-08-01'"
        }
        ```

        ---

        ## Cómo probar la API localmente
        Si deseas probar la API en tu entorno local:
        1. Asegúrate de tener Python y las dependencias instaladas (`fastapi`, `pandas`, `uvicorn`).
        2. Ejecuta el servidor:
        ```bash
        uvicorn main:app --reload
        ```
        3. Accede a la documentación interactiva en:
        ```
        http://127.0.0.1:8000/docs
        ```

        ---

        ## Despliegue en Heroku
        1. Usa `git` para versionar tu código.
        2. Crea un archivo `Procfile` con el siguiente contenido:
        ```
        web: uvicorn main:app --host=0.0.0.0 --port=${PORT}
        ```
        3. Haz deploy con:
        ```bash
        git push heroku main
        ```
        ## Documentación propia de la API (https://abtest-fastapi-662c944e83d2.herokuapp.com/docs)
        
                    1. Dar click en Try out.
                    2. Escribir el nombre del experimento (recuerda reemplazar \ por |).
                    3. Escribir el dia a consultar.
                    4. Dar click en Execute.
                    5. Revisar responses.            
        """)
        st.image("images/fastapiim1.png", caption="Imagen 1", use_column_width=True)
        st.image("images/fastapiim2.png", caption="Imagen 2", use_column_width=True)

    elif st.session_state.selected_main == "Nivel 4: Llamadas API":
        st.header("Nivel 4 - Llamadas API")

        # Cargar datos Gold
        pd_gold_df = load_gold_data()

        # Filtrar experimentos únicos
        experiments = pd_gold_df["experiment_name"].unique()
        selected_experiment = st.selectbox("Seleccione un experimento", experiments)

        if selected_experiment:
            # Reemplazar \ por |
            formatted_experiment = selected_experiment.replace("/", "|")

            # Filtrar las fechas disponibles para el experimento
            experiment_dates = pd_gold_df[pd_gold_df["experiment_name"] == selected_experiment]["day"].unique()
            selected_date = st.selectbox("Seleccione una fecha", experiment_dates)

            if selected_date:
                # URL base de la API
                base_url = "https://abtest-fastapi-662c944e83d2.herokuapp.com"
                endpoint = f"/experiment/{formatted_experiment}/result?day={selected_date}"
                full_url = base_url + endpoint

                st.write(f"Consultando la API: {full_url}")

                # Realizar la consulta a la API
                try:
                    response = requests.get(full_url)

                    if response.status_code == 200:
                        result = response.json()
                        st.success("Consulta exitosa:")
                        st.json(result)
                    elif response.status_code == 404:
                        st.warning("No se encontraron resultados para este experimento y fecha.")
                        st.json(response.json())
                    else:
                        st.error(f"Error inesperado: {response.status_code}")
                        st.json(response.json())

                except Exception as e:
                    st.error(f"Error al conectar con la API: {e}")