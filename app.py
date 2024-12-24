import streamlit as st
from pyspark.sql import SparkSession
import pandas as pd
import plotly.express as px

# Configuración de la sesión de Spark
spark = SparkSession.builder \
    .appName("DeltaTableTest") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0,org.apache.hadoop:hadoop-aws:3.3.4") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .getOrCreate()

# Configuración de la página
st.set_page_config(page_title="AB test - MELI", page_icon=":bar_chart:", layout="wide")

# URL del logo
logo_url = "https://http2.mlstatic.com/frontend-assets/ml-web-navigation/ui-navigation/6.6.73/mercadolibre/logo_large_25years@2x.png?width=300"

# Variables para la selección de secciones
if 'selected_main' not in st.session_state:
    st.session_state.selected_main = None

# Rutas a las tablas Delta en S3
bucket_name = "abtest-meli"
bronze_path = f"s3a://{bucket_name}/delta-table-bronze/"
silver_path = f"s3a://{bucket_name}/delta-table-silver-v1/"
gold_aggregate_path = f"s3a://{bucket_name}/delta-table-gold-aggregate/"
gold_tunnel_path = f"s3a://{bucket_name}/delta-table-gold-tunnel-v1/"

# Barra lateral con el logo y menú
with st.sidebar:
    st.image(logo_url, use_container_width=True)  # Muestra el logo desde la URL
    st.title("Menú Principal")
      

    # Botones principales para secciones
    if st.button("Nivel 1: ETL"):
        st.session_state.selected_main = "Nivel 1: ETL"

    if st.button("Nivel 2: Analytics"):
        st.session_state.selected_main = "Nivel 2: Analytics"

    if st.button("Bonus"):
        st.session_state.selected_main = "Bonus"

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
                3. **Nivel Gold**: Cálculo de métricas agregadas a nivel de experimento y variante.
            
                ## Validaciones Realizadas

                1. **Datos Nulos**:
                - Se verificó que no existan valores nulos en las columnas clave.

                2. **Timestamps Inválidos**:
                - Se validó que todas las marcas de tiempo sigan un formato válido.

                3. **Duplicados**:
                - Se eliminaron filas duplicadas en base a todas las columnas clave relevantes.

                4. **Formato de Experimentos**:
                - Se aseguró que los datos de experimentos sigan la estructura esperada (`key=value`).

                ## Posibles Siguientes Pasos:

                1. **Validaciones Adicionales**:
                - Análisis de outliers en las marcas de tiempo.
                - Validación de consistencia entre `event_name` y otros campos (e.g., `item_id`).

                2. **Análisis Exploratorio**:
                - Generación de gráficos para evaluar tendencias por experimento y variante.

                3. **Automatización**:
                - Creación de scripts de validación automática para cada etapa del pipeline.                """)
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
            bronze_df = spark.read.format("delta").load(bronze_path)
            pd_bronze_df = pd.DataFrame(bronze_df.limit(1000).collect(), columns=[field.name for field in bronze_df.schema.fields])
            st.write("**Transformaciones principales:** Limpieza inicial de datos, eliminación de duplicados y validaciones básicas.")
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
            silver_df = spark.read.format("delta").load(silver_path)
            pd_silver_df = pd.DataFrame(silver_df.collect(), columns=[field.name for field in silver_df.schema.fields])
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
                     
                ### Archivo ETL: models/ETL.py""")
            # Cargar y mostrar datos de la tabla Gold
            gold_df = spark.read.format("delta").load(gold_aggregate_path)
            pd_gold_df = pd.DataFrame(gold_df.limit(1000).collect(), columns=[field.name for field in gold_df.schema.fields])
            st.write("**Transformaciones principales:** Cálculo de usuarios únicos y compras realizadas.")
            st.dataframe(pd_gold_df)

            # Leer los datos tunnel
            df_tunnel = spark.read.format("delta").load(gold_tunnel_path)
            pandas_tunnel_df = pd.DataFrame(df_tunnel.collect(), columns=[field.name for field in df_tunnel.schema.fields])
            st.write("**Datos Tunnel:** Número de usuarios (únicos) que realizaron cada evento.")
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
            "Opciones de Deseable",
            options=["Correctitud datos", "Nivel de confianza", "Modelo Bayesiano - Binario"],
            horizontal=True
        )

        # Mostrar el contenido correspondiente
        if menu_options == "Correctitud datos":
            st.header("Versionado de Código")
            st.write("Versionado de código con Git (incluso puede publicarse en tu cuenta personal de GitHub).")

        elif menu_options == "Nivel de confianza":
            st.header("Feature Engineering")
            st.write("Indicar y calcular posibles candidatos de features que podrían utilizarse tanto columnas originales y transformaciones.")

        elif menu_options == "Modelo Bayesiano - Binario":
            st.header("Modelo Predictivo")
            st.write("Realice un modelo predictivo.")
        

    elif st.session_state.selected_main == "Bonus":
        menu_options = st.radio(
            "Opciones de Bonus",
            options=["Manejo de Environment de Desarrollo", "Identificar Nuevos Atributos"],
            horizontal=True
        )

        # Mostrar el contenido correspondiente
        if menu_options == "Manejo de Environment de Desarrollo":
            st.header("Manejo de Environment de Desarrollo")
            st.write("Manejo de environment de desarrollo mediante alguna tecnología (e.g. Docker, virtualenv, conda).")

        elif menu_options == "Identificar Nuevos Atributos":
            st.header("Identificar Nuevos Atributos")
            st.write("Identificar nuevos atributos / tablas que podrían ser relevantes o necesarias para un mejor análisis.")