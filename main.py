from fastapi import FastAPI, HTTPException
from pyspark.sql import SparkSession
import boto3
import os
from mangum import Mangum

# Configuración de FastAPI
app = FastAPI()

# Configuración de S3
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_FILE_KEY = os.getenv("S3_FILE_KEY", "data.csv")

# Crear cliente de S3
s3_client = boto3.client("s3", region_name=AWS_REGION)

# Crear sesión de Spark
spark = SparkSession.builder.appName("ABTestAPI").getOrCreate()

# Leer y procesar el archivo CSV
def load_and_preprocess_data():
    try:
        # Leer archivo desde S3
        file_uri = f"s3a://{S3_BUCKET_NAME}/{S3_FILE_KEY}"
        df = spark.read.csv(file_uri, header=True, inferSchema=True)

        # Registrar DataFrame como tabla temporal
        df.createOrReplaceTempView("events")
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar los datos: {e}")


def parse_experiments(df):
    """
    Función para descomponer la columna `experiments` en experimentos individuales y variantes.
    """
    # Explode la columna `experiments`
    df_exploded = df.withColumn("experiment", spark.functions.explode(spark.functions.split(df["experiments"], ", ")))
    
    # Separar `experiment` en `experiment_name` y `variant`
    df_exploded = df_exploded.withColumn("experiment_name", spark.functions.split(df_exploded["experiment"], "=").getItem(0)) \
                             .withColumn("variant_id", spark.functions.split(df_exploded["experiment"], "=").getItem(1))
    
    return df_exploded

# Endpoint principal
@app.get("/")
async def root():
    return {"message": "API de AB Testing activa."}

 
# Endpoint para consultar por experimento y día/hora
@app.get("/experiment/{experiment_id}/result")
async def get_experiment_results(experiment_id: str, day: str):
    """
    Endpoint para consultar los resultados de un experimento en un día y hora específicos.
    """
    try:
        # Cargar datos y procesar
        df = load_and_preprocess_data()
        df_exploded = parse_experiments(df)

        # Filtrar por experimento y fecha/hora
        df_exploded.createOrReplaceTempView("exploded_events")
        query = f"""
        SELECT
            experiment_name,
            variant_id,
            COUNT(DISTINCT user_id) AS unique_users,
            COUNT(CASE WHEN event_name = 'BUY' THEN 1 ELSE NULL END) AS purchases,
            DATE_FORMAT(timestamp, 'yyyy-MM-dd HH') AS hour
        FROM exploded_events
        WHERE experiment_name LIKE '%{experiment_id}%'
          AND DATE_FORMAT(timestamp, 'yyyy-MM-dd HH') = '{day}'
        GROUP BY experiment_name, variant_id, DATE_FORMAT(timestamp, 'yyyy-MM-dd HH')
        """
        filtered_data = spark.sql(query).toPandas()

        if filtered_data.empty:
            raise HTTPException(status_code=404, detail="No se encontraron datos para este experimento en el día/hora especificados")

        # Construir respuesta
        response = {
            "results": {
                "exp_name": experiment_id,
                "day": day,
                "total_participants": int(filtered_data["unique_users"].sum()),
                "variants": [],
            }
        }

        # Añadir variantes
        for _, row in filtered_data.iterrows():
            response["results"]["variants"].append({
                "variant_id": row["variant_id"],
                "number_of_purchases": int(row["purchases"]),
                "unique_users": int(row["unique_users"]),
            })

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento: {e}")


# Mangum para conectar FastAPI con API Gateway y Lambda
handler = Mangum(app)
