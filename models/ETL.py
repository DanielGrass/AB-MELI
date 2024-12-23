from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, regexp_extract, split, explode, col, when, lit, row_number, sum as F_sum, last, countDistinct, to_date, to_timestamp
from pyspark.sql import Window

spark = SparkSession.builder \
    .appName("DeltaTableTest") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0,org.apache.hadoop:hadoop-aws:3.3.4") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .config("spark.local.dir", "C:/tmp/spark") \
    .getOrCreate()

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

# Contar las filas duplicadas antes de eliminarlas
df_duplicates_before = df.groupBy("user_id", "timestamp", "event_name", "item_id", "site", "experiments").count().filter(col("count") > 1)
print(f"Filas duplicadas encontradas antes de eliminarlas: {df_duplicates_before.count()}")

# Mostrar las filas duplicadas
df_duplicates_before.show()

df_cleaned = df.dropDuplicates(["user_id", "timestamp", "event_name", "item_id", "site", "experiments"])

# Confirmar que los duplicados han sido eliminados
df_duplicates_after = df_cleaned.groupBy("user_id", "timestamp", "event_name", "item_id", "site", "experiments").count().filter(col("count") > 1)
print(f"Filas duplicadas restantes: {df_duplicates_after.count()}")
df_duplicates_after.show()

#4. Validar que experiments tenga el formato esperado
df_valid_experiments = df_cleaned.withColumn("is_valid_experiment", regexp_extract("experiments", r"(\w+=\w+)", 0) != "")
invalid_experiments = df_valid_experiments.filter(~col("is_valid_experiment"))
print(f"Filas con 'experiments' inválidos: {invalid_experiments.count()}")
invalid_experiments.show()

# Guardar como Delta Table
df_cleaned.orderBy("user_id", "timestamp").write.format("delta").mode("overwrite").save(delta_table_bronze_path)
print(f"Delta Table guardada en: {delta_table_bronze_path}")


########################################
###### 2. Crear tabla silver (transform data)
########################################

# Ruta donde se guardará la tabla Delta
delta_table_silver_path = f"s3a://{bucket_name}/delta-table-silver-v1/"

# Cargar la tabla Delta Bronze
df_delta = spark.read.format("delta").load(delta_table_bronze_path)

# # PARTE I: Marcar eventos relacionados con compras

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


# Asignar grupo 0 a eventos no relacionados con compras
df_delta = df_delta.withColumn(
    "flag_purchase",
    when(col("buy_group").isNull(), lit(False)).otherwise(lit(True))
)

df_delta = df_delta.orderBy("user_id", "timestamp")

# Mostrar resultados
df_delta.select("user_id", "timestamp", "event_name", "buy_group", "flag_purchase").filter(col("user_id") == 1683).show(n=1000,truncate=False) #Usuario con usa sola sesion y con compra
df_delta.select("user_id", "timestamp", "event_name", "buy_group", "flag_purchase").filter(col("user_id") == 847572).show(n=1000,truncate=False) #Usuario con varias compras
df_delta.select("user_id", "timestamp", "event_name", "buy_group", "flag_purchase").filter(col("user_id") == 1876).show(n=1000,truncate=False) #Usuario con varias sesiones pero sin compras

# PARTE II: Explosión de datos después de marcar
# Limpiar la columna 'experiments' para quitar las llaves '{' y '}'
df_cleaned = df_delta.withColumn("experiments", regexp_replace(col("experiments"), "[{}]", ""))

# Separar los experimentos en un array
df_array = df_cleaned.withColumn("experiments_array", split(col("experiments"), ", "))

# Explode para separar cada experimento en filas individuales
df_exploded = df_array.withColumn("experiment", explode(col("experiments_array")))

# Separar experiment_name y variant_id
df_final = df_exploded.withColumn("experiment_name", split(col("experiment"), "=")[0]) \
                      .withColumn("variant_id", split(col("experiment"), "=")[1])

# Seleccionar columnas finales ordenadas por usuario y timestamp
df_result = df_final.select("user_id", "timestamp", "event_name", "item_id", "site", "experiment_name", "variant_id", "flag_purchase") \
                    .orderBy("user_id", "timestamp")


# Guardar la tabla Silver
df_result.write.format("delta").mode("overwrite").save(delta_table_silver_path)
df_result.filter(col("user_id") == 1683).show(n=1000,truncate=False) #Usuario con usa sola sesion y con compra
df_result.filter(col("user_id") == 847572).show(n=1000,truncate=False) #Usuario con varias compras
df_result.filter(col("user_id") == 1876).show(n=1000,truncate=False) #Usuario con varias sesiones pero sin compras

print(f"Delta Table guardada en: {delta_table_silver_path}")


########################################
###### 3. Crear tabla Gold (resultados agregados)
########################################

delta_table_gold_aggregate_path = f"s3a://{bucket_name}/delta-table-gold-aggregate/"
delta_table_gold_tunnel_path = f"s3a://{bucket_name}/delta-table-gold-tunnel/"

# Cargar la tabla Delta Bronze
df_delta_silver = spark.read.format("delta").load(delta_table_silver_path)

# Extraer el día de la columna de timestamp
df_delta_silver = df_delta_silver.withColumn("day", to_date(col("timestamp")))

# Agrupar por día, experimento y variante, y calcular métricas
df_aggregated = df_delta_silver.groupBy("day", "experiment_name", "variant_id").agg(
    countDistinct("user_id").alias("users"),  # Número de usuarios distintos
    countDistinct(when(col("flag_purchase") == True, col("user_id"))).alias("purchases")  # Número de usuarios distintos con compra
).orderBy("day", "experiment_name")


# Mostrar resultados
df_aggregated.orderBy("day", "experiment_name").show(n=1000, truncate=False)

# Guardar la tabla Gold - Agregado
df_aggregated.write.format("delta").mode("overwrite").save(delta_table_gold_aggregate_path)

# Agrupar por día, evento, experimento y variante, y calcular métricas
df_tunnel = df_delta_silver.groupBy("day", "event_name", "experiment_name", "variant_id").agg(
    countDistinct("user_id").alias("users"),  # Número de usuarios distintos
).orderBy("day", "experiment_name", "event_name")

# Mostrar resultados
df_tunnel.orderBy("day", "experiment_name", "event_name").show(n=1000, truncate=False)

# Guardar la tabla Gold - tunnel
df_tunnel.write.format("delta").mode("overwrite").save(delta_table_gold_tunnel_path)
