
# Proyecto: AB Testing con Streamlit y FastAPI

## Descripción General
Este proyecto integra herramientas como **Streamlit**, **FastAPI**, **PySpark** y **NumPyro** para analizar datos relacionados con pruebas A/B y exponerlos a través de una API.

### Componentes Principales:
1. **ETL**:
   - Transformación de datos en tres niveles (Bronze, Silver, Gold).
   - Limpieza, enriquecimiento y agregación de métricas.
2. **Analytics**:
   - Análisis estadísticos y bayesianos para evaluar variantes de experimentos.
   - Visualizaciones detalladas usando **Plotly**.
3. **API**:
   - FastAPI para exponer los resultados de los experimentos.
   - Documentación interactiva disponible.
4. **Llamadas API**:
   - Ejecución de solicitudes a la API y visualización de respuestas.

---

## Documentación de hipótesis y asunciones

### Hipótesis del Negocio
- Los experimentos A/B cuentan con datos limpios y correctamente etiquetados.
- La tasa de conversión es una métrica clave para evaluar el rendimiento de las variantes.
- Las variantes con mayores compras tienen un mejor desempeño.

### Hipótesis Técnicas
- Los datos de entrada están estructurados en tablas Delta alojadas en S3.
- Fechas y experimentos en los datos no tienen inconsistencias.
- El análisis bayesiano supone probabilidades de conversión con una distribución Beta.

---

## Documentación para interpretar la resolución

### Niveles del Proyecto:
1. **Nivel 1 (ETL)**:
   - Transformación en niveles (Bronze, Silver, Gold) para limpiar y enriquecer datos.
2. **Nivel 2 (Analytics)**:
   - Cálculo de tasas de conversión y análisis estadísticos.
3. **Nivel 3 (API)**:
   - Exposición de resultados mediante FastAPI. Repo: https://github.com/DanielGrass/fastapi-abtest-meli
4. **Nivel 4 (Llamadas API)**:
   - Consulta de la API desde Streamlit.

### Errores Potenciales:
- Datos faltantes o mal formateados pueden afectar los análisis.
- Modelos estadísticos dependen de la calidad de los datos.

---

## Instrucciones para uso y arranque del proyecto

1. **Instalación de Dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuración del Proyecto**:
   - Configura credenciales de AWS para acceder a S3.
   - Ejecuta la aplicación Streamlit:
     ```bash
     streamlit run main.py
     ```
   - Inicia la API localmente:
     ```bash
     uvicorn main:app --reload
     ```

3. **Despliegue en Heroku**:
   - Crea un archivo `Procfile`:
     ```
     web: uvicorn main:app --host=0.0.0.0 --port=${PORT}
     ```
   - Sube el proyecto:
     ```bash
     git push heroku main
     ```

---

## Escalabilidad y tradeoffs futuros

1. **Limitaciones Actuales**:
   - Uso de pandas y consultas directas en tablas Delta no son ideales para grandes volúmenes.
   - Spark puede requerir optimización adicional.

2. **Estrategias de Escalabilidad**:
   - Bases de datos distribuidas como Redshift o Snowflake.
   - Reemplazar pandas con Dask o PySpark.
   - Escalar la API con AWS Lambda o contenedores.

3. **Tradeoffs**:
   - Escalar puede aumentar costos operativos.
   - Migrar a tecnologías distribuidas requiere más recursos.

---

## URL de la API y ejemplos de consulta

1. **URL Base**:
   ```
   https://abtest-fastapi-662c944e83d2.herokuapp.com
   ```

2. **Ejemplo 1: Resultado satisfactorio**:
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

3. **Ejemplo 2: Fecha sin resultados**:
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

## Autor
**Daniel Grass**
