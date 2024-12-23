import streamlit as st
from modules.theme_toggle import initialize_theme,toggle_theme, apply_styles

# Inicializar el estado del tema
initialize_theme()

# Configuración de la página
st.set_page_config(page_title="CategorifAI", page_icon=":bar_chart:", layout="wide")

# Aplicar los estilos según el modo seleccionado
apply_styles()

# URL del logo
logo_url = "https://http2.mlstatic.com/frontend-assets/ml-web-navigation/ui-navigation/6.6.73/mercadolibre/logo_large_25years@2x.png?width=300"

# Variables para la selección de secciones
if 'selected_main' not in st.session_state:
    st.session_state.selected_main = None
    
# Barra lateral con el logo y menú
with st.sidebar:
    st.image(logo_url, use_column_width=True)  # Muestra el logo desde la URL
    st.title("Menú Principal")
      
    # Botón para cambiar entre modo claro y oscuro con íconos de sol y luna
    theme_button = "🌙" if not st.session_state.dark_mode else "☀️"
    st.button(theme_button, on_click=toggle_theme)

    # Botones principales para secciones
    if st.button("Requerido"):
        st.session_state.selected_main = "Requerido"

    if st.button("Deseable"):
        st.session_state.selected_main = "Deseable"

    if st.button("Bonus"):
        st.session_state.selected_main = "Bonus"

# Menú horizontal a la derecha basado en la selección
st.title("Data Science Technical Challenge - CategorifAI")
st.subheader("Presentado por: Daniel Grass")
if st.session_state.selected_main:


    # Mostrar un menú horizontal según la selección del botón principal
    if st.session_state.selected_main == "Requerido":
        menu_options = st.radio(
            "Opciones de Requerido",
            options=["Data QA", "Reporting", "Machine Learning"],
            horizontal=True
        )

        # Mostrar el contenido correspondiente
        if menu_options == "Data QA":
            st.header("Data QA")
            st.write("Se debe chequear la calidad del dataset para hacer una evaluación de qué tan apropiados son los datos para tareas de Data Science.")
            st.write("Proponga un conjunto de correcciones en los datos de ser necesario.")

        elif menu_options == "Reporting":
            st.header("Reporting")
            st.write("Documente los resultados e insights obtenidos durante la exploración y describa conclusiones desde una perspectiva de negocio, soportado por gráficos, tablas, y métricas.")

        elif menu_options == "Machine Learning":
            st.header("Machine Learning")
            st.write("Describa las posibles tareas de Machine Learning que podrían realizarse desde el dataset dado, que podrían ser valiosas en el dominio dado (sólo explicar, no entrenar un modelo).")

    elif st.session_state.selected_main == "Deseable":
        menu_options = st.radio(
            "Opciones de Deseable",
            options=["Versionado de Código", "Feature Engineering", "Modelo Predictivo", "Mostrar Skills en Python", "Casos de Uso", "Métricas"],
            horizontal=True
        )

        # Mostrar el contenido correspondiente
        if menu_options == "Versionado de Código":
            st.header("Versionado de Código")
            st.write("Versionado de código con Git (incluso puede publicarse en tu cuenta personal de GitHub).")

        elif menu_options == "Feature Engineering":
            st.header("Feature Engineering")
            st.write("Indicar y calcular posibles candidatos de features que podrían utilizarse tanto columnas originales y transformaciones.")

        elif menu_options == "Modelo Predictivo":
            st.header("Modelo Predictivo")
            st.write("Realice un modelo predictivo.")

        elif menu_options == "Mostrar Skills en Python":
            st.header("Mostrar Skills en Python")
            st.write("Teniendo buenas prácticas en la estructura del código y la documentación.")

        elif menu_options == "Casos de Uso":
            st.header("Casos de Uso")
            st.write("Describir posibles casos de uso a tratar con este dataset que podrían agregar valor al negocio dado, indicando métodos, técnicas, y algoritmos por cada uno de ellos, así como justificando las decisiones tomadas.")

        elif menu_options == "Métricas":
            st.header("Métricas")
            st.write("Definir y calcular las métricas que considere más relevantes para la problemática propuesta.")

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