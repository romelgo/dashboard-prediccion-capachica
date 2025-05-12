import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# configuracion importante 
st.set_page_config(
    page_title="Dashboard Interactivo - Turismo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# Cargar el archivo CSV
dataset_path = "visitantes_capachica_normalizado.csv"

df = pd.read_csv(dataset_path)
# Título del dashboard
# CTRL+I para iniciar copilot
st.title("Dashboard Interactivo - Turismo- CAPACHICA")
st.markdown("Este dashboard permite analizar los datos de visitantes extranjeros por asociación en Capachica, desde el año 2018 hasta el 2023. Los datos son filtrables por asociación, tipo de visitante y rango de años.")
#----------------------
df = pd.read_csv(dataset_path)
# Dividir en 5 columnas
col1, col2, col3, col4, col5 = st.columns(5)

# Total de visitantes extranjeros


# Total de visitantes extranjeros con estilo personalizado
total_extranjeros = int(df[df['TIPO'] == 'EXTRANJERO']['VISITANTES'].sum())
col1.markdown(f"""
    <div class="metric-container" style="
        background-color: #F45D01;
        border-radius: 5px;
        box-shadow: 2px 2px 5px #F54F49;
        padding: 10px;
        text-align: center;">
        <div class="metric-label">Total visitantes Extranjeros</div>
        <div class="metric-value">{total_extranjeros}</div>
    </div>
""", unsafe_allow_html=True)

# Total de visitantes nacionales

# Agregar estilo CSS personalizado
st.markdown("""
    <style>
    .metric-container {
        background-color: #E883F5;
        border-radius: 5px;
        box-shadow: 2px 2px 5px #C88EF5;
        padding: 10px;
        text-align: center;
    }
    .metric-container .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: white;
    }
    .metric-container .metric-label {
        font-size: 16px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Total de visitantes nacionales con estilo personalizado
total_nacionales = int(df[df['TIPO'] == 'NACIONAL']['VISITANTES'].sum())
col2.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Total visitantes Nacionales</div>
        <div class="metric-value">{total_nacionales}</div>
    </div>
""", unsafe_allow_html=True)
# Número de asociaciones

# Número de asociaciones con estilo personalizado
total_asociaciones = df['ASOCIACION'].nunique()
col3.markdown(f"""
    <div class="metric-container" style="
        background-color: #F5BB38;
        border-radius: 5px;
        box-shadow: 2px 2px 5px #F5AF53;
        padding: 10px;
        text-align: center;">
        <div class="metric-label">Total Asociaciones</div>
        <div class="metric-value">{total_asociaciones}</div>
    </div>
""", unsafe_allow_html=True)
# Año con más visitas
anio_mas_visitas = df.groupby('AÑO')['VISITANTES'].sum().idxmax()
visitas_mas = df.groupby('AÑO')['VISITANTES'].sum().max()
col4.metric("Año con Más Visitas", anio_mas_visitas, visitas_mas)

# Año con menos visitas
anio_menos_visitas = df.groupby('AÑO')['VISITANTES'].sum().idxmin()
visitas_menos = df.groupby('AÑO')['VISITANTES'].sum().min()
col5.metric("Año con Menos Visitas", anio_menos_visitas, visitas_menos, delta_color="inverse")

#-------------
# Declaramos 2 columnas en una proporción de 60% y 40%
c1,c2 = st.columns([60,40])
with c1:

    # Título
    st.markdown("#### Visitantes Extranjeros por Asociación - Capachica (2018-2023)")

    # Ruta del archivo
    dataset_path = "visitantes_capachica_normalizado.csv"

    # Leer el CSV
    df_largo = pd.read_csv(dataset_path)

    # Filtrar visitantes extranjeros
    extranjeros = df_largo[df_largo['TIPO'] == 'EXTRANJERO']

    # Convertir año a entero si es necesario
    extranjeros['AÑO'] = extranjeros['AÑO'].astype(int)

    # Filtro por rango de años para esta gráfica
    
    min_year = int(extranjeros['AÑO'].min())
    max_year = int(extranjeros['AÑO'].max())
    rango_anios_grafica = st.slider("Selecciona el rango de años para la gráfica", min_year, max_year, (min_year, max_year))

    # Filtrar datos según el rango de años seleccionado
    extranjeros_filtrados = extranjeros[(extranjeros['AÑO'] >= rango_anios_grafica[0]) & (extranjeros['AÑO'] <= rango_anios_grafica[1])]

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    for asociacion in extranjeros_filtrados['ASOCIACION'].unique():
        data = extranjeros_filtrados[extranjeros_filtrados['ASOCIACION'] == asociacion]
        ax.plot(data['AÑO'], data['VISITANTES'], marker='o', label=asociacion)

    ax.set_title(f"Visitantes Extranjeros por Asociación ({rango_anios_grafica[0]}–{rango_anios_grafica[1]})\nCAPACHICA")
    ax.set_xlabel("Año")
    ax.set_ylabel("Número de Visitantes")
    ax.grid(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Mostrar en Streamlit
    st.pyplot(fig)
    
    
with c2:
    # Gráfica minimalista: Visitantes Nacionales y Extranjeros
    st.markdown("#### Visitantes Nacionales y Extranjeros")

    # Agrupar datos por tipo de visitante
    visitantes_tipo = df.groupby('TIPO')['VISITANTES'].sum()

    # Crear gráfico circular
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        visitantes_tipo,
        labels=visitantes_tipo.index,
        autopct='%1.1f%%',
        colors=['#DE6DA4', '#FFA07A'],
        startangle=90,
        wedgeprops={'edgecolor': '#E0811B'}
    )
    #ax.set_title("Visitantes Nacionales vs Extranjeros")

    # Mostrar en Streamlit
    st.pyplot(fig)

# Declaramos 3 columnas en una proporción de 40%, 40% y 20%
c1,c2,c3 = st.columns([40,40,20])
with c1:
    # Filtrar datos de visitantes extranjeros por asociación
    st.sidebar.header("Filtro por Asociación")
    asociaciones = df['ASOCIACION'].unique()
    asociacion_seleccionada = st.sidebar.selectbox("Selecciona una Asociación", asociaciones)

    # Filtro para visitantes nacionales y extranjeros
    st.sidebar.header("Filtro por Tipo de Visitante")
    tipo_visitante = st.sidebar.radio("Selecciona el Tipo de Visitante", ['EXTRANJERO', 'NACIONAL'])

    # Filtrar datos según la asociación seleccionada y tipo de visitante
    datos_filtrados = df[(df['ASOCIACION'] == asociacion_seleccionada) & (df['TIPO'] == tipo_visitante)]

    # Mostrar datos filtrados
    st.markdown(f"#### Visitantes {tipo_visitante.capitalize()}s - {asociacion_seleccionada}")
    #st.dataframe(datos_filtrados)

    # Gráfica de barras de visitantes por año según el tipo seleccionado
    #st.subheader(f"Gráfica de Visitantes {tipo_visitante.capitalize()}s por Año")
    
    
    # Gráfica de líneas de visitantes por año según el tipo seleccionado
    grafica_visitantes = datos_filtrados.groupby('AÑO')['VISITANTES'].sum()
    fig, ax = plt.subplots()
    ax.plot(
        grafica_visitantes.index, 
        grafica_visitantes.values, 
        marker='o', 
        color='#27D8F4' if tipo_visitante == 'NACIONAL' else '#FF5733', 
        linestyle='-', 
        label=f"Visitantes {tipo_visitante.capitalize()}s"
    )
    ax.set_ylabel(f"Número de Visitantes {tipo_visitante.capitalize()}s")
    ax.set_xlabel("Año")
    #ax.set_title(f"Gráfica de Visitantes {tipo_visitante.capitalize()}s por Año")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    
with c2:
    # Filtro por rango de años
    st.sidebar.header("Filtro por Rango de Años")
    min_year = int(df['AÑO'].min())
    max_year = int(df['AÑO'].max())
    rango_anios = st.sidebar.slider("Selecciona el rango de años", min_year, max_year, (min_year, max_year))

    # Filtrar datos según el rango de años seleccionado
    datos_filtrados_anios = datos_filtrados[(datos_filtrados['AÑO'] >= rango_anios[0]) & (datos_filtrados['AÑO'] <= rango_anios[1])]

    # Mostrar datos filtrados por rango de años
    st.markdown(f"#### Visitantes {tipo_visitante.capitalize()}s - {asociacion_seleccionada} (Años {rango_anios[0]} - {rango_anios[1]})")
    # st.dataframe(datos_filtrados_anios)

    # Gráfica de barras de visitantes por año según el rango seleccionado
    # st.subheader(f"Gráfica de Visitantes {tipo_visitante.capitalize()}s por Año (Años {rango_anios[0]} - {rango_anios[1]})")
    grafica_visitantes_anios = datos_filtrados_anios.groupby('AÑO')['VISITANTES'].sum()
    fig, ax = plt.subplots()
    grafica_visitantes_anios.plot(kind='bar', ax=ax, color='#27D8F4' if tipo_visitante == 'NACIONAL' else '#FF5733')
    ax.set_ylabel(f"Número de Visitantes {tipo_visitante.capitalize()}s")
    ax.set_xlabel("Año")
    st.pyplot(fig)
with c3:
    
    # Contar visitantes por tipo para la asociación seleccionada
    visitantes_extranjeros = int(datos_filtrados[datos_filtrados['TIPO'] == 'EXTRANJERO']['VISITANTES'].sum())
    visitantes_nacionales = int(datos_filtrados[datos_filtrados['TIPO'] == 'NACIONAL']['VISITANTES'].sum())

    # Mostrar los totales en Streamlit
    st.markdown(f"##### Totales de Visitantes para: {asociacion_seleccionada}")
    st.markdown(f"- **Extranjeros:** {visitantes_extranjeros}")
    st.markdown(f"- **Nacionales:** {visitantes_nacionales}")



# Título
st.title("Visitantes Extranjeros por Asociación - Capachica (2018-2023)")

# Ruta del archivo
dataset_path = "visitantes_capachica_normalizado.csv"
# Leer el CSV
df_largo = pd.read_csv(dataset_path)

# Filtrar visitantes extranjeros
extranjeros = df_largo[df_largo['TIPO'] == 'EXTRANJERO']

# Convertir año a string si es necesario
extranjeros['AÑO'] = extranjeros['AÑO'].astype(str)

# Crear gráfico
fig, ax = plt.subplots(figsize=(12, 6))
for asociacion in extranjeros['ASOCIACION'].unique():
    data = extranjeros[extranjeros['ASOCIACION'] == asociacion]
    ax.plot(data['AÑO'], data['VISITANTES'], marker='o', label=asociacion)

ax.set_title("Visitantes Extranjeros por Asociación (2018–2023)\nCAPACHICA")
ax.set_xlabel("Año")
ax.set_ylabel("Número de Visitantes")
ax.grid(True)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(rotation=45)
plt.tight_layout()

# Mostrar en Streamlit
st.pyplot(fig)

