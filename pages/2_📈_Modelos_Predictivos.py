import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Configuración de la página
st.set_page_config(
    page_title="Modelos Predictivos Turismo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Modelo de Pronóstico de Turistas Extranjeros")
st.markdown("Este modelo permite realizar predicciones utilizando Regresión Lineal, Holt-Winters o ARIMA.")

# Cargar el dataset
dataset_path = "./turistas_extranjeros.csv"
df = pd.read_csv(dataset_path)

# Convertir la columna 'Fecha' a formato datetime
df['Fecha'] = pd.to_datetime(df['Fecha'], origin='1899-12-30', unit='D')

# Mostrar los datos cargados
# st.markdown("### Datos del Dataset")
# st.dataframe(df)


# Selección del modelo de predicción
st.sidebar.header("Configuración del Modelo")

tipo_modelo = st.sidebar.selectbox(
    "Selecciona el modelo de predicción",
    ["Regresión Lineal", "Holt-Winters", "ARIMA"],  # Opciones disponibles
    index=0  # Índice de la opción predeterminada (0 para "Regresión Lineal")
)

frecuencia = st.sidebar.selectbox("Selecciona la frecuencia de predicción", ["Años", "Meses", "Semanas"])
periodos_a_predecir = st.sidebar.slider(f"Selecciona el número de {frecuencia.lower()} a predecir", min_value=1, max_value=10, value=5)

# Calcular los períodos futuros
if frecuencia == "Años":
    pasos = periodos_a_predecir * 12  # 12 meses por año
elif frecuencia == "Meses":
    pasos = periodos_a_predecir
elif frecuencia == "Semanas":
    pasos = periodos_a_predecir * 4  # Aproximadamente 4 semanas por mes

# Preparar los datos
y = df['Turistas']  # Número de turistas como variable dependiente


#----
# Declaramos 3 columnas en una proporción de 33%, 33% y 33%
c1,c2,c3 = st.columns([33,33,33])
with c1:
    # Preparar los datos para Regresión Lineal
    X = np.array(df.index).reshape(-1, 1)  # Índices como variable independiente
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo de regresión lineal
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Generar predicciones para los períodos futuros
    X_futuro = np.arange(len(df), len(df) + pasos).reshape(-1, 1)
    y_futuro = modelo.predict(X_futuro)

    # Crear un DataFrame para las predicciones futuras
    df_predicciones = pd.DataFrame({
        "Periodo": [f"{frecuencia[:-1]} {i+1}" for i in range(periodos_a_predecir)],
        "Predicción de Turistas": y_futuro[:periodos_a_predecir].astype(int)
    })

    # Mostrar las predicciones futuras
    st.markdown("### Predicciones Futuras (Regresión Lineal)")
    #st.dataframe(df_predicciones)

    # Gráfico minimalista
    # st.markdown("### Gráfico de Predicciones (Regresión Lineal)")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index, y, label="Datos Reales", color="blue", marker="o", linestyle="-")
    ax.plot(X_futuro[:periodos_a_predecir], y_futuro[:periodos_a_predecir], label="Predicción", color="red", marker="o", linestyle="--")
    ax.set_xlabel("Período")
    ax.set_ylabel("Número de Turistas")
    ax.set_title(f"Predicción de Turistas para los Próximos {periodos_a_predecir} {frecuencia.lower()}")
    ax.legend()
    ax.grid(visible=False)  # Desactivar cuadrícula para un diseño más limpio
    st.pyplot(fig)
    
with c2:
        # Preparar el modelo Holt-Winters
    modelo = ExponentialSmoothing(
        y,
        seasonal='add',
        seasonal_periods=12
    ).fit()

    # Generar predicciones para los períodos futuros
    y_futuro = modelo.forecast(steps=pasos)

    # Crear un DataFrame para las predicciones futuras
    df_predicciones = pd.DataFrame({
        "Periodo": [f"{frecuencia[:-1]} {i+1}" for i in range(periodos_a_predecir)],
        "Predicción de Turistas": y_futuro[:periodos_a_predecir].astype(int)
    })

    # Mostrar las predicciones futuras
    st.markdown("### Predicciones Futuras (Holt-Winters)")
    #st.dataframe(df_predicciones)
    
    # Gráfico minimalista
    #st.markdown("### Gráfico de Predicciones (Holt-Winters)")
    fig, ax = plt.subplots(figsize=(8, 4))

    # Generar el eje x para los datos reales y las predicciones
    x_real = np.arange(len(df))  # Índices para los datos reales
    x_futuro = np.arange(len(df), len(df) + len(y_futuro))  # Índices para las predicciones futuras

    # Graficar los datos reales
    ax.plot(x_real, y, label="Datos Reales", color="blue", marker="o", linestyle="-")

    # Graficar las predicciones futuras
    ax.plot(x_futuro[:periodos_a_predecir], y_futuro[:periodos_a_predecir], label="Predicción", color="green", marker="o", linestyle="--")

    # Configurar etiquetas y título
    ax.set_xlabel("Período")
    ax.set_ylabel("Número de Turistas")
    ax.set_title(f"Predicción de Turistas para los Próximos {periodos_a_predecir} {frecuencia.lower()}")
    ax.legend()
    ax.grid(visible=False)  # Desactivar cuadrícula para un diseño más limpio

    # Mostrar el gráfico
    st.pyplot(fig)
with c3:
# Ajustar el modelo ARIMA
        modelo = ARIMA(y, order=(1, 1, 1)).fit()

        # Generar predicciones para los períodos futuros
        y_futuro = modelo.forecast(steps=pasos)

        # Crear un DataFrame para las predicciones futuras
        df_predicciones = pd.DataFrame({
            "Periodo": [f"{frecuencia[:-1]} {i+1}" for i in range(periodos_a_predecir)],
            "Predicción de Turistas": y_futuro[:periodos_a_predecir].astype(int)
        })

        # Mostrar las predicciones futuras
        st.markdown("### Predicciones Futuras (ARIMA)")
        #st.dataframe(df_predicciones)

        # Gráfico minimalista
        # st.markdown("### Gráfico de Predicciones (ARIMA)")
        fig, ax = plt.subplots(figsize=(8, 4))

        # Generar el eje x para los datos reales y las predicciones
        x_real = np.arange(len(df))  # Índices para los datos reales
        x_futuro = np.arange(len(df), len(df) + len(y_futuro))  # Índices para las predicciones futuras

        # Graficar los datos reales
        ax.plot(x_real, y, label="Datos Reales", color="blue", marker="o", linestyle="-")

        # Graficar las predicciones futuras
        ax.plot(x_futuro[:periodos_a_predecir], y_futuro[:periodos_a_predecir], label="Predicción", color="purple", marker="o", linestyle="--")

        # Configurar etiquetas y título
        ax.set_xlabel("Período")
        ax.set_ylabel("Número de Turistas")
        ax.set_title(f"Predicción de Turistas para los Próximos {periodos_a_predecir} {frecuencia.lower()}")
        ax.legend()
        ax.grid(visible=False)  # Desactivar cuadrícula para un diseño más limpio

        # Mostrar el gráfico
        st.pyplot(fig)
  

#----
c1,c2 = st.columns([70,30])

with c1:
    
    if tipo_modelo == "Regresión Lineal":

        # Preparar los datos para Regresión Lineal
        X = np.array(df.index).reshape(-1, 1)  # Índices como variable independiente
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear y entrenar el modelo de regresión lineal
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

        # Generar predicciones para los períodos futuros
        X_futuro = np.arange(len(df), len(df) + pasos).reshape(-1, 1)
        y_futuro = modelo.predict(X_futuro)

        # Crear un DataFrame para las predicciones futuras
        df_predicciones = pd.DataFrame({
            "Periodo": [f"{frecuencia[:-1]} {i+1}" for i in range(periodos_a_predecir)],
            "Predicción de Turistas": y_futuro[:periodos_a_predecir].astype(int)
        })

        # Mostrar las predicciones futuras
        #st.markdown("### Predicciones Futuras (Regresión Lineal)")
        #st.dataframe(df_predicciones)

        # Gráfico minimalista
        st.markdown("### Gráfico de Predicciones (Regresión Lineal)")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df.index, y, label="Datos Reales", color="blue", marker="o", linestyle="-")
        ax.plot(X_futuro[:periodos_a_predecir], y_futuro[:periodos_a_predecir], label="Predicción", color="red", marker="o", linestyle="--")
        ax.set_xlabel("Período")
        ax.set_ylabel("Número de Turistas")
        ax.set_title(f"Predicción de Turistas para los Próximos {periodos_a_predecir} {frecuencia.lower()}")
        ax.legend()
        ax.grid(visible=False)  # Desactivar cuadrícula para un diseño más limpio
        st.pyplot(fig)

    elif tipo_modelo == "Holt-Winters":
        
        # Preparar el modelo Holt-Winters
        modelo = ExponentialSmoothing(
            y,
            seasonal='add',
            seasonal_periods=12
        ).fit()

        # Generar predicciones para los períodos futuros
        y_futuro = modelo.forecast(steps=pasos)

        # Crear un DataFrame para las predicciones futuras
        df_predicciones = pd.DataFrame({
            "Periodo": [f"{frecuencia[:-1]} {i+1}" for i in range(periodos_a_predecir)],
            "Predicción de Turistas": y_futuro[:periodos_a_predecir].astype(int)
        })

        # Mostrar las predicciones futuras
        #st.markdown("### Predicciones Futuras (Holt-Winters)")
        #st.dataframe(df_predicciones)
        
        # Gráfico minimalista
        st.markdown("### Gráfico de Predicciones (Holt-Winters)")
        fig, ax = plt.subplots(figsize=(8, 4))

        # Generar el eje x para los datos reales y las predicciones
        x_real = np.arange(len(df))  # Índices para los datos reales
        x_futuro = np.arange(len(df), len(df) + len(y_futuro))  # Índices para las predicciones futuras

        # Graficar los datos reales
        ax.plot(x_real, y, label="Datos Reales", color="blue", marker="o", linestyle="-")

        # Graficar las predicciones futuras
        ax.plot(x_futuro[:periodos_a_predecir], y_futuro[:periodos_a_predecir], label="Predicción", color="green", marker="o", linestyle="--")

        # Configurar etiquetas y título
        ax.set_xlabel("Período")
        ax.set_ylabel("Número de Turistas")
        ax.set_title(f"Predicción de Turistas para los Próximos {periodos_a_predecir} {frecuencia.lower()}")
        ax.legend()
        ax.grid(visible=False)  # Desactivar cuadrícula para un diseño más limpio

        # Mostrar el gráfico
        st.pyplot(fig)
        
    elif tipo_modelo == "ARIMA":

            # Ajustar el modelo ARIMA
            modelo = ARIMA(y, order=(1, 1, 1)).fit()

            # Generar predicciones para los períodos futuros
            y_futuro = modelo.forecast(steps=pasos)

            # Crear un DataFrame para las predicciones futuras
            df_predicciones = pd.DataFrame({
                "Periodo": [f"{frecuencia[:-1]} {i+1}" for i in range(periodos_a_predecir)],
                "Predicción de Turistas": y_futuro[:periodos_a_predecir].astype(int)
            })

            # Mostrar las predicciones futuras
            #st.markdown("### Predicciones Futuras (ARIMA)")
            #st.dataframe(df_predicciones)

            # Gráfico minimalista
            st.markdown("### Gráfico de Predicciones (ARIMA)")
            fig, ax = plt.subplots(figsize=(8, 4))

            # Generar el eje x para los datos reales y las predicciones
            x_real = np.arange(len(df))  # Índices para los datos reales
            x_futuro = np.arange(len(df), len(df) + len(y_futuro))  # Índices para las predicciones futuras

            # Graficar los datos reales
            ax.plot(x_real, y, label="Datos Reales", color="blue", marker="o", linestyle="-")

            # Graficar las predicciones futuras
            ax.plot(x_futuro[:periodos_a_predecir], y_futuro[:periodos_a_predecir], label="Predicción", color="purple", marker="o", linestyle="--")

            # Configurar etiquetas y título
            ax.set_xlabel("Período")
            ax.set_ylabel("Número de Turistas")
            ax.set_title(f"Predicción de Turistas para los Próximos {periodos_a_predecir} {frecuencia.lower()}")
            ax.legend()
            ax.grid(visible=False)  # Desactivar cuadrícula para un diseño más limpio

            # Mostrar el gráfico
            st.pyplot(fig)
with c2:
        # Mostrar el DataFrame de Predicciones Futuras
        st.markdown("### DataFrame de Predicciones Futuras")
        st.dataframe(df_predicciones)
# Gráfico comparativo de los tres modelos
st.markdown("### Comparación de Modelos Predictivos")

# Generar predicciones para los tres modelos
# Regresión Lineal
modelo_rl = LinearRegression()
X = np.array(df.index).reshape(-1, 1)
modelo_rl.fit(X, y)
X_futuro = np.arange(len(df), len(df) + pasos).reshape(-1, 1)
y_futuro_rl = modelo_rl.predict(X_futuro)

# Holt-Winters
modelo_hw = ExponentialSmoothing(y, seasonal='add', seasonal_periods=12).fit()
y_futuro_hw = modelo_hw.forecast(steps=pasos)

# ARIMA
modelo_arima = ARIMA(y, order=(1, 1, 1)).fit()
y_futuro_arima = modelo_arima.forecast(steps=pasos)

# Crear el gráfico
fig, ax = plt.subplots(figsize=(10, 6))

# Datos reales
ax.plot(df.index, y, label="Datos Reales", color="blue", marker="o", linestyle="-")

# Predicciones de Regresión Lineal
ax.plot(np.arange(len(df), len(df) + pasos), y_futuro_rl, label="Regresión Lineal", color="red", marker="o", linestyle="--")

# Predicciones de Holt-Winters
ax.plot(np.arange(len(df), len(df) + pasos), y_futuro_hw, label="Holt-Winters", color="green", marker="o", linestyle="--")

# Predicciones de ARIMA
ax.plot(np.arange(len(df), len(df) + pasos), y_futuro_arima, label="ARIMA", color="purple", marker="o", linestyle="--")

# Configurar etiquetas y título
ax.set_xlabel("Período")
ax.set_ylabel("Número de Turistas")
ax.set_title(f"Comparación de Modelos Predictivos para los Próximos {periodos_a_predecir} {frecuencia.lower()}")
ax.legend()
ax.grid(visible=False)

# Mostrar el gráfico
#st.pyplot(fig)



# Crear el gráfico
fig, ax = plt.subplots(figsize=(10, 6))

# Generar fechas para los períodos futuros
fechas_futuras = pd.date_range(start=df['Fecha'].iloc[-1], periods=pasos + 1, freq='M')[1:]

# Datos reales
ax.plot(df['Fecha'], y, label="Datos Reales", color="blue", marker="o", linestyle="-")

# Predicciones de Regresión Lineal
ax.plot(fechas_futuras, y_futuro_rl, label="Regresión Lineal", color="red", marker="o", linestyle="--")

# Predicciones de Holt-Winters
ax.plot(fechas_futuras, y_futuro_hw, label="Holt-Winters", color="green", marker="o", linestyle="--")

# Predicciones de ARIMA
ax.plot(fechas_futuras, y_futuro_arima, label="ARIMA", color="purple", marker="o", linestyle="--")

# Configurar etiquetas y título
ax.set_xlabel("Fecha")
ax.set_ylabel("Número de Turistas")
ax.set_title(f"Comparación de Modelos Predictivos para los Próximos {periodos_a_predecir} {frecuencia.lower()}")
ax.legend()
ax.grid(visible=False)

# Rotar las etiquetas del eje x para mayor claridad
plt.xticks(rotation=45)

# Mostrar el gráfico
st.pyplot(fig)
