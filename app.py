"""Cartera óptima con acciones argentinas - Programación para Economistas (segundo parcial)."""
import numpy as np
import pandas as pd
import yfinance as yf
import scipy.optimize as sco
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Cartera óptima con acciones argentinas", layout="wide")

TICKERS = ['GGAL', 'YPF', 'PAM', 'BMA', 'GLD']
INICIO, FIN = '2022-01-01', '2026-11-16'   # fechas fijas para que los resultados no cambien
N = len(TICKERS)


@st.cache_data(ttl=3600, show_spinner="Bajando precios de Yahoo Finance...")
def bajar_precios():
    return yf.download(TICKERS, INICIO, FIN, progress=False)['Close'][TICKERS]


def armar_carteras(precios, ultimo_anio_entrenamiento):
    """Calcula las tres carteras con los datos de entrenamiento y las evalúa en el período de prueba."""
    retornos = np.log(precios / precios.shift()).dropna()
    entrenamiento = retornos.loc[:f'{ultimo_anio_entrenamiento}-12-31']
    prueba = retornos.loc[f'{ultimo_anio_entrenamiento + 1}-01-01':]
    mu = (entrenamiento.mean() * 252).values
    cov = (entrenamiento.cov() * 252).values

    retorno = lambda w: w @ mu
    volatilidad = lambda w: np.sqrt(w @ cov @ w)

    cotas = [(0, 1)] * N
    restriccion = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    w0 = np.ones(N) / N
    w_minvol = sco.minimize(volatilidad, w0, method='SLSQP', bounds=cotas, constraints=restriccion).x
    w_sharpe = sco.minimize(lambda w: -retorno(w) / volatilidad(w), w0, method='SLSQP',
                            bounds=cotas, constraints=restriccion).x
    pesos = pd.DataFrame({'Equiponderada': w0, 'Mínima volatilidad': w_minvol,
                          'Máx. retorno/riesgo': w_sharpe}, index=TICKERS)

    tabla_entren = pd.DataFrame({n: {'retorno': retorno(w.values), 'volatilidad': volatilidad(w.values)}
                                 for n, w in pesos.items()}).T
    cartera_prueba = pd.DataFrame({n: prueba @ w.values for n, w in pesos.items()})
    return dict(pesos=pesos, mu=mu, cov=cov, retorno=retorno, volatilidad=volatilidad,
                tabla_entren=tabla_entren, cartera_prueba=cartera_prueba,
                n_entren=len(entrenamiento), n_prueba=len(prueba))


def informe_plantilla(hechos, tabla_prueba):
    mejor = tabla_prueba['valor_final'].idxmax()
    peor = tabla_prueba['valor_final'].idxmin()
    return (f"INFORME AUTOMÁTICO (plantilla)\n\n{hechos}\n\n"
            f"En el período de prueba, la cartera con mejor resultado fue '{mejor}' y la peor fue '{peor}'. "
            "Los resultados dependen del período elegido y de la evolución del oro; el desempeño pasado no "
            "garantiza resultados futuros. Ejercicio académico, no es una recomendación de inversión.")


def redactar_con_ia(hechos, api_key):
    """Gemini redacta el informe usando SOLO los números dados. Devuelve None si algo falla."""
    if not api_key:
        return None
    try:
        import time
        from google import genai
        from google.genai import types
        try:   # máximo 20 segundos por intento y sin reintentos automáticos
            opciones = types.HttpOptions(timeout=20000, retry_options=types.HttpRetryOptions(attempts=1))
        except Exception:
            opciones = types.HttpOptions(timeout=20000)
        cliente = genai.Client(api_key=api_key, http_options=opciones)
        inicio = time.time()
        excluir = ('image', 'tts', 'live', 'audio', 'embedding', 'robotics', 'computer-use')
        modelos = sorted({m.name.replace('models/', '') for m in cliente.models.list()
                          if 'flash' in m.name and 'generateContent' in (getattr(m, 'supported_actions', None) or [])
                          and not any(x in m.name for x in excluir)}, reverse=True)
        modelos = [m for m in modelos if 'latest' in m] + [m for m in modelos if 'latest' not in m]   # primero los estables
        instruccion = ("Sos un analista económico. Con los datos que siguen, redactá en español un informe de 150 a 200 "
                       "palabras para un ahorrista que quiere repartir su capital entre acciones argentinas y oro. "
                       "Usá ÚNICAMENTE los números que te doy, sin inventar ninguno. Explicá qué cartera conviene y por "
                       "qué, mencioná si el ranking cambió entre entrenamiento y prueba, y cerrá con las limitaciones. "
                       "Aclará que es un ejercicio académico y no una recomendación de inversión.\n\nDATOS:\n" + hechos)
        for modelo in modelos[:5]:
            if time.time() - inicio > 45:   # no hacer esperar al usuario más de 45 segundos en total
                break
            try:
                return cliente.models.generate_content(model=modelo, contents=instruccion).text
            except Exception:
                continue
    except Exception:
        pass
    return None


# ------------------------------------------------------------------ interfaz
st.title("Cartera óptima con acciones argentinas")
st.caption("Programación para Economistas · Segundo parcial · UADE")

with st.sidebar:
    st.header("Parámetros")
    capital = st.number_input("Capital a invertir (US$)", min_value=1000, max_value=10_000_000, value=10_000, step=1000)
    anio = st.slider("Último año del período de entrenamiento", 2023, 2025, 2024,
                     help="El programa elige el reparto con los datos hasta fin de ese año y lo evalúa con los años siguientes.")
    st.markdown("**Activos:** GGAL, YPF, PAM y BMA (acciones argentinas que cotizan en Nueva York, en dólares) y GLD (oro).")

precios = bajar_precios()
if precios.isnull().any().any() or len(precios) < 300:
    st.error("No se pudieron bajar los precios completos. Probá recargar la página en unos minutos.")
    st.stop()

r = armar_carteras(precios, anio)
pesos, cartera_prueba = r['pesos'], r['cartera_prueba']
valor = capital * np.exp(cartera_prueba.cumsum())
tabla_prueba = pd.DataFrame({n: {'retorno': cartera_prueba[n].mean() * 252,
                                 'volatilidad': cartera_prueba[n].std() * np.sqrt(252),
                                 'valor_final': valor[n].iloc[-1]} for n in cartera_prueba}).T

st.write(f"**Pregunta:** si tengo US${capital:,.0f} para invertir en acciones argentinas y oro, ¿cómo conviene repartirlos "
         f"para tener el mejor equilibrio entre ganancia y riesgo? ¿Y ese reparto habría funcionado después?")
st.write(f"Entrenamiento: {precios.index[0].year} a {anio} ({r['n_entren']} días) · "
         f"Prueba: {anio + 1} a {precios.index[-1].date()} ({r['n_prueba']} días)")

t1, t2, t3, t4, t5 = st.tabs(["Reparto óptimo", "Carteras posibles", "Prueba", "Informe con IA", "Acerca del trabajo"])

with t1:
    st.subheader("¿Cuánto poner en cada activo?")
    st.write("Porcentaje del capital (arriba) y dólares (abajo).")
    st.dataframe((pesos * 100).round(1).astype(str) + " %", width='stretch')
    st.dataframe((pesos * capital).round(0).map(lambda x: f"US$ {x:,.0f}"), width='stretch')
    st.subheader("Resultado en el período de entrenamiento")
    te = r['tabla_entren'].copy()
    te['retorno / riesgo'] = te['retorno'] / te['volatilidad']
    st.dataframe(te.round(3), width='stretch')
    st.caption("Retorno y volatilidad anualizados. Retorno / riesgo: ratio de Sharpe con tasa libre de riesgo igual a 0.")

with t2:
    st.subheader("Todas las carteras posibles")
    rng = np.random.default_rng(42)
    W = rng.dirichlet(np.ones(N), 5000)
    ret_sim = W @ r['mu']
    vol_sim = np.array([r['volatilidad'](w) for w in W])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sc = ax.scatter(vol_sim, ret_sim, c=ret_sim / vol_sim, cmap='viridis', s=8, alpha=0.6)
    fig.colorbar(sc, label='retorno / riesgo')
    vol_act = np.sqrt(np.diag(r['cov']))
    ax.scatter(vol_act, r['mu'], s=80, color='black', label='activos individuales')
    for i, t in enumerate(TICKERS):
        ax.annotate(t, (vol_act[i], r['mu'][i]), textcoords='offset points', xytext=(6, 4))
    for nombre, marcador, color in [('Mínima volatilidad', '*', 'lime'), ('Máx. retorno/riesgo', '*', 'gold'),
                                    ('Equiponderada', 'D', 'red')]:
        w = pesos[nombre].values
        ax.scatter(r['volatilidad'](w), r['retorno'](w), marker=marcador, s=350 if marcador == '*' else 120,
                   color=color, edgecolor='black', label=nombre)
    ax.set_xlabel('Riesgo (volatilidad anual)')
    ax.set_ylabel('Retorno anual')
    ax.legend()
    st.pyplot(fig)
    st.caption("Cada punto es un reparto al azar del capital. Los activos solos quedan en el borde: combinarlos diversifica.")

with t3:
    st.subheader("¿Habría funcionado después?")
    st.write("Las ponderaciones se eligieron con el período de entrenamiento y **no se recalculan**: se evalúan con datos que el programa no vio.")
    st.line_chart(valor)
    mostrar = tabla_prueba.copy()
    mostrar['valor_final'] = mostrar['valor_final'].map(lambda x: f"US$ {x:,.0f}")
    st.dataframe(mostrar.rename(columns={'valor_final': f'US${capital:,.0f} se convirtieron en'}).round(3),
                 width='stretch')

lineas = []
for nombre in pesos.columns:
    p = ', '.join(f"{t} {pesos.loc[t, nombre] * 100:.1f}%" for t in TICKERS if pesos.loc[t, nombre] > 0.005)
    lineas.append(f"- Cartera {nombre}: ponderaciones {p}. Entrenamiento: retorno anual "
                  f"{r['tabla_entren'].loc[nombre, 'retorno'] * 100:.1f}%, volatilidad "
                  f"{r['tabla_entren'].loc[nombre, 'volatilidad'] * 100:.1f}%. Prueba: retorno anual "
                  f"{tabla_prueba.loc[nombre, 'retorno'] * 100:.1f}%, volatilidad "
                  f"{tabla_prueba.loc[nombre, 'volatilidad'] * 100:.1f}%, los US${capital:,.0f} se convirtieron en "
                  f"US${tabla_prueba.loc[nombre, 'valor_final']:,.0f}.")
hechos = '\n'.join(lineas)

with t4:
    st.subheader("Informe automático")
    st.write("Python calcula todos los números y una IA (Gemini, de Google) los redacta. La IA no calcula ni inventa cifras.")
    with st.expander("Ver los datos que recibe la IA"):
        st.text(hechos)
    if st.button("Generar informe"):
        try:
            clave = st.secrets.get("GEMINI_API_KEY", None)
        except Exception:
            clave = None
        with st.spinner("Redactando..."):
            texto = redactar_con_ia(hechos, clave)
        if texto is None:
            st.info("No se pudo usar la IA en este momento: se muestra el informe de plantilla.")
            texto = informe_plantilla(hechos, tabla_prueba)
        st.write(texto)

with t5:
    st.subheader("Acerca del trabajo")
    st.markdown("""
**Clasificación:** aplicación web de análisis y optimización con un asistente de IA generativa.

**Objetivo:** ayudar a decidir cómo repartir un capital entre acciones argentinas y oro, y comprobar si ese reparto habría funcionado con datos posteriores.

**Usuarios:** ahorristas que invierten en dólares, estudiantes de Economía.

**Tecnología:** Python, pandas, numpy, scipy (optimización de cartera de Markowitz), matplotlib, yfinance (precios de Yahoo Finance), Gemini (redacción del informe) y Streamlit (esta página).

**Limitaciones:**
- Los resultados dependen del período elegido y del oro, que subió mucho en la prueba.
- El método tiende a concentrar el dinero en pocos activos y es sensible a los datos.
- El rendimiento pasado no garantiza resultados futuros.
- Es un ejercicio académico, no una recomendación de inversión.
""")
