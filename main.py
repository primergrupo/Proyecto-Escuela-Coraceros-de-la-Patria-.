import io
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ==================================================================
# Configuración general
# ==================================================================
st.set_page_config(
    page_title="Hábitos alimenticios en las infancias",
    page_icon="🍎",
    layout="wide",
)

st.title("🍎 Análisis de hábitos alimenticios en las infancias")
st.caption("Datos recopilados con EpiCollect5 · Visualización con Pandas + Plotly")

DATA_DIR = Path("data")
DEFAULT_FILE = DATA_DIR / "epicollect5_export.csv"


# ==================================================================
# Carga y limpieza del CSV de EpiCollect5
# ==================================================================
@st.cache_data(show_spinner="Cargando y limpiando datos...")
def load_epicollect_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Lee un CSV exportado de EpiCollect5 y devuelve un DataFrame limpio."""
    sample = file_bytes[:4096].decode("utf-8", errors="ignore")
    sep = ";" if sample.count(";") > sample.count(",") else ","

    df_raw = pd.read_csv(
        io.BytesIO(file_bytes),
        sep=sep,
        dtype=str,
        keep_default_na=True,
        low_memory=False,
    )

    tipos_conocidos = {
        "text", "integer", "decimal", "select_one", "select_multiple",
        "date", "time", "datetime", "geopoint", "photo", "audio",
        "video", "barcode", "note", "begin_group", "end_group",
    }

    tipo_row = df_raw.iloc[0].astype(str).str.lower()
    if tipo_row.isin(tipos_conocidos).mean() > 0.3:
        df = df_raw.iloc[2:].copy()
    elif tipo_row.isin(tipos_conocidos).mean() > 0.0:
        df = df_raw.iloc[1:].copy()
    else:
        df = df_raw.copy()

    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)

    def clean_col(c: str) -> str:
        c = str(c).strip()
        c = re.sub(r"^grp\d+[._]", "", c)
        c = c.replace(".", " · ").strip(" ·")
        return c if c else "col"

    df.columns = [clean_col(c) for c in df.columns]

    for col in df.columns:
        serie = df[col].astype(str).str.strip()
        serie = serie.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

        num = pd.to_numeric(serie, errors="coerce")
        if num.notna().mean() > 0.9 and serie.notna().mean() > 0.5:
            df[col] = num
            continue

        try:
            fecha = pd.to_datetime(serie, errors="coerce", utc=True)
            if fecha.notna().mean() > 0.9 and serie.notna().mean() > 0.5:
                df[col] = fecha
                continue
        except Exception:
            pass

        df[col] = serie

    return df


def get_uploaded_or_default():
    st.sidebar.header("📂 Datos de entrada")
    uploaded = st.sidebar.file_uploader(
        "Subí tu CSV de EpiCollect5",
        type=["csv"],
        help="Exportá el proyecto desde EpiCollect5 como CSV y subilo aquí.",
    )
    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name

    if DEFAULT_FILE.exists():
        st.sidebar.info(f"Usando archivo por defecto:\n`{DEFAULT_FILE.name}`")
        return DEFAULT_FILE.read_bytes(), DEFAULT_FILE.name

    st.sidebar.warning("No hay archivo por defecto. Subí un CSV para comenzar.")
    return None


# ==================================================================
# Interfaz principal
# ==================================================================
entrada = get_uploaded_or_default()
if entrada is None:
    st.info("👈 Subí un archivo CSV exportado de EpiCollect5 desde la barra lateral.")
    st.stop()

file_bytes, filename = entrada
df = load_epicollect_csv(file_bytes, filename)

st.success(
    f"Archivo cargado: **{filename}** · "
    f"{df.shape[0]} respuestas · {df.shape[1]} columnas"
)

with st.expander("🔍 Vista previa de los datos crudos (primeras 10 filas)"):
    st.dataframe(df.head(10), use_container_width=True)

# ------------------------------------------------------------------
# Clasificar columnas por tipo
# ------------------------------------------------------------------
col_num = df.select_dtypes(include="number").columns.tolist()
col_cat = [
    c for c in df.columns
    if c not in col_num
    and not pd.api.types.is_datetime64_any_dtype(df[c])
    and df[c].nunique(dropna=True) <= 30
]

# ------------------------------------------------------------------
# Filtros dinámicos
# ------------------------------------------------------------------
st.sidebar.header("🎛️ Filtros")
df_f = df.copy()

for col in col_cat[:5]:
    opciones = sorted(df_f[col].dropna().unique().tolist(), key=str)
    if len(opciones) < 2 or len(opciones) > 50:
        continue
    sel = st.sidebar.multiselect(col, opciones, default=opciones)
    if sel:
        df_f = df_f[df_f[col].isin(sel)]

for col in col_num[:3]:
    if df_f[col].dropna().empty:
        continue
    lo, hi = float(df_f[col].min()), float(df_f[col].max())
    if lo == hi:
        continue
    rango = st.sidebar.slider(col, lo, hi, (lo, hi))
    df_f = df_f[df_f[col].between(*rango)]

if df_f.empty:
    st.warning("⚠️ Los filtros no devolvieron resultados. Ajustá los criterios.")
    st.stop()

# ------------------------------------------------------------------
# Métricas resumen
# ------------------------------------------------------------------
st.subheader("📊 Resumen")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Respuestas filtradas", len(df_f))
m2.metric("Columnas analizadas", df_f.shape[1])
if col_num:
    m3.metric(f"Promedio · {col_num[0]}", f"{df_f[col_num[0]].mean():.2f}")
    m4.metric(f"Mediana · {col_num[0]}", f"{df_f[col_num[0]].median():.2f}")

# ------------------------------------------------------------------
# Pestañas
# ------------------------------------------------------------------
tab_uni, tab_bi, tab_corr, tab_tabla = st.tabs(
    ["📈 Univariado", "🔀 Bivariado", "🔗 Correlaciones", "📋 Tabla y descarga"]
)

with tab_uni:
    st.markdown("#### Distribución de una variable")
    todas = col_num + col_cat
    if not todas:
        st.info("No hay variables adecuadas para graficar.")
    else:
        var = st.selectbox("Elegí una variable", todas, key="uni_var")
        serie = df_f[var].dropna()

        if var in col_num:
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(
                    px.histogram(df_f, x=var, nbins=20, title=f"Histograma · {var}"),
                    use_container_width=True, theme="streamlit",
                )
            with col_b:
                st.plotly_chart(
                    px.box(df_f, y=var, points="all", title=f"Boxplot · {var}"),
                    use_container_width=True, theme="streamlit",
                )
        else:
            conteo = serie.value_counts().reset_index()
            conteo.columns = [var, "frecuencia"]
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(
                    px.bar(conteo, x=var, y="frecuencia", text="frecuencia",
                           title=f"Conteo · {var}"),
                    use_container_width=True, theme="streamlit",
                )
            with col_b:
                st.plotly_chart(
                    px.pie(conteo, names=var, values="frecuencia",
                           title=f"Proporción · {var}"),
                    use_container_width=True, theme="streamlit",
                )

with tab_bi:
    st.markdown("#### Relación entre dos variables")
    if len(df_f.columns) < 2:
        st.info("Se necesitan al menos 2 columnas.")
    else:
        c1, c2, c3 = st.columns(3)
        var_x = c1.selectbox("Variable X", df_f.columns, key="bi_x")
        var_y = c2.selectbox("Variable Y", df_f.columns, key="bi_y")
        color = c3.selectbox("Color (opcional)", [None] + col_cat, key="bi_color")

        try:
            if var_x in col_num and var_y in col_num:
                fig = px.scatter(df_f, x=var_x, y=var_y, color=color,
                                 trendline="ols", title=f"{var_y} vs {var_x}")
            elif var_x in col_num or var_y in col_num:
                num = var_x if var_x in col_num else var_y
                cat = var_y if num == var_x else var_x
                fig = px.box(df_f, x=cat, y=num, color=color,
                             title=f"{num} por {cat}")
            else:
                tabla = df_f.groupby([var_x, var_y]).size().reset_index(name="frecuencia")
                fig = px.bar(tabla, x=var_x, y="frecuencia", color=var_y,
                             barmode="group", title=f"{var_x} × {var_y}")
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        except Exception as e:
            st.error(f"No se pudo generar el gráfico: {e}")

with tab_corr:
    st.markdown("#### Matriz de correlación (variables numéricas)")
    if len(col_num) < 2:
        st.info("Se necesitan al menos 2 variables numéricas.")
    else:
        corr = df_f[col_num].corr(numeric_only=True)
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Correlación de Pearson",
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        if col_cat:
            st.markdown("#### Tabla de frecuencia por variable categórica")
            var_cat = st.selectbox("Variable categórica", col_cat, key="corr_cat")
            freq = (
                df_f[var_cat].value_counts(dropna=False)
                .rename_axis(var_cat).reset_index(name="frecuencia")
            )
            freq["%"] = (freq["frecuencia"] / freq["frecuencia"].sum() * 100).round(1)
            st.dataframe(freq, use_container_width=True)

with tab_tabla:
    st.markdown("#### Datos filtrados")
    st.dataframe(df_f, use_container_width=True)
    st.download_button(
        "📥 Descargar CSV filtrado",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name=f"{Path(filename).stem}_filtrado.csv",
        mime="text/csv",
    )
    st.caption(
        "El archivo original nunca se modifica. "
        "Todas las transformaciones ocurren en memoria."
    )
