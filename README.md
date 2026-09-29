# quant-management

Librería de análisis de riesgo y gestión cuantitativa de carteras desarrollada para la asignatura **Quant Portfolio Management**.

El proyecto está diseñado como un monorepo extensible que servirá como base para las siguientes prácticas del curso: análisis de riesgo, factores, optimización de carteras y backtesting.

## Estructura del proyecto

```text
quant-management/
├── README.md
├── pyproject.toml
├── quantmgmt/
│   ├── data/
│   │   └── loader.py
│   ├── risk/
│   │   ├── analyzer.py
│   │   ├── returns.py
│   │   ├── volatility.py
│   │   ├── drawdown.py
│   │   ├── ratios.py
│   │   ├── tail.py
│   │   └── custom.py
│   ├── analytics/
│   ├── optimizers/
│   └── backtest/
├── notebooks/
│   └── 01_riesgo.ipynb
├── tests/
└── data/
    └── cache/
```

## Instalación

Se recomienda utilizar un entorno virtual de Python.

### Crear entorno virtual

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Instalar el proyecto

Desde la raíz del repositorio:

```bash
python -m pip install -e ".[dev]"
```

La instalación en modo editable permite modificar el código de `quantmgmt` sin tener que reinstalar el paquete después de cada cambio.

## Ejemplo de uso

### Descargar precios

```python
from quantmgmt.data import load_prices

prices = load_prices(
    tickers=["SPY", "EEM", "TLT", "GLD", "VNQ"],
    start="2010-01-01",
    end="2026-09-29",
)
```

Los datos descargados se almacenan en `data/cache/` para evitar descargas innecesarias y facilitar la reproducibilidad del análisis.

### Convertir precios a retornos

```python
from quantmgmt.risk import to_returns

returns = to_returns(
    prices,
    method="simple",
)
```

También se soportan retornos logarítmicos:

```python
log_returns = to_returns(
    prices,
    method="log",
)
```

## API funcional

Las métricas pueden utilizarse directamente como funciones:

```python
from quantmgmt.risk import (
    annualized_volatility,
    cagr,
    max_drawdown,
    sharpe_ratio,
    var_historical,
)

growth = cagr(returns)
volatility = annualized_volatility(returns)
drawdown = max_drawdown(returns)
sharpe = sharpe_ratio(returns)
var = var_historical(returns)
```

Las funciones aceptan tanto `pandas.Series` como `pandas.DataFrame`.

## RiskAnalyzer

La librería proporciona también una interfaz orientada a objetos que agrupa las métricas sin duplicar la lógica matemática.

```python
from quantmgmt.risk import RiskAnalyzer

analyzer = RiskAnalyzer(
    returns,
    periods_per_year=252,
)

print(analyzer.cagr())
print(analyzer.volatility())
print(analyzer.sharpe())
print(analyzer.max_drawdown())
print(analyzer.var())
```

Es posible generar directamente una tabla resumen:

```python
summary = analyzer.summary(
    confidence_level=0.95,
)

print(summary)
```

## Métricas implementadas

### Rentabilidad

- `to_returns`
- `cumulative_returns`
- `annualized_return`
- `cagr`

### Dispersión

- `annualized_volatility`
- `downside_deviation`
- `rolling_volatility`

### Drawdown

- `drawdown_series`
- `max_drawdown`
- `time_under_water`
- `recovery_time`

### Ratios

- `sharpe_ratio`
- `sortino_ratio`
- `calmar_ratio`
- `tracking_error`

### Riesgo de cola

- `var_historical`
- `var_parametric`
  - Gaussiano
  - Cornish-Fisher
- `cvar_historical`
- `cvar_parametric`
- `skewness`
- `kurtosis`

## Capital Resilience Ratio

Como métrica propia se propone el **Capital Resilience Ratio (CRR)**:

```text
              CAGR
CRR = -------------------------------
      sqrt(|MDD| * CVaR)
      * (1 + TUWmax / periods_per_year)
```

La métrica combina cuatro dimensiones:

- crecimiento compuesto,
- profundidad del drawdown,
- severidad de la cola,
- persistencia de las pérdidas.

Un CRR mayor indica una mayor capacidad histórica para generar crecimiento compuesto en relación con el riesgo de pérdida y el tiempo necesario para recuperar máximos anteriores.

Su uso está pensado principalmente para realizar comparaciones relativas entre carteras.

Ejemplo:

```python
from quantmgmt.risk import capital_resilience_ratio

crr = capital_resilience_ratio(
    returns,
    confidence_level=0.95,
    periods_per_year=252,
)
```

## Tests

Los tests utilizan `pytest`.

Ejecutar la suite completa:

```bash
python -m pytest -v
```

Ejecutar únicamente un módulo:

```bash
python -m pytest tests/test_tail.py -v
```

Los tests comprueban tanto resultados numéricos como propiedades financieras relevantes de las métricas.

## Notebook de análisis

El análisis aplicado se encuentra en:

```text
notebooks/01_riesgo.ipynb
```

Se estudian cinco ETFs representativos de distintas fuentes de riesgo:

- SPY
- EEM
- TLT
- GLD
- VNQ

El histórico utilizado comienza en 2010 y utiliza frecuencia diaria.

El notebook incluye:

- análisis de precios y retornos,
- tabla resumen de métricas,
- drawdowns históricos,
- volatilidad móvil,
- histogramas con VaR y CVaR,
- análisis de skewness y kurtosis,
- comparación entre Sharpe y Calmar,
- comparación del VaR entre distintos regímenes de mercado,
- análisis del Capital Resilience Ratio,
- conclusiones de inversión.

## Convenciones

Los retornos se expresan internamente en formato decimal:

```text
0.01  = +1%
-0.02 = -2%
```

Para datos diarios se utiliza por defecto:

```python
periods_per_year = 252
```

VaR y CVaR se expresan como **magnitudes de pérdida positivas**.

Por ejemplo:

```text
VaR 95% = 0.025
```

representa una pérdida del 2,5%.

La kurtosis utilizada es **excess kurtosis**, por lo que una distribución normal tiene aproximadamente:

```text
kurtosis = 0
```

## Reproducibilidad

Los precios descargados se almacenan localmente en:

```text
data/cache/
```

Esta carpeta está excluida del control de versiones mediante `.gitignore`.

Los datos pueden reconstruirse ejecutando de nuevo el notebook o utilizando `load_prices()`.

## Tecnologías

- Python
- NumPy
- pandas
- matplotlib
- yfinance
- pytest
- Jupyter