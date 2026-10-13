# Modelización de Riesgo de Crédito (PD, EAD, LGD)
### Estimación de Pérdida Esperada (IFRS 9 / Basilea) y Análisis de Cartera

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Licencia](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Descripción del Proyecto

Este proyecto aborda la modelización del riesgo de crédito minorista a través de los tres componentes fundamentales utilizados en la regulación bancaria (Basilea IRB) y en la normativa contable de provisiones (**IFRS 9 / NIIF 9**):

$$\text{ECL} = \text{PD} \times \text{EAD} \times \text{LGD}$$

* **PD (*Probability of Default*):** Probabilidad de que un prestatario incurra en impago.
* **EAD (*Exposure at Default*):** Exposición económica o saldo pendiente en el momento del impago.
* **LGD (*Loss Given Default*):** Porcentaje de la deuda que no se logra recuperar tras el impago.

A partir de una base histórica de **1,25 millones de préstamos personales** de LendingClub (originados entre 2016 y 2018), se desarrolla un pipeline cuantitativo completo que abarca desde la depuración y tratamiento de datos hasta la estimación de la pérdida esperada de balance y el análisis de la madurez de la cartera.

---

## 2. Objetivos Principales

1. **Modelización individual de componentes:** Estimar modelos independientes para PD, EAD y LGD priorizando la interpretabilidad econométrica y la estabilidad temporal.
2. **Estimación equilibrada de provisiones:** Cuantificar la pérdida esperada ($\text{ECL}$) buscando cubrir adecuadamente las pérdidas reales observadas, evitando tanto la infrovisión (riesgo de solvencia) como la sobreprovisión desmedida (coste de inmovilización de capital).
3. **Incorporación de Márgenes de Conservadurismo (MoC):** Evaluar cómo añadir márgenes prudenciales sobre las exposiciones de mayor riesgo para cumplir con los estándares supervisores.
4. **Análisis de la dinámica temporal y censura:** Diagnosticar cómo influye el grado de madurez de las cohortes (préstamos que aún no han completado su vida contractual) en las métricas de rendimiento y error.

---

## 3. Datos y Estrategia Temporal

El conjunto de datos contiene préstamos originados entre 2016 y 2018. Para respetar la naturaleza secuencial del negocio de crédito y evitar evaluar modelos con información futura o mezclas temporales, se estructura el análisis por cohortes anuales:

| Cohorte | Observaciones | Préstamos Vivos | Tasa de Mora Observada | Papel en el Proyecto |
| :--- | :---: | :---: | :---: | :--- |
| **2016** | 425.567 | ~36% | 16,69% | **Entrenamiento (Train):** Cartera madura que ha completado la ventana de 36 meses. |
| **2017** | 401.094 | ~63% | 10,41% | **Validación temporal (Val OOT):** Cartera intermedia con seguimiento de 12 a 24 meses. |
| **2018** | 426.181 | ~90% | 2,91% | **Evaluación y diagnóstico (Test):** Cartera reciente con censura temporal alta. |

### Criterios de Selección de Variables (Prevención de Fugas)
* Se eliminan todas las variables de seguimiento y recaudación posteriores a la concesión (`recoveries`, `last_pymnt_amnt`, etc.) para asegurar que el modelo solo use información disponible en el momento de la solicitud.
* Se excluyen variables como el rating interno preasignado (`grade`) y el tipo de interés (`interest_rate`), dado que reflejan decisiones comerciales y evaluaciones previas de la entidad, permitiendo al modelo aprender directamente de las características fundamentales del solicitante (ingresos, ratio de endeudamiento dti, historial de mora, líneas abiertas, etc.).
* Se realiza una selección de variables mediante contrastes econométricos por bloques (Test de Wald), depurando redundancias y multicolinealidad.

---

## 4. Metodología de Modelización

```
                          ESQUEMA DEL PIPELINE DE MODELIZACIÓN
                          
  Variables de Entrada       PD: Regresión Logística        EAD: GLM Fraccional s/ CCF
  (Solicitud y Buró)    ──>  (Parsimoniosa, 45 vars)   ──>  (R² ≈ 0.90)
                                     │                                  │
                                     └──────────────────┬───────────────┘
                                                        │
                                                        v
                                              LGD: Regresión Ridge
                                              (Severidad condicional)
                                                        │
                                                        v
                                          Cálculo de Provisión (ECL)
                                          + Margen Prudencial (MoC P88)
```

### 4.1 Probabilidad de Default (PD)
Se compararon modelos lineales regularizados (Lasso, Ridge, Elastic Net) y algoritmos basados en árboles (Random Forest, Gradient Boosting). 
* Se seleccionó una **Regresión Logística (Logit MLE)** depurada con 45 variables por su equilibrio entre capacidad discriminante ($\text{AUC} \approx 0,675$, alcanzando el $98\%$ del rendimiento de modelos de boosting más complejos) y su alta estabilidad temporal y facilidad de auditoría e interpretación.

### 4.2 Exposición en el Momento de Impago (EAD / CCF)
Dado que los préstamos van amortizando cuotas a lo largo del tiempo, no se pierde el importe inicial completo.
* Se modeliza el **Factor de Conversión de Crédito (CCF)**, que mide la fracción de capital insolvente al fallar, acotada entre $0$ y $1$.
* Se utiliza un **Modelo Lineal Generalizado Fraccional (GLM QMLE)**, alcanzando un $R^2 \approx 0,897$, lo que mejora notablemente respecto a utilizar un factor constante de exposición.

### 4.3 Severidad de la Pérdida (LGD)
* En préstamos personales sin colateral o garantía real, las recuperaciones tras el impago suelen ser bajas (la severidad media en la cartera se sitúa en torno al $90\% - 94\%$).
* Se utiliza una **Regresión Lineal Regularizada (Ridge)** sobre los créditos fallidos para estimar la severidad esperada.

### 4.4 Integración y Margen de Conservadurismo (MoC)
Bajo los estándares de supervisión bancaria, es habitual aplicar un Margen de Conservadurismo (*Margin of Conservatism*) para gestionar la incertidumbre de estimación:
* Se analiza una arquitectura en dos tramos: provisión base estándar sobre la mayor parte de la cartera, reforzada con un margen prudencial (percentil 88 de severidad) en los préstamos identificados en el tramo de mayor riesgo (*watchlist*).
* Este enfoque logra cubrir holgadamente la pérdida real en validación ($1,94\times$ de cobertura) sin generar el sobredimensionamiento excesivo que causaría recalibrar de forma artificial toda la base.

---

## 5. Principales Resultados y Diagnóstico de Cartera

### Validación Temporal (Cohorte 2017)
* La pérdida real registrada en validación fue de **490,8 M USD**.
* El modelo con margen prudencial focalizado dotó **950,3 M USD**, asegurando que menos del $10\%$ de los casos de impago queden subestimados en severidad, al tiempo que reduce en más de un $50\%$ la desviación en clientes sanos respecto a modelos sobreponderados de forma agresiva.

### Diagnóstico de la Cohorte 2018 (Censura Temporal por la Derecha)
Al evaluar el comportamiento del modelo sobre la cohorte de 2018 (Test), se observan particularidades cuantitativas derivadas del momento del ciclo de vida en que se encuentra la cartera:

1. **Alta proporción de préstamos abiertos (90,1%):** Al ser préstamos recientes, la inmensa mayoría sigue pagando sus cuotas con normalidad y no ha tenido tiempo de llegar al horizonte natural de impago (que suele manifestarse entre los meses 12 y 24).
2. **Tasa de mora aparente reducida (2,91%):** La tasa de default observada es baja no necesariamente por mayor calidad crediticia, sino por la falta de tiempo de maduración (efecto de censura por la derecha).
3. **Impagos tempranos y saldo insolvente:** Los pocos impagos que han ocurrido en 2018 son muy tempranos (mediana de ~5 cuotas pagadas), por lo que acumulan muy poca amortización y presentan una pérdida unitaria media superior a la de años anteriores.
4. **Interpretación de métricas de error (MAE):** En esta cohorte, el error medio se concentra principalmente en clientes sanos donde el modelo asigna una provisión prudencial por riesgo latente futuro, mientras que contablemente aún no se ha producido ningún quebranto.

---

## 6. Limitaciones del Estudio y Líneas Futuras

* **Tratamiento de la censura temporal:** Para incorporar carteras en curso en entornos de producción, resulta adecuado explorar modelos de análisis de supervivencia (como regresiones de Cox o modelos de tiempo acelerado de fallo AFT) o fijar ventanas de observación homogéneas a 12 meses (*12-month PD*).
* **Sesgo de selección (*Reject Inference*):** La base contiene únicamente operaciones aprobadas y formalizadas. Para utilizar el modelo como herramienta de admisión de nuevos solicitantes, convendría aplicar técnicas de inferencia de rechazados (como la corrección de Heckman).
* **Ciclo de recuperaciones (*Workout Period*):** En cohortes recientes, los procesos de recobro judicial y extrajudicial aún no han finalizado, por lo que las tasas de recuperación efectivas podrían incrementarse con el tiempo.

---

## 7. Estructura del Repositorio

```text
credit-risk/
│
├── notebooks/
│   └── credit-risk-notebook.ipynb    # Notebook completo con EDA, modelos de PD, EAD, LGD,
│                                      # cálculo de ECL y diagnóstico de censura.
│
├── data/                             # Datos (no incluidos en el repo por tamaño)
│   ├── X.csv                         # Matriz de variables explicativas
│   └── target.csv                    # Variable objetivo de default (y)
│
├── requirements.txt                  # Librerías necesarias para replicar el entorno
├── LICENSE                           # Licencia de uso
└── README.md                         # Descripción general del proyecto
```

### Cómo Ejecutar el Proyecto
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/alexalvarezds/credit-risk.git
   cd credit-risk
   ```
2. Instalar dependencias en un entorno virtual de Python:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Ejecutar el notebook secuencialmente desde Jupyter Lab o VSCode: `notebooks/credit-risk-notebook.ipynb`.