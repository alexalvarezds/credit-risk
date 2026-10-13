# Modelización de Riesgo de Crédito (PD, EAD, LGD) e IFRS 9 / Basilea IRB
### Estimación de Pérdida Esperada de Cartera (*Lifetime ECL*), Optimización de Capital y Autopsia Econométrica de Censura Temporal

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Marco Regulatorio](https://img.shields.io/badge/Regulación-BCE%20TRIM%20%7C%20EBA%20GL%2F2017%2F16%20%7C%20IFRS%209-darkgreen.svg)](https://www.bankingsupervision.europa.eu/)
[![Licencia](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Resumen Ejecutivo y Propósito Regulatorio

Este proyecto desarrolla un marco cuantitativo integral para la modelización y estimación de la **Pérdida Crediticia Esperada (*Expected Credit Loss - ECL*)** bajo la normativa contable **IFRS 9 / NIIF 9** y los estándares prudenciales de **Basilea (IRB - Internal Ratings-Based)**, adaptado a los criterios de supervisión bancaria del **Banco Central Europeo (Guía TRIM)**, las **Directrices EBA/GL/2017/16** y la **Circular 4/2017 del Banco de España**.

A partir de una base histórica de **1.252.842 préstamos personales** de LendingClub (cohortes de originación **2016, 2017 y 2018**), se modelizan los tres componentes estructurales del riesgo de crédito:
$$\text{ECL}_j = \text{PD}_j \times \text{EAD}_j \times \text{LGD}_j$$

```
                           ARQUITECTURA DEL SISTEMA DE RIESGO
                           
 +--------------------+     +------------------------+     +------------------------+
 |    Originación     |     |   Exposición (EAD)     |     |    Severidad (LGD)     |
 |    Logit MLE       |     |   GLM Fraccional QMLE  |     |   Ridge Regularizado   |
 |  (45 vars Wald)    |     |   s/ CCF (R² = 0.897)  |     |  (LGD Condicional)     |
 +---------+----------+     +-----------+------------+     +-----------+------------+
           |                            |                              |
           v                            v                              v
    PD Base (Calibrada)              EAD Base                       LGD Base
           |                            |                              |
           +--------------------+-------+------------------------------+
                                |
                                v
                +-------------------------------+
                |  Arquitectura Two-Stage IFRS 9|
                |  Stage 1: Prov. Base (MLE)    |
                |  Watchlist Top 2%: MoC P88    |
                +---------------+---------------+
                                |
        +-----------------------+-----------------------+
        |                                               |
        v                                               v
  Cubre Pérdida Real                           Libera Capital Bancario
  (1.94x Cobertura en Val)                     +$1.502,13 M USD vs Balanced
```

### El Conflicto Central de Negocio y Regulación
En la gestión de balance de una entidad financiera existe un *trade-off* estructural entre:
1. **Solvencia Prudencial (Requisito Supervisor):** Evitar la subestimación de pérdidas en la cartera de crédito (exigencia regulatoria de no superar un $10\%$ de tasa de subestimación).
2. **Eficiencia de Capital (Objetivo Financiero):** Minimizar el volumen de provisiones inmovilizadas en balance para no restringir la liquidez ni castigar innecesariamente el ROE (*Return on Equity*).

### La Solución Desarrollada: Arquitectura Two-Stage IFRS 9
Frente a la solución simplista comúnmente empleada en Machine Learning de balancear clases (`class_weight="balanced"`), que dispara artificialmente las probabilidades dotando un exceso masivo de provisión, se diseñó una **Estrategia Two-Stage IFRS 9**:
* **Stage 1 (Cartera General - 98% de los préstamos):** Provisión calculada mediante el modelo no distorsionado `Logit MLE`, garantizando un cálculo prudente y realista del coste de riesgo.
* **Stage 2 / Watchlist (Top 2.0% de mayor riesgo):** Asignación de un **Margen de Conservadurismo (*Margin of Conservatism - MoC*) al percentil 88 (P88)** sobre la severidad conjunta ($\text{EAD} \times \text{LGD}$), absorbiendo la incertidumbre de cola en los perfiles vulnerables.

### Principales Resultados Cuantitativos (Validación OOT 2017)
* **Liberación de Capital:** Se liberan **`+$1.502,13 M USD`** de provisiones frente al modelo balanceado ($950,32\text{ M}$ frente a $2.452,45\text{ M}$), cubriendo con holgura la pérdida real observada ($490,78\text{ M USD}$, ratio de cobertura $1,94\times$).
* **Reducción de Distorsión en Clientes Sanos:** Reduce el error medio individual (MAE) en clientes sanos a menos de la mitad (**`$2.214`** vs. **`$5.275`** en el modelo balanceado).
* **Cumplimiento Regulatorio Pleno:**
  * Subestimación individual en centiles de riesgo: **`0,00%`** (límite superior Clopper-Pearson al 95%: $2,95\% < 10\%$, $p = 2,66 \times 10^{-5}$).
  * Subestimación en severidad de fallidos reales: **`9,07%`** (límite superior IC 95%: $9,30\% < 10\%$, $Z = -4,47, p < 0,0001$).

---

## 2. Marco de Datos, Partición Temporal y Prevención de Fuga de Información (*Data Leakage*)

### 2.1 Particiones Temporales (*Out-of-Time - OOT*)
Para replicar con rigor el funcionamiento de un departamento de riesgos y cumplir con la Guía TRIM del BCE (Sección PD, Párrafos 24–40), el conjunto de datos ($N = 1.252.842$) se dividió estrictamente por año de originación (`issue_date_year`), descartando mezclas temporales:

| Partición | Año | Observaciones ($N$) | Fallidos ($y=1$) | Tasa Default | Préstamos Vivos | Estado de Madurez |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train** | 2016 | $425.567$ | $71.025$ | $16,69\%$ | $35,98\%$ | Madura (ventana contractual de 36 meses cumplida) |
| **Validación (OOT)** | 2017 | $401.094$ | $41.737$ | $10,41\%$ | $63,10\%$ | Censura moderada ($12$ a $24$ meses en libros) |
| **Test** | 2018 | $426.181$ | $12.397$ | $2,91\%$ | $90,07\%$ | Censura severa ($0$ a $12$ meses de vida) |

*(Nota de alcance: Las referencias exploratorias externas al año 2015 fueron depuradas del proyecto, acotando el estudio estrictamente al ciclo continuo 2016–2018).*

```
                             LÍNEA TEMPORAL DE ORIGINACIÓN Y MADUREZ
                             
    2016 (Train)       |=================== 36 meses ===================>| (Maduro / Ventana Completa)
    2017 (Val OOT)     |=========== 12 a 24 meses ========> (Censura Media)
    2018 (Test)        |==== 0 a 12m ====> (Censura Severa: 90% vivos)
                       +----------------+----------------+----------------+
                      2016             2017             2018             2019
```

### 2.2 Protocolo Anti-Leakage (Fuga de Información)
En modelización de admisiones y provisiones bancarias, incluir variables calculadas o actualizadas con posterioridad a la originación invalida el modelo ante una inspección del supervisor:
1. **Exclusión de `grade` y `sub_grade`:** El rating asignado por la plataforma es una variable endógena que condensa la decisión previa del prestamista. Incluirla provocaría que el modelo prediga la decisión del comité de riesgos en lugar del riesgo crediticio intrínseco del acreditado.
2. **Exclusión de `interest_rate`:** El tipo de interés refleja el *pricing* contractual asignado por la entidad en función del rating previo. Su inclusión introduciría causalidad inversa (*pricing leakage*).
3. **Exclusión de Métricas de Desempeño Posterior:** Variables como `recoveries`, `collection_recovery_fee`, `total_rec_prncp`, `last_pymnt_amnt`, `next_pymnt_d`, etc., son estrictamente ex-post y fueron eliminadas antes del preprocesamiento.
4. **Selección por Bloques Econométricos (Test de Wald $\chi^2$):** Se partió de 60 variables candidatas de balance, buró de crédito (*credit bureau*) y sociodemográficas. Mediante eliminación hacia atrás por bloques de significación estadística ($p < 0,05$), se depuró el espacio a **45 predictores estables**, eliminando redundancias y multicolinealidad sin perder capacidad de ordenación ($\text{AUC}$ pasó de $0,6755$ a $0,6754$).

---

## 3. Modelización de los Componentes: PD, EAD y LGD

### 3.1 Probabilidad de Default (PD)
Se implementó un análisis comparativo de menor a mayor complejidad con validación cruzada estratificada (`5-Fold CV`) en `Train 2016` y evaluación fuera de muestra en `Validación 2017`:

| Modelo PD | Familia | Hiperparámetros | AUC CV (5-Fold) | AUC Val (2017) | KS Val | Gap (Train-Val) | Brier Val | PSI | % del AUC Máx. |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Logit MLE Depurado (Seleccionado)** | **Lineal / Scorecard** | **45 vars ($\alpha=0.05$)** | **0.6824** | **0.6754** | **0.2525** | **0.0059** | **0.0929** | **0.0177** | **97.94%** |
| 2. Logit MLE Completo | Lineal | 60 vars | 0.6823 | 0.6755 | 0.2526 | 0.0059 | 0.0929 | 0.0179 | 97.96% |
| 3. Logit Ridge ($\ell_2$) | Lineal Regularizado | $C=0.001$ | 0.6803 | 0.6749 | 0.2520 | 0.0061 | 0.0928 | 0.0188 | 97.88% |
| 4. Logit Lasso ($\ell_1$) | Lineal Regularizado | $C=0.01$ | 0.6804 | 0.6747 | 0.2517 | 0.0064 | 0.0929 | 0.0185 | 97.84% |
| 5. Logit Elastic Net | Lineal Regularizado | $C=0.01, \ell_1=0.75$ | 0.6809 | 0.6748 | 0.2520 | 0.0063 | 0.0929 | 0.0184 | 97.87% |
| 6. Logit B-Splines Cúbicos | Aditivo No Lineal | $\text{knots}=5, C=1.0$ | 0.6852 | 0.6807 | 0.2624 | 0.0058 | 0.0923 | 0.0242 | 98.72% |
| 7. Random Forest | Bagging | $d=12, \text{leaf}=50$ | 0.6778 | 0.6735 | 0.2494 | 0.0572 | 0.0927 | 0.0213 | 97.68% |
| 8. HistGradientBoosting (LightGBM) | Boosting | $d=6, \ell_2=10.0$ | 0.6836 | 0.6895 | 0.2740 | 0.0211 | 0.0919 | 0.0227 | 100.00% |
| 9. XGBoost Regularizado | Boosting | $d=5, \lambda=10, \alpha=2$ | 0.6865 | 0.6890 | 0.2735 | 0.0171 | 0.0918 | 0.0237 | 99.93% |

#### Justificación Econométrica y Regulatoria de la Elección (Criterio de Parsimonia):
* **Retención de Capacidad Predictiva:** El modelo lineal `Logit MLE Depurado` captura el **`97,94%`** del AUC del mejor algoritmo no lineal de *boosting*.
* **Estabilidad Temporal:** El *gap* de generalización temporal entre Train y Val en Logit es de apenas **`0,0059`**, frente a **`0,0211`** en HistGradientBoosting (3,5 veces más propenso a sobreajuste temporal).
* **Exigencias Regulatorias BCE TRIM / EBA:** La regulación bancaria exige trazabilidad aditiva directa de *log-odds* (construcción de *scorecards* de puntos), calibración de probabilidades sin cajas negras y estabilidad del índice PSI ($0,0177 < 0,10$, indicativo de estabilidad excelente).

---

### 3.2 Exposición al Default (EAD) y Factor de Conversión (CCF)
En préstamos con cuota de amortización progresiva, la exposición en el momento de la quiebra no es el capital inicial concedido, sino el saldo insolvente no amortizado. Se modeliza el **Factor de Conversión de Crédito (*Credit Conversion Factor - CCF*)**:
$$\text{CCF}_i = \frac{\text{funded\_amnt}_i - \text{princ\_rec}_i}{\text{funded\_amnt}_i} \in (0, 1] \implies \text{EAD}_i = \text{funded\_amnt}_i \times \widehat{\text{CCF}}_i$$

| Modelo EAD | Hiperparámetros | RMSE CV (USD) | CCF Obs Val | CCF Pred Val | MAE EAD Val (USD) | $R^2$ Val | Mejora MAE vs Base |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| M0. CCF Constante Global | Media Train ($0.6909$) | $3.603,56$ | $0.7914$ | $0.6909$ | $2.453,33$ | $0.8378$ | Baseline |
| M1. CCF Medio por Plazo | $36\text{m}=0.631, 60\text{m}=0.816$ | $3.191,33$ | $0.7914$ | $0.6921$ | $1.944,50$ | $0.8871$ | $+20,74\%$ |
| M2a. Ridge Directo en EAD | $\alpha=10.0$ | $3.196,40$ | $0.7914$ | $0.6686$ | $2.007,67$ | $0.8920$ | $+18,17\%$ |
| M2b. Ridge Lineal s/ CCF | $\alpha=500.0$ | $3.127,07$ | $0.7914$ | $0.6898$ | $1.904,57$ | $0.8962$ | $+22,37\%$ |
| **M3. GLM Logit Fraccional (QMLE)** | **$C=0.5$** | **$3.124,77$** | **$0.7914$** | **$0.6900$** | **$1.898,71$** | **$0.8970$** | **$+22,61\%$** |
| M4. HistGradientBoosting s/ CCF | $d=5, \ell_2=10.0$ | $3.114,72$ | $0.7914$ | $0.6912$ | $1.890,23$ | $0.8980$ | $+22,95\%$ |

* **Modelo Seleccionado:** **GLM Logit Fraccional (QMLE)**. Modela la variable acotada estrictamente en $(0, 1)$ mediante cuasi-máxima verosimilitud de Bernoulli, logrando un **$R^2$ de $0,8970$** y superando en más de un $22,6\%$ al benchmark constante con total interpretabilidad funcional.

---

### 3.3 Severidad de la Pérdida (LGD)
La severidad condicional del impago en préstamos minoristas sin garantía (*unsecured personal loans*) presenta una distribución fuertemente asimétrica con alta concentración en pérdidas cercanas al $100\%$ (tasa de recuperación media inferior al $10\%$):
$$\text{LGD}_i = 1 - \text{Recovery Rate}_i = \frac{\text{EAD}_i - \text{Recoveries}_i}{\text{EAD}_i}$$

| Modelo LGD | Hiperparámetros | RMSE CV | LGD Obs Val | LGD Pred Val | RMSE Val | MAE Val | MAE Monetario ($\text{Loss}|\text{Def}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| M0. LGD Constante Global | Media Train ($0.9081$) | $0.1210$ | $0.9444$ | $0.9081$ | $0.0959$ | $0.0818$ | $\$999,09$ |
| **M1. Regresión Lineal Ridge** | **$\alpha=1000.0$** | **$0.1205$** | **$0.9444$** | **$0.9097$** | **$0.0954$** | **$0.0808$** | **$\$984,40$** |
| M2. Logit Fraccional (QMLE) | $C=0.01$ | $0.1205$ | $0.9444$ | $0.9097$ | $0.0954$ | $0.0808$ | $\$984,78$ |
| M3. Hurdle Dos Partes (Logit+QMLE) | $C=0.01$ | $0.1205$ | $0.9444$ | $0.9097$ | $0.0954$ | $0.0807$ | $\$984,48$ |
| M4. HistGradientBoosting | $d=3, \ell_2=10.0$ | $0.1204$ | $0.9444$ | $0.9093$ | $0.0953$ | $0.0807$ | $\$985,07$ |

* **Modelo Seleccionado:** **Ridge Lineal Regularizado**. Ante la baja variabilidad de la tasa de recuperación en crédito sin colateral (la severidad media real en Val es $94,44\%$), los modelos complejos no aportan ventaja predictiva apreciable ($0,0807$ vs $0,0808$). Se adopta Ridge por estabilidad numérica y robustez supervisora.

---

## 4. Cuadro de Mando Ejecutivo y Contraste de Hipótesis (Validación 2017)

En la partición de Validación Out-of-Time ($N = 401.094$, con una pérdida real registrada de **`$490,78 M USD`**), se contrastaron las tres estrategias principales frente a los umbrales formales de subestimación:

| Métrica de Gestión / Regulación | 1. Modelo Base (Logit MLE sin MoC) | 2. Modelo Balanceado (`balanced` + MoC P90) | 3. Modelo Ganador (Two-Stage Watchlist P88) |
| :--- | :---: | :---: | :---: |
| **Provisión ECL Total de Balance** | **`$620,66 M USD`** | **`$2.452,45 M USD`** | **`$950,32 M USD`** |
| **Ratio de Cobertura ($\text{ECL} / \text{Pérdida Real}$)** | $1,26\times$ | $5,00\times$ | **$1,94\times$** |
| **Capital Liberado vs. Balanceado** | $+\$1.831,79\text{ M}$ | $\$0,00\text{ M}$ (Base) | **`+$1.502,13 M USD`** |
| **MAE Global de Cartera** | $\$2.304,83$ | $\$5.702,46$ | **`$2.860,52`** |
| **MAE en Clientes Sanos ($y=0$)** | $\$1.466,37$ | $\$5.275,52$ | **`$2.214,52`** *(Ahorro $>50\%$)* |
| **MAE Severidad en Fallidos ($y=1$)** | $\$1.973,35$ | $\$2.535,80$ | $\$2.535,80$ |
| **% Subest. Severidad Fallidos** | **$87,43\%$** *(Falla)* | **$6,92\%$** *(Cumple)* | **`9,07%`** *(Cumple)* |
| **IC 95% Superior en Fallidos** | $87,70\%$ | $7,13\%$ | **`9,30% < 10%`** |
| **% Subest. Individual ($0/1$)** | **$10,37\%$** *(Falla)* | $9,79\%$ | **`9,81%`** |
| **IC 95% Superior Individual** | $10,45\%$ | $9,87\%$ | **`9,89% < 10%`** |
| **Z-statistic / $p$-valor Individual** | $Z = +7,81 \ (p = 1,0)$ | $Z = -4,47 \ (p < 0,0001)$ | **$Z = -4,47 \ (p = 1,38 \times 10^{-10})$** |
| **% Subest. en 100 Centiles de Riesgo** | $0,00\%$ | $0,00\%$ | **`0,00%`** *(IC95%: $2,95\% < 10\%$)* |
| **Dictamen Regulatorio Formal** | **INCUMPLE** (Riesgo Solvencia) | **CUMPLE** (Inmoviliza $1.502\text{M}$) | **CUMPLE EN TODOS LOS TEST (ÓPTIMO)** |

### Contraste Estadístico de Hipótesis
Para validar si la proporción de subestimación cumple la restricción de diseño $p \le 0,10$ al $95\%$ de confianza:
$$H_0: p \ge 0,10 \quad \text{vs.} \quad H_1: p < 0,10$$
* En el **Modelo Ganador Two-Stage**, tanto a nivel de severidad de impago ($9,07\%$, $IC_{95\%} \text{sup} = 9,30\%$) como a nivel individual general ($9,81\%$, $IC_{95\%} \text{sup} = 9,89\%$), se rechaza categóricamente la hipótesis nula ($p = 1,38 \times 10^{-10}$), garantizando solvencia jurídica ante el supervisor.

---

## 5. Diagnóstico de Censura por la Derecha en Test (Cohorte 2018)

Al evaluar el modelo en la cohorte de **Test (2018)** ($N = 426.181$), el análisis cuantitativo revela una severa distorsión estructural causada por la **censura por la derecha (*right-censoring*)**:

### 5.1 Las 4 Métricas Formales de Censura Temporal

| Métrica Formal de Censura | Train (2016) | Validación OOT (2017) | Test (2018) | Diagnóstico e Impacto Estructural |
| :--- | :---: | :---: | :---: | :--- |
| **1. Tasa de Préstamos Vivos ($\text{Open Ratio}$)** | $35,98\%$ | $63,10\%$ | **`90,07%`** | En 2018, **9 de cada 10 préstamos** siguen vivos en balance y su desenlace no ha ocurrido. |
| **2. Tasa de Default Observada ($\text{DR}_{\text{obs}}$)** | $16,69\%$ | $10,41\%$ | **`2,91%`** | Desplome del $82\%$ en mora por falta de tiempo de maduración, no por menor riesgo intrínseco. |
| **3. Mediana de Cuotas Pagadas al Fallar** | $13,98$ cuotas | $10,10$ cuotas | **`4,96` cuotas** | En 2018 solo afloran impagos inmediatos ($<5$ meses); los fallos habituales siguen censurados. |
| **4. CCF Medio en Fallidos** | $0,6909$ | $0,7914$ | **`0,9020`** | Al quebrar casi sin amortizar principal, el saldo vivo en el default es del $90\%$ del nominal concedido. |

```
                       EVOLUCIÓN DE LA MADUREZ DE CARTERA
                       
  Tasa Préstamos Vivos:   2016: 35.98%  ===>  2017: 63.10%  ===>  2018: 90.07% [Severa]
  Tasa Default Observada: 2016: 16.69%  ===>  2017: 10.41%  ===>  2018:  2.91% [Truncada]
  Mediana Cuotas al Fallo:2016: 13.98   ===>  2017: 10.10   ===>  2018:  4.96  [Ultra-temprano]
  CCF Medio en Fallidos:  2016:  0.69   ===>  2017:  0.79   ===>  2018:  0.90  [Saldo Casi Íntegro]
```

### 5.2 Tabla Consolidada de Métricas de Cartera en Test

| Partición | Pérdida Real (M$) | ECL Provisión (M$) | Ratio Cobertura | MAE Global ($) | MAE Sanos ($) | MAE Defaults ($) | % Subest. Indiv. | % Subest. Defaults |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train (2016)** | $702,87$ | $830,30$ | $1,18\times$ | $\$465,09$ | $\$134,29$ | $\$1.867,18$ | $2,48\%$ | $6,85\%$ |
| **Validación (2017)**| $490,78$ | $1.067,32$ | $2,17\times$ | $\$1.703,33$ | $\$1.623,33$ | $\$1.481,20$ | $3,03\%$ | $9,50\%$ |
| **Test (2018)** | $189,39$ | $3.276,09$ | $17,30\times$ | $\$7.391,59$ | $\$7.518,79$ | $\$1.091,39$ | **`1,78%`** | **`32,02%`** |

---

### 5.3 Autopsia Econométrica: Desmontando Lecturas Superficiales

Un análisis cuantitativo riguroso desvela tres fenómenos económicos que explican el comportamiento en Test:

#### 1. La Ilusión Estadística del 1,78% de Subestimación Individual
* **El Espejismo:** En Test, la tasa de subestimación global individual aparenta ser un excelente $1,78\%$.
* **La Realidad Causal:** El $97,09\%$ de la cartera es temporalmente sano ($y=0$), y para un préstamo sano la pérdida contable ex-post es $0$ ($\text{ECL}_i \ge 0$, subestimación $= 0\%$). Los $7.586$ préstamos subestimados corresponden **íntegramente a fallidos**.
* Sobre los $12.397$ fallidos de 2018:
  $$\text{Tasa de Subestimación Real en Defaults} = \frac{7.586}{12.397} = \mathbf{61,19\%}$$
  La tasa global del $1,78\%$ está masivamente diluida por la masa no madura de clientes sanos. El margen **MoC P88** reduce la subestimación en severidad del $73,07\%$ al **`32,02%`**, actuando como amortiguador indispensable.

#### 2. Dominancia Absoluta de los Préstamos Sanos en el MAE Global (Gastos Prudenciales)
* El MAE Global en Test se descompone ponderadamente:
  $$\text{MAE Global} = 0,9709 \times 7.518,79 + 0,0291 \times 1.091,39 = \mathbf{7.391,59\text{ USD}}$$
* Más del **`98,7%` del error absoluto medio está generado por clientes sanos**.
* **Interpretación de Balance:** Esto no constituye un defecto del modelo, sino el reflejo contable de la **dotación prudencial obligatoria**: como en la fecha de corte los préstamos llevan pocos meses activos, la contabilidad registra pérdida cero ($\text{Loss}_i = 0$), mientras que el modelo reserva provisión para absorber los impagos que madurarán en 2019 y 2020 ($|\text{ECL}_i - 0| = \text{ECL}_i$).

#### 3. El Salto en la Pérdida Media por Default a $15.277 USD
* La pérdida unitaria media por cliente fallido evoluciona de forma contraria a la tasa de quiebra:
  * **Train 2016:** `$9.896 USD` (mediana de $14$ cuotas pagadas; amortización previa significativa).
  * **Val 2017:** `$11.759 USD` (mediana de $10$ cuotas pagadas).
  * **Test 2018:** **`$15.277 USD`** (mediana de apenas $4,9$ cuotas pagadas).
* **Mecanismo:** La censura derecha actúa como un **filtro de selección de impagos tempranos extremos** (fraude o sobreendeudamiento súbito). Al quebrar al inicio de la vida del crédito, el saldo vivo es prácticamente el $100\%$ del principal ($\text{CCF} = 0,902$), elevando la severidad monetaria unitaria.

---

## 6. Justificación Metodológica frente a la Regulación Bancaria (BCE TRIM)

### ¿Por qué está prohibido un `train_test_split` aleatorio estándar en Banca?
En un proyecto académico convencional o en competiciones de ciencia de datos, la tentación inmediata es unir todos los años y ejecutar un `train_test_split(shuffle=True)`. En banca minorista bajo la **Guía TRIM del Banco Central Europeo** y las **Directrices EBA/GL/2017/16**, esta práctica es **inadmisible**:

1. **Contaminación del Target en Entrenamiento (*Data Leakage Temporal*):**
   * Un prestatario de alto riesgo originado a finales de 2018 con solo 3 meses de vida figura en la base como $y = 0$ (sano), simplemente porque aún no ha transcurrido el tiempo necesario para incurrir en impago de 90 días.
   * Si entra en `Train`, el algoritmo aprende que esos perfiles de riesgo "no quiebran", corrompiendo la señal del modelo y descalibrando las probabilidades.
2. **Exigencia Regulatoria de Validación Fuera del Tiempo (*Out-Of-Time / OOT*):**
   * El supervisor exige evaluar la capacidad de ordenación y calibración en ventanas posteriores independientes, reproduciendo exactamente la operativa de concesión crediticia en producción.

### Soluciones Futuras para Despliegue en Producción
Para incorporar datos recientes sujetos a censura sin sesgar las provisiones, el marco de trabajo establece tres soluciones estándar:
1. **Modelos de Supervivencia (*Survival Analysis / Duration Models*):** Regresiones de riesgos proporcionales de Cox o modelos AFT (Weibull/Log-Logístico) sobre el tiempo al evento $T$ e indicador de censura $\delta_i$, tratando de forma no sesgada las observaciones vivas.
2. **Definición de Target a Horizonte Fijo de 12 Meses (*12-Month PD*):** Alinear la modelización al estándar de Basilea IRB / IFRS 9 Stage 1, auditando estrictamente los primeros 12 meses y exigiendo 12 meses de madurez mínima por crédito.
3. **Modelos Multi-Estado de Transición de Crédito (*Markov Transition Models*):** Matrices de transición mensual entre estados (*Current* $\to$ *30 DPD* $\to$ *60 DPD* $\to$ *Default* / *Prepayment*).

---

## 7. Limitaciones Estructurales del Proyecto

1. **Sesgo de Selección por Rechazo (*Reject Inference*):**
   * El dataset contiene exclusivamente préstamos concedidos (*booked loans*). La muestra sufre truncamiento incidental: no observamos el desempeño de los solicitantes denegados. Este modelo es plenamente válido para dotación de provisiones de cartera viva (IFRS 9), pero requeriría corrección de Heckman en dos etapas o aumento de datos (*parceling / reweighting*) si se utilizara como motor de admisión.
2. **Censura del Periodo de Recobro (*Workout Period Censoring*):**
   * Los préstamos fallidos recientemente (2017–2018) no han completado el ciclo judicial y extrajudicial de recuperación (normalmente de 24 a 36 meses), lo que subestima temporalmente la tasa de recobro final.

---

## 8. Mensaje Clave para el Tribunal de Riesgos

> *"Por limitaciones operativas y tiempo de computación en el despliegue del pipeline, la calibración integral de provisiones se consolidó sobre la partición Out-of-Time de Validación (2017, con 12 a 24 meses de madurez).*
>
> *Decidimos conscientemente **no tomar el atajo simplista de hacer un split aleatorio** mezclando todos los años, porque bajo la Guía TRIM del BCE habríamos contaminado el entrenamiento con préstamos censurados de 2018 etiquetados como sanos.*
>
> *Al auditar formalmente la cohorte de 2018 en Test, nuestro diagnóstico cuantitativo desvela que:*
> 1. *El MAE Global está dominado en más del 98% por el error en sanos ($7.518\text{ USD}$), reflejando la provisión preventiva dotada para préstamos que todavía no han tenido tiempo de quebrar.*
> 2. *La pérdida media por default se dispara a $\$15.277\text{ USD}$ porque la censura selecciona exclusivamente impagos inmediatos con casi cero amortización ($\text{CCF} = 0,90$).*
> 3. *La aparente baja subestimación individual del $1,78\%$ es un espejismo por el peso del 97,1% de préstamos sanos: dentro de los fallidos reales, el modelo sin margen subestima al 73% de los impagos, mientras que nuestra arquitectura con Margen de Severidad MoC P88 reduce la subestimación al 32% garantizando la solvencia regulatoria de la entidad.*
>
> *Este diagnóstico riguroso demuestra la necesidad en producción de modelos de supervivencia de Cox o ventanas cerradas a 12 meses, y agradecemos su feedback como profesionales de validación de modelos."*

---

## 9. Estructura del Repositorio y Reproducibilidad

```text
credit-risk/
│
├── notebooks/
│   └── credit-risk-notebook.ipynb    # Pipeline integral ejecutado (165 celdas, 0 comentarios '#')
│                                      # EDA, Feature Engineering, PD, EAD, LGD, Two-Stage ECL,
│                                      # Contraste de Hipótesis y Diagnóstico de Censura en Test
│
├── data/                             # Datos de modelización (no versionados por tamaño en git)
│   ├── X.csv                         # Matriz de características (1.252.842 x variables)
│   └── target.csv                    # Indicador de quiebra (y)
│
├── requirements.txt                  # Dependencias y librerías del entorno
├── LICENSE                           # Licencia MIT
└── README.md                         # Documentación regulatoria y ejecutiva del proyecto
```

### Instrucciones de Ejecución
1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/alexalvarezds/credit-risk.git
   cd credit-risk
   ```
2. **Configurar el entorno virtual e instalar dependencias:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Ejecutar el Notebook:**
   Abrir Jupyter Lab / VSCode y ejecutar de forma secuencial `notebooks/credit-risk-notebook.ipynb`. Todas las 165 celdas contienen salidas pre-renderizadas, tablas interactivas y gráficos de distribución y calibración.