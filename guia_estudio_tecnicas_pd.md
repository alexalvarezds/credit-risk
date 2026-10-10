# Guía de Estudio e Investigación: Técnicas Avanzadas de Modelado e Inferencia para Credit Risk (PD y Expected Loss)

Esta guía recopila y desarrolla en profundidad todas las técnicas estadísticas, econométricas y de *Machine Learning* aplicadas en el cuaderno [`notebooks/02-eda-full-dataset.ipynb`](notebooks/02-eda-full-dataset.ipynb). Está estructurada para servir como **hoja de ruta de estudio**, conectando la intuición matemática, la implementación práctica, su utilidad en negocio bancario (**Expected Loss**, $\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$) y la bibliografía de referencia.

---

## Índice de Bloques Temáticos

1. [Arquitectura de Datos en Riesgo de Crédito: Censura, Ventanas y Sesgos](#1-arquitectura-de-datos-en-riesgo-de-crédito-censura-ventanas-y-sesgos)
2. [Econometría de la Regresión Logística (MLE) e Inferencia Asintótica](#2-econometría-de-la-regresión-logística-mle-e-inferencia-asintótica)
3. [Modelado No Lineal de Caja Blanca: Scorecard WoE, PAVA y B-Splines Ortogonalizados](#3-modelado-no-lineal-de-caja-blanca-scorecard-woe-pava-y-b-splines-ortogonalizados)
4. [Modelos Aditivos (GAM / EBM) y Boosting con Restricciones Monótonas](#4-modelos-aditivos-gam--ebm-y-boosting-con-restricciones-monótonas)
5. [Métricas de Discriminación y Contrastes Formales (DeLong y Bootstrap)](#5-métricas-de-discriminación-y-contrastes-formales-delong-y-bootstrap)
6. [Calibración de Probabilidades, Estabilidad (PSI) y Expected Loss ($\text{EL}$)](#6-calibración-de-probabilidades-estabilidad-psi-y-expected-loss-el)
7. [Bibliografía y Referencias Clave para Profundizar](#7-bibliografía-y-referencias-clave-para-profundizar)

---

## 1. Arquitectura de Datos en Riesgo de Crédito: Censura, Ventanas y Sesgos

### 1.1 Censura por la Derecha (*Right-Censoring*) y Ventana Fija de Observación
* **Qué es:** En análisis de supervivencia y riesgo de crédito, una observación está **censurada por la derecha** cuando el periodo de seguimiento termina antes de que finalice el contrato y sin que el evento (el impago o *default*) haya ocurrido todavía.
* **Por qué ocurre en LendingClub:** Si defines el target simplemente como `loan_status == "Charged Off"` en la fecha de descarga del dataset (marzo de 2019), un préstamo de 60 meses concedido en 2017 que sigue al corriente (`Current`) lleva solo ~18 meses de vida. Etiquetarlo como `0` ("buen pagador") asume implícitamente que no impagará en los 42 meses restantes.
* **Cómo se resuelve en la industria bancaria (*Cohort / Fixed-Window Approach*):**
  1. **PD a 12 meses (*12-Month PD*, Basilea IRB e IFRS 9 *Stage 1*):** Se fija una ventana temporal estricta de $W = 12$ meses desde la originación ($t_0$). Solo entran al entrenamiento préstamos con antigüedad $\ge 12$ meses, y la variable objetivo vale $y_{12\text{m}} = 1$ **si y solo si** el impago ocurre dentro del intervalo $[t_0, t_0 + 12\text{m}]$. Si el cliente impaga en el mes 28, para el modelo de PD a 12 meses de originación su etiqueta es $0$.
  2. **PD de Vida Completa (*Lifetime PD*, IFRS 9 *Stage 2*):** Requiere modelos de supervivencia (*Cox Proportional Hazards*, curvas *Kaplan-Meier* o modelos discretos mes a mes por edad del préstamo / *vintage analysis*) que traten explícitamente los préstamos `Current` como observaciones censuradas en su mes actual de exposición.
* **Qué hicimos en el cuaderno (Sección 15.2):** Utilizamos el saldo vivo (`remaining_princ_for_tot_amnt_fund > 0`) para auditar qué porcentaje exacto de préstamos seguía abierto en cada año, y reconstruimos la ventana fija de 12 cuotas equivalentes pagadas ($\frac{\text{princ\_rec} + \text{interest\_rec}}{\text{monthly\_payment}} \le 12.5$) exclusivamente para construir $y_{12\text{m}}$, demostrando cómo desaparece el desplome artificial de la tasa de default en 2017.
* **Conceptos clave para buscar:** *Right-censoring in credit risk*, *Fixed performance window PD*, *Vintage analysis / Cure rate*, *IFRS 9 12-month vs Lifetime PD*, *Discrete-time hazard models*.

### 1.2 Circularidad de Precios (*Endogeneity of Pricing*) y Sesgo de Selección (*Reject Inference*)
* **Endogeneidad del tipo de interés (`interest_rate`) y `grade` (Sección 8.1):**
  * En plataformas o bancos donde el tipo de interés se fija en función del riesgo percibido ($\text{interest\_rate} = g(\text{Score interno}, \text{Plazo})$), meter `interest_rate` como predictor en un modelo que pretende calificar nuevas solicitudes produce **circularidad**: no puedes usar el precio del préstamo para estimar la PD con la que vas a calcular el precio del préstamo. Además, `interest_rate` absorbe la señal de las demás variables y distorsiona sus signos condicionales (como vimos con `DirectPay` y `num_inq_in_12mths`). Por eso en la **Variante C** prescindimos tanto de `grade` como de `interest_rate`.
* **Sesgo de truncamiento muestral (*Sample Selection Bias / Reject Inference*):**
  * Solo observamos el comportamiento de pago de los solicitantes **aprobados**. Los rechazados no tienen etiqueta $y$. Como la muestra aprobada ya ha pasado un filtro de solvencia, la varianza de las características $X$ está restringida (*range restriction*), lo que hace que el ROC-AUC observado en aprobados (`~0.68–0.71`) sea notablemente inferior al que tendría el mismo modelo sobre toda la población solicitante.
* **Conceptos clave para buscar:** *Reject inference in credit scoring*, *Heckman selection bias*, *Endogeneity of interest rate in default models*, *Adverse selection in lending*.

### 1.3 Nulos Informativos (*Missing Not At Random* — MNAR) y Descomposición Indicador + Mediana Condicional
* **El problema:** En variables de buró como `mths_since_last_delinq` (meses desde la última mora), el valor es `NaN` cuando el cliente **nunca ha sido moroso**. Si imputas un valor numérico arbitrario (como `-1` o `999`) y metes la variable en un modelo lineal, obligas a que la misma pendiente $\beta$ explique dos cosas opuestas: tener o no tener historial de mora, y cuán reciente fue la mora entre quienes sí la tuvieron.
* **La formulación econométrica limpia (Sección 8):**
  Para cualquier variable continua $X_j$ que solo existe cuando ocurre un evento previo:
  $$\beta_{\text{flag}} \cdot \mathbb{I}(X_j \text{ observado}) + \beta_{\text{cont}} \cdot \mathbb{I}(X_j \text{ observado}) \cdot (X_j - m_{\text{Train}})$$
  Al definir la bandera `has_delinq` $= \mathbb{I}(X_j \text{ observado})$ y rellenar los `NaN` de $X_j$ con la mediana de los observados en `Train` ($m_{\text{Train}}$), el término $(X_j - m_{\text{Train}})$ vale exactamente `0` para los clientes sin mora. Así:
  * $\beta_{\text{flag}}$ mide exactamente la diferencia de *log-odds* entre un cliente con una mora de antigüedad mediana y un cliente con historial limpio.
  * $\beta_{\text{cont}}$ mide exclusivamente el efecto de que la mora sea más reciente o más lejana entre quienes sí tienen mora.

---

## 2. Econometría de la Regresión Logística (MLE) e Inferencia Asintótica

### 2.1 Estimación por Máxima Verosimilitud (MLE) y la Propiedad del Intercepto
* **Formulación:** Dado $y_i \in \{0, 1\}$ y el predictor lineal $\eta_i = \beta_0 + \mathbf{x}_i^\top \boldsymbol{\beta}$, la regresión logística modela:
  $$p_i = P(y_i = 1 \mid \mathbf{x}_i) = \sigma(\eta_i) = \frac{1}{1 + e^{-\eta_i}}$$
  Maximizamos la log-verosimilitud de Bernoulli:
  $$\ell(\beta_0, \boldsymbol{\beta}) = \sum_{i=1}^N \Big[ y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \Big]$$
* **Condición de primer orden (Ecuaciones de Score):**
  Derivando respecto al intercepto $\beta_0$ y al vector $\boldsymbol{\beta}$ e igualando a cero:
  $$\frac{\partial \ell}{\partial \beta_0} = \sum_{i=1}^N (y_i - \hat{p}_i) = 0 \implies \frac{1}{N}\sum_{i=1}^N \hat{p}_i = \bar{y}$$
  $$\frac{\partial \ell}{\partial \boldsymbol{\beta}} = \sum_{i=1}^N (y_i - \hat{p}_i)\mathbf{x}_i = \mathbf{0}$$
  * **Consecuencia clave (Sección 12.1):** Todo modelo logístico con intercepto estimado por máxima verosimilitud sin penalización reproduce **por identidad algebraica** la tasa media de default del conjunto de entrenamiento ($\bar{\hat{p}}_{\text{Train}} \equiv \bar{y}_{\text{Train}}$). En cambio, usar `class_weight="balanced"` multiplica cada término de la verosimilitud por pesos artificiales $w_1, w_0$, desplazando el intercepto y destruyendo la calibración de las probabilidades.

### 2.2 Matriz de Información de Fisher y Distribución Asintótica de $\hat{\boldsymbol{\beta}}$
* **Qué es:** La curvatura negativa (Hessiano cambiado de signo) de la log-verosimilitud en el óptimo es la **Matriz de Información de Fisher observada**:
  $$\mathcal{I}(\hat{\boldsymbol{\beta}}) = -\frac{\partial^2 \ell}{\partial \boldsymbol{\beta}\,\partial \boldsymbol{\beta}^\top} = \tilde{X}^\top W \tilde{X}$$
  donde $\tilde{X} = [\mathbf{1}, X] \in \mathbb{R}^{N \times (p+1)}$ es la matriz de diseño y $W = \text{diag}\big(\hat{p}_1(1-\hat{p}_1), \dots, \hat{p}_N(1-\hat{p}_N)\big)$ es la matriz diagonal de varianzas de Bernoulli de cada préstamo.
* **Por qué es fundamental:** Por el Teorema Central del Límite para estimadores de máxima verosimilitud, cuando $N$ es grande:
  $$\hat{\boldsymbol{\beta}} \;\underset{\text{asint.}}{\sim}\; \mathcal{N}\Big(\boldsymbol{\beta}, \; \widehat{\Sigma}\Big), \qquad \text{con } \widehat{\Sigma} = \big(\tilde{X}^\top W \tilde{X}\big)^{-1}$$
  * La raíz cuadrada de la diagonal principal de $\widehat{\Sigma}$ nos da los **errores estándar exactos** $\text{SE}(\hat{\beta}_j) = \sqrt{\widehat{\Sigma}_{jj}}$.
  * Los términos fuera de la diagonal $\widehat{\Sigma}_{jk}$ miden la **covarianza entre coeficientes estimados**, imprescindible para contrastar bloques de variables o construir bandas de confianza de curvas spline.
* **Conceptos clave para buscar:** *Observed Fisher Information Matrix in GLM*, *Iteratively Reweighted Least Squares (IRLS)*, *Asymptotic covariance of MLE*.

### 2.3 Factor de Inflación de la Varianza (VIF) y el Efecto de la Categoría de Referencia
* **Fórmula:** $\text{VIF}_j = \frac{1}{1 - R_j^2} = (\mathbf{R}_{XX}^{-1})_{jj}$, donde $R_j^2$ es el coeficiente de determinación de regresar la columna $X_j$ sobre las demás columnas de $X$, y $\mathbf{R}_{XX}$ es la matriz de correlaciones de los predictores.
* **Por qué importa la categoría de referencia en las *dummies* (Sección 8 y 11.1):**
  Si una variable categórica tiene una categoría minúscula (como `car` en `loan_purpose`, $<1.5\%$ de los datos) y la dejas como referencia omitida, la suma de las restantes *dummies* vale `1` para el $98.5\%$ de las filas (casi idéntica a la columna constante del intercepto), haciendo que todas las *dummies* parezcan fuertemente colineales entre sí. Al elegir siempre **la moda en `Train`** (`debt_consolidation`, $>55\%$ de los datos) como categoría de referencia, las *dummies* restantes son dispersas y casi ortogonales, reduciendo el VIF por debajo de $5$.

### 2.4 Test de Wald Individual ($z$) vs. Test de Wald Conjunto por Bloques ($\chi^2$)
* **Individual (1 gl):** Para un único parámetro $\beta_j$, bajo $H_0: \beta_j = 0$:
  $$z_j = \frac{\hat{\beta}_j}{\sqrt{\widehat{\Sigma}_{jj}}} \sim \mathcal{N}(0, 1), \qquad z_j^2 \sim \chi^2_1$$
* **Conjunto por Bloque ($k$ gl — Sección 11.2 y 14.2):**
  Cuando una variable conceptual se expande en $k$ columnas en la matriz de diseño (por ejemplo, las $k-1$ *dummies* de `loan_purpose` o las $3$ bases $[z_j, c_{j,1}, c_{j,2}]$ de un spline), sus estimadores $\hat{\boldsymbol{\beta}}_G \in \mathbb{R}^k$ están correlacionados entre sí según la submatriz $\widehat{\Sigma}_{GG} \in \mathbb{R}^{k \times k}$. Para contrastar simultáneamente $H_0: \boldsymbol{\beta}_G = \mathbf{0}$ se utiliza la forma cuadrática de Mahalanobis (Estadístico de Wald Multiparamétrico):
  $$W_G = \hat{\boldsymbol{\beta}}_G^\top \, \widehat{\Sigma}_{GG}^{-1} \, \hat{\boldsymbol{\beta}}_G \;\underset{H_0}{\sim}\; \chi^2_k$$
  * **Regla de oro:** Nunca elimines una *dummy* individual ni una base spline suelta mirando su $z_j$ individual; evalúa y elimina siempre el bloque completo mediante $W_G$.

### 2.5 Criterios de Información (AIC y BIC) y Razón de Verosimilitudes
* **Qué miden:** Cuando añadimos parámetros al modelo (por ejemplo, pasando de 60 columnas en el Logit lineal a 79 en el Logit con B-Splines), la log-verosimilitud $\ell$ siempre sube en `Train`. Para comprobar si esa subida compensa la complejidad añadida usamos:
  $$\text{AIC} = -2\ell(\hat{\boldsymbol{\beta}}) + 2k, \qquad \text{BIC} = -2\ell(\hat{\boldsymbol{\beta}}) + k\ln(N)$$
  Con $N = 425{,}567$, el **BIC** impone una penalización severísima por cada parámetro extra ($\ln(N) \approx 12.96$ puntos de deviance por cada grado de libertad). Que nuestro Logit con B-Splines reduzca el BIC en más de $1{,}300$ puntos frente al Logit lineal demuestra que la curvatura añadida tiene un respaldo empírico abrumador.

---

## 3. Modelado No Lineal de Caja Blanca: Scorecard WoE, PAVA y B-Splines Ortogonalizados

### 3.1 Algoritmo PAVA (*Pool Adjacent Violators*), Weight of Evidence ($\text{WoE}$) e Information Value ($\text{IV}$)
* **Qué es PAVA (`IsotonicRegression` — Sección 13.1):**
  Si divides una variable continua (como ingresos o ratio de deuda) en 20 cuantiles, por ruido muestral la tasa de default puede subir en el cuantil 7 y bajar ligeramente en el cuantil 8. El algoritmo **PAVA** recorre los cuantiles de izquierda a derecha y, cada vez que encuentra dos tramos adyacentes que violan la monotonía esperada, **los fusiona en un único tramo ponderado por su tamaño muestral**, repitiendo hasta obtener una escalera perfectamente monótona.
* **Weight of Evidence ($\text{WoE}$):**
  Para cada tramo $b \in \{1, \dots, B\}$, sea $D_b = \frac{\text{Malos}_b}{\text{Total Malos}}$ la proporción de defaults que caen en el tramo $b$, y $G_b = \frac{\text{Buenos}_b}{\text{Total Buenos}}$ la proporción de sanos en el tramo $b$:
  $$\text{WoE}_b = \ln\left(\frac{D_b}{G_b}\right) = \ln\left(\frac{P(y_i = 1 \mid x_i \in b)}{P(y_i = 0 \mid x_i \in b)}\right) - \ln\left(\frac{\bar{y}}{1-\bar{y}}\right)$$
  * **Propiedad mágica del WoE:** Sustituir el tramo $b$ por su $\text{WoE}_b$ transforma la variable directamente a la **escala lineal del log-odds**. Si metes una sola variable transformada en WoE dentro de una regresión logística, su pendiente teórica óptima es exactamente $\beta = 1.0$.
* **Information Value ($\text{IV}$):**
  $$\text{IV} = \sum_{b=1}^B (D_b - G_b)\,\ln\left(\frac{D_b}{G_b}\right)$$
  Matemáticamente es la **divergencia de Kullback-Leibler simétrica (divergencia de Jeffreys)** entre la distribución de la variable en los clientes morosos ($D$) y su distribución en los clientes sanos ($G$).
* **Conceptos clave para buscar:** *Pool Adjacent Violators Algorithm (PAVA)*, *Monotonic Optimal Binning*, *Weight of Evidence and Information Value*, *Jeffreys divergence in credit scoring*.

### 3.2 B-Splines Cúbicos y Ortogonalización por Proyección (Sección 14.1)
* **Qué es un B-Spline cúbico (`SplineTransformer`):**
  Dado un conjunto de nudos interiores $t_1 < t_2 < \dots < t_m$, un spline cúbico es una función formada por polinomios de grado 3 en cada tramo $[t_k, t_{k+1}]$ que se pegan en los nudos manteniendo continuas la función, su primera derivada (pendiente) y su segunda derivada (curvatura).
* **Por qué ortogonalizamos las bases en la Sección 14.1:**
  Las bases locales de De Boor $B_1(x), \dots, B_K(x)$ cumplen $\sum_k B_k(x) = 1$ y generan también la recta $x$. Si metes $[z_j, B_1(z_j), \dots, B_K(z_j)]$ en una regresión con intercepto, hay dependencia lineal exacta.
  Mediante proyección por mínimos cuadrados ordinarios en `Train`:
  1. Extraemos de la matriz de bases $B(z_j)$ toda la parte explicada por la constante y la recta $[1, z_j]$:
     $$R_j = B(z_j) - [1, z_j]\,\widehat{\Gamma}_j, \qquad \widehat{\Gamma}_j = \big([1, z_j]^\top [1, z_j]\big)^{-1} [1, z_j]^\top B(z_j)$$
  2. Aplicamos **SVD** a los residuos $R_j = U_j S_j V_j^\top$ en `Train` y tomamos las 2 direcciones principales ortonormales de curvatura $C_j = [c_{j,1}, c_{j,2}]$.
  * **Ventaja estadística:** Como $[1, z_j]^\top C_j = \mathbf{0}$ y $C_j^\top C_j = N \cdot I_2$ en `Train`, el término lineal $z_j$ y las dos componentes de curvatura $c_{j,1}, c_{j,2}$ tienen **correlación cero entre sí** ($\text{VIF} \approx 1$). Esto permite hacer el **Test de Wald de No Linealidad** contrastando únicamente si $\beta_{j,\text{curv1}} = \beta_{j,\text{curv2}} = 0$.
* **Conceptos clave para buscar:** *Restricted Cubic Splines (Harrell)*, *Natural Cubic Splines*, *Orthogonal polynomial / spline bases in GLM*, *Wald test for non-linearity*.

### 3.3 Bandas de Confianza Analíticas al 95% y *Odds Ratios* Intercuartílicos (Sección 14.3)
* **Varianza de una combinación lineal de parámetros:**
  Si la curva aprendida para la variable $j$ evaluada en el punto $x$ respecto a la mediana $x_{\text{med}}$ es $\Delta \hat{f}_j(x) = \big(\mathbf{b}_j(x) - \mathbf{b}_j(x_{\text{med}})\big)^\top \hat{\boldsymbol{\beta}}_{G_j}$, como es una combinación lineal de $\hat{\boldsymbol{\beta}}_{G_j}$, su varianza exacta es:
  $$\text{Var}\big(\Delta \hat{f}_j(x)\big) = \mathbf{d}_j(x)^\top \, \widehat{\Sigma}_{G_j G_j} \, \mathbf{d}_j(x), \qquad \text{donde } \mathbf{d}_j(x) = \mathbf{b}_j(x) - \mathbf{b}_j(x_{\text{med}})$$
  Y aplicando la función logística $\sigma(\cdot)$ a los extremos del intervalo en *log-odds* $\big[\eta(x) \pm 1.96\,\text{SE}(\Delta \hat{f}_j(x))\big]$, obtenemos una **banda de confianza al 95% exacta** que respeta siempre los límites $(0, 1)$ de la probabilidad.
* **Odds Ratio Intercuartílico ($\text{OR}_{P_{75}/P_{25}}$):**
  Evaluando $\mathbf{d}_j = \mathbf{b}_j(P_{75}) - \mathbf{b}_j(P_{25})$, el *log-odds ratio* de pasar del percentil 25 al 75 es $\mathbf{d}_j^\top \hat{\boldsymbol{\beta}}_{G_j}$ con error estándar $\sqrt{\mathbf{d}_j^\top \widehat{\Sigma}_{G_j G_j} \mathbf{d}_j}$.

---

## 4. Modelos Aditivos (GAM / EBM) y Boosting con Restricciones Monótonas

### 4.1 GAM / EBM (*Explainable Boosting Machine*) mediante Árboles 1D
* **Qué es:** Un Modelo Aditivo Generalizado (GAM) tiene la forma:
  $$\text{logit}\big(P(y = 1 \mid \mathbf{x})\big) = \beta_0 + f_1(x_1) + f_2(x_2) + \dots + f_p(x_p)$$
  En un *Gradient Boosting* normal (como XGBoost o LightGBM con `max_depth=5`), un árbol puede cortar primero por `interest_rate`, en el segundo nivel por `loan_term_months` y en el tercero por `dti`, creando interacciones de orden 3.
  Si pasamos el hiperparámetro `interaction_constraints = [[c] for c in columns]`, prohibimos que un mismo árbol utilice más de una variable: cada árbol solo puede hacer cortes sobre una única columna $x_j$. Al sumar todos los árboles que usaron $x_j$, obtenemos una función puramente univariante $f_j(x_j)$.
* **Por qué es una herramienta diagnóstica potentísima (Sección 13.2 y 13.4):**
  Te permite aislar con exactitud quirúrgica **cuánto gana el *Gradient Boosting* por hacer curvas 1D mejores que el Logit** (la diferencia entre `GAM 1D` y `Logit Lineal`) frente a **cuánto gana por cruzar variables** (la diferencia entre `XGBoost libre` y `GAM 1D`). En nuestro proyecto demostró que el `Logit con B-Splines` ya capturaba el 100% de lo que captura el `GAM 1D`.
* **Conceptos clave para buscar:** *Generalized Additive Models (Hastie & Tibshirani)*, *InterpretML Explainable Boosting Machine (Lou, Caruana, Gehrke)*, *XGBoost interaction_constraints*.

### 4.2 Restricciones de Monotonía en Gradient Boosting (`monotone_constraints`)
* **Cómo funciona internamente en XGBoost:**
  Durante la construcción de cada árbol, cuando el algoritmo evalúa partir una hoja en dos hijos (izquierdo $x_j \le s$ con peso predicho $w_L$ y derecho $x_j > s$ con peso predicho $w_R$) sobre una variable con restricción creciente (`+1`), impone la condición $w_L \le w_R$ (y además propaga las cotas mínimas y máximas hacia los nodos descendientes para que ningún corte posterior en otra variable viole la desigualdad). Si un corte candidato tuviera $w_L > w_R$, el algoritmo descarta ese corte o proyecta ambos pesos al promedio.
* **Qué aporta en banca:** Elimina las oscilaciones por sobreajuste en zonas con pocos datos y garantiza que ningún cliente vea reducida su PD estimada al empeorar un ratio financiero.
* **Conceptos clave para buscar:** *Monotonic constraints in Gradient Boosted Trees*, *Defensible Machine Learning in Credit Risk (SR 11-7)*.

---

## 5. Métricas de Discriminación y Contrastes Formales (DeLong y Bootstrap)

### 5.1 Relación Matemática entre ROC-AUC, Estadístico U de Mann-Whitney y Gini
* **Interpretación probabilística del ROC-AUC:**
  Sean $S^+ = \{\hat{p}_i : y_i = 1\}$ las predicciones de los $m$ morosos y $S^- = \{\hat{p}_j : y_j = 0\}$ las de los $n$ buenos pagadores. El área bajo la curva ROC coincide exactamente con el estadístico U de Mann-Whitney normalizado:
  $$\text{AUC} = \frac{1}{m \cdot n} \sum_{i \in S^+} \sum_{j \in S^-} \psi(\hat{p}_i, \hat{p}_j), \qquad \text{donde } \psi(a, b) = \begin{cases} 1 & \text{si } a > b \\ 0.5 & \text{si } a = b \\ 0 & \text{si } a < b \end{cases}$$
* **Coeficiente de Gini (Accuracy Ratio):** $\text{Gini} = 2\cdot\text{AUC} - 1$. Mide el cociente entre el área encerrada entre la curva CAP (*Cumulative Accuracy Profile*) del modelo y la diagonal aleatoria frente al área del modelo perfecto.

### 5.2 Test Exacto de DeLong para Comparar Dos Curvas ROC Correlacionadas (Sección 15.1)
* **El problema:** Si evalúas dos modelos $A$ y $B$ sobre el mismo conjunto de validación, $\widehat{\text{AUC}}_A$ y $\widehat{\text{AUC}}_B$ están positivamente correlacionados ($\text{Cov}(\widehat{\text{AUC}}_A, \widehat{\text{AUC}}_B) > 0$). Si usaras la fórmula de muestras independientes, sobrestimarías el error estándar de la diferencia y perderías potencia estadística.
* **Cómo funciona el método de DeLong (1988) y el algoritmo rápido $O(N \log N)$ de Sun & Xu (2014):**
  1. Para cada moroso $i \in \{1,\dots,m\}$, se define su **valor de colocación (*placement value*)** $V_{10}(i) = \frac{1}{n}\sum_{j=1}^n \psi(\hat{p}_i, \hat{p}_j)$, que es la fracción de buenos pagadores a los que ese moroso supera en score (y análogamente $V_{01}(j)$ para cada buen pagador).
  2. El algoritmo rápido de **Sun & Xu (2014)** observa que no hace falta comparar los $m \times n$ pares uno por uno ($O(N^2)$, que con $400{,}000$ filas tardaría horas), sino que $V_{10}(i)$ y $V_{01}(j)$ se obtienen instantáneamente restando el rango de la observación en la muestra total menos su rango dentro de su propia clase (**cálculo por *midranks* en $O(N \log N)$**, que tarda menos de 1 segundo).
  3. La matriz de covarianzas $2 \times 2$ entre $(\widehat{\text{AUC}}_A, \widehat{\text{AUC}}_B)$ viene dada por:
     $$S = \frac{1}{m}\text{Cov}\big(V_{10}^{(A)}, V_{10}^{(B)}\big) + \frac{1}{n}\text{Cov}\big(V_{01}^{(A)}, V_{01}^{(B)}\big)$$
     de donde se obtiene el error estándar exacto de $\Delta\text{AUC} = \widehat{\text{AUC}}_B - \widehat{\text{AUC}}_A$:
     $$\text{SE}(\Delta\text{AUC}) = \sqrt{S_{11} + S_{22} - 2 S_{12}}, \qquad z_{\text{DeLong}} = \frac{\Delta\text{AUC}}{\text{SE}(\Delta\text{AUC})} \sim \mathcal{N}(0, 1)$$
* **Conceptos clave para buscar:** *DeLong test for correlated ROC curves (1988)*, *Sun and Xu (2014) Fast Implementation of DeLong's Algorithm*, *Mann-Whitney U-statistic structural components (Sen 1960)*.

### 5.3 Estadístico KS (*Kolmogorov-Smirnov*) y Bootstrap Estratificado Pareado
* **Qué mide el KS:**
  $$\text{KS} = \sup_{t \in [0, 1]} \big| F_0(t) - F_1(t) \big|$$
  donde $F_0(t) = P(\hat{p} \le t \mid y = 0)$ es el porcentaje acumulado de buenos pagadores por debajo del umbral $t$ y $F_1(t) = P(\hat{p} \le t \mid y = 1)$ es el de morosos.
* **Por qué usamos Bootstrap Pareado Estratificado (Sección 15.1):**
  Para calcular el intervalo de confianza de $\Delta\text{KS} = \text{KS}_B - \text{KS}_A$, en cada réplica $b = 1,\dots,B$ extraemos con reemplazamiento $m$ índices de la clase `1` y $n$ índices de la clase `0`, y evaluamos **ambos modelos sobre exactamente la misma muestra remuestreada**. Eso preserva la correlación pareada entre modelos y evita sesgos en la proporción de impagos.

---

## 6. Calibración de Probabilidades, Estabilidad (PSI) y Expected Loss ($\text{EL}$)

### 6.1 Brier Score y Brier Skill Score (Mejora sobre Constante)
* **Descomposición de Murphy (1973) del Brier Score:**
  $$\text{Brier} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{p}_i)^2 = \underbrace{\bar{y}(1 - \bar{y})}_{\text{Incertidumbre (Constante)}} + \underbrace{\text{Rel}}_{\text{Fallo de Calibración}} - \underbrace{\text{Res}}_{\text{Resolución / Discriminación}}$$
  * Es una **regla de puntuación estrictamente propia (*strictly proper scoring rule*)**: solo alcanza su mínimo esperado cuando el modelo predice la verdadera probabilidad condicionada $\hat{p}_i = P(y_i = 1 \mid \mathbf{x}_i)$.
  * El **Brier Skill Score** ($\text{BSS} = 1 - \frac{\text{Brier}}{\text{Brier}_{\text{const}}}$) elimina el término de incertidumbre base $\bar{y}(1-\bar{y})$ para medir en porcentaje cuánto error cuadrático reduce el modelo respecto a predecir siempre la media.
* **Conceptos clave para buscar:** *Strictly proper scoring rules (Gneiting & Raftery)*, *Murphy decomposition of Brier Score (Reliability, Resolution, Uncertainty)*, *Brier Skill Score*.

### 6.2 Por qué un Desplazamiento Constante en *Log-Odds* produce un Ratio `Obs / Pred` Variable entre Deciles (Sección 15.3–15.4)
* **El fenómeno que observamos en el cuaderno:**
  Cuando la tasa base de impago cae entre el periodo de entrenamiento ($\bar{\pi}_{\text{Train}} \approx 16.7\%$) y el de validación ($\bar{\pi}_{\text{Val}} \approx 10.4\%$) mantendo la misma relación con las variables $X$, el teorema de Bayes (corrección de *prior* de King & Zeng, 2001) establece que el verdadero *log-odds* en la nueva cohorte se desplaza en una constante $\delta_0$:
  $$\text{logit}(p_{\text{Val}}(\mathbf{x})) = \text{logit}(p_{\text{Train}}(\mathbf{x})) + \delta_0, \qquad \delta_0 = \ln\left(\frac{\bar{\pi}_{\text{Val}}}{1-\bar{\pi}_{\text{Val}}}\right) - \ln\left(\frac{\bar{\pi}_{\text{Train}}}{1-\bar{\pi}_{\text{Train}}}\right) < 0$$
  Al pasar de escala *log-odds* a escala de probabilidades mediante $\sigma(\eta) = \frac{1}{1+e^{-\eta}}$, el cociente entre la probabilidad real en validación y la predicha por el modelo de entrenamiento **no es constante**, sino que depende del nivel de riesgo $\hat{p}_{\text{Train}}$:
  $$\frac{p_{\text{Val}}(\mathbf{x})}{\hat{p}_{\text{Train}}(\mathbf{x})} = \frac{e^{\delta_0}}{(1 - \hat{p}_{\text{Train}}(\mathbf{x})) + \hat{p}_{\text{Train}}(\mathbf{x})\,e^{\delta_0}}$$
  * En el **Decil 1** ($\hat{p}_{\text{Train}} \to 0$), el denominador vale $\approx 1$, por lo que la razón cae hasta $e^{\delta_0}$ (el mínimo).
  * En el **Decil 10** ($\hat{p}_{\text{Train}}$ alto), el denominador es menor que $1$ (porque $e^{\delta_0} < 1$), por lo que la razón `Obs / Pred` sube acercándose a $1$.
  * **Por qué esto es importantísimo:** Explica matemáticamente por qué la razón `observado / predicho` variaba entre el Decil 1 y el Decil 10, y por qué al aplicar el ajuste en escala *log-odds* ($\delta_0 \approx -0.47$, o **Escalamiento Logístico de Platt** $\delta_0 + \delta_1\text{logit}(\hat{p})$ con $\delta_1 = 1.00$) quedan calibrados simultáneamente todos los deciles de la cartera.
* **Conceptos clave para buscar:** *Prior shift / Label shift in logistic regression (King & Zeng 2001, Saerens et al. 2002)*, *Platt scaling in log-odds*, *Logistic recalibration (Cox 1958: calibration intercept and calibration slope)*.

### 6.3 PSI (*Population Stability Index*) y Monitoreo en Producción
* **Fórmula:** $\text{PSI} = \sum_{d=1}^{10} (p_d^{\text{actual}} - p_d^{\text{esperado}}) \ln\left(\frac{p_d^{\text{actual}}}{p_d^{\text{esperado}}}\right)$.
* **Distinción entre *Covariate Shift* y *Concept / Prior Shift*:**
  * El PSI mide si ha cambiado la distribución de las puntuaciones $P(\hat{p}(X))$ (provocado por un cambio en el perfil de los solicitantes entrantes $P(X)$, llamado *covariate shift*).
  * El PSI **no mide** si ha cambiado $P(Y=1 \mid X)$ (*concept drift* o *prior shift*). Por eso en monitorización bancaria el PSI se calcula todos los meses (temprano, sin esperar a ver impagos), y se complementa trimestral/anualmente con tests de calibración (Hosmer-Lemeshow / Spiegelhalter / Brier / intercepto-pendiente de Cox) una vez maduran las cohortes.

### 6.4 Integración en Expected Loss ($\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$)
* **Por qué la calibración por tramos es el puente hacia $\text{EL}$:**
  Para una cartera de $N$ préstamos, la provisión total por pérdida esperada es:
  $$\text{EL}_{\text{total}} = \sum_{i=1}^N \widehat{\text{PD}}_i \times \widehat{\text{LGD}}_i \times \widehat{\text{EAD}}_i$$
  Si el importe del préstamo ($\text{EAD}_i$) está correlacionado con el decil de riesgo (como ocurre en nuestra cartera, donde los préstamos más grandes tienden a concentrarse en deciles superiores), no basta con que la $\text{PD}$ media global sea correcta: **es indispensable que la $\text{PD}$ esté bien calibrada decil a decil**, pues un sesgo en los deciles 8–10 (donde los importes $\text{EAD}_i$ son mayores) distorsionaría desproporcionadamente la pérdida esperada en dólares.

---

## 7. Bibliografía y Referencias Clave para Profundizar

1. **Modelado Regresivo, Splines, Tests de Wald por Bloques y Calibración:**
   * **Frank E. Harrell Jr.** — *Regression Modeling Strategies: With Applications to Linear Models, Logistic and Ordinal Regression, and Survival Analysis* (2nd Ed., Springer, 2015).
     * *Qué leer:* Capítulo 2 (*Continuous Predictors — Splines & Non-linearity tests*), Capítulo 4 (*Multivariable Modeling Strategies — Block Wald tests & Backward stepdown*), Capítulo 10 (*Binary Logistic Regression*).
2. **Credit Scoring Bancario, WoE, IV, PSI y Scorecards:**
   * **Naeem Siddiqi** — *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring* (Wiley, 2006 / *Intelligent Credit Scoring*, 2017).
     * *Qué leer:* Construcción de tramos monótonos, escalas de IV, transformación a puntos y umbrales de PSI.
   * **Thomas, L. C., Edelman, D. B., & Crook, J. N.** — *Credit Scoring and Its Applications* (SIAM, 2nd Ed., 2017).
     * *Qué leer:* *Reject inference*, ventanas fijas de observación, Basilea IRB e IFRS 9.
3. **Test de DeLong para Curvas ROC:**
   * **DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988).** *"Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach."* Biometrics, 44(3), 837–845.
   * **Sun, X., & Xu, W. (2014).** *"Fast Implementation of DeLong's Algorithm for Comparing the Areas Under Correlated Receiver Operating Characteristic Curves."* IEEE Signal Processing Letters, 21(11), 1389–1393.
4. **Recalibración Logística y Corrección por Cambio de Tasa Base (*Prior Shift*):**
   * **Cox, D. R. (1958).** *"Two further applications of a model for binary regression."* Biometrika (define el intercepto y la pendiente de calibración logística).
   * **King, G., & Zeng, L. (2001).** *"Logistic Regression in Rare Events Data."* Political Analysis, 9(2), 137–163 (demuestra la fórmula exacta de corrección del intercepto en *log-odds* cuando cambia la proporción marginal de unos).
   * **Steyerberg, E. W. (2019).** *Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating* (2nd Ed., Springer — excelente tratamiento de Brier Score, recalibración en *log-odds* y validación temporal).
5. **GAM / EBM y Monotonía en Boosting:**
   * **Lou, Y., Caruana, R., & Gehrke, J. (2012).** *"Intelligible Models for Classification and Regression."* KDD.
   * **Hastie, T., Tibshirani, R., & Friedman, J.** — *The Elements of Statistical Learning* (2nd Ed., Springer — Capítulo 5 sobre Splines y Capítulo 9 sobre GAMs).
