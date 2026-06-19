# Predicción de Ventas de Alto Valor en E-commerce mediante Machine Learning

## 1. Definición del Problema

El comercio electrónico se desarrolla sobre una infraestructura de datos que registra de forma continua cada interacción de intercambio entre un cliente y la plataforma. Esta capacidad de captura masiva de información incluye el tipo de producto adquirido, la categoría, la cantidad, el precio unitario, la ubicación geográfica y el momento exacto de la compra que representa una ventaja competitiva cuando es procesada de manera inteligente.

Sin embargo, la mayoría de las organizaciones aún gestionan este volumen de datos de forma deficiente, limitándose al reporte histórico en lugar de extraer valor de la información disponible.

En este contexto, la aplicación de técnicas de Machine Learning sobre registros de intercambio permite transformar datos operacionales en conocimiento para futuras tomas de decisiones respaldadas en evidencia cuantitativa. 

La clasificación de las transacciones según su valor económico forman un caso de uso tipico de esta capacidad, haciendo referencia a que no todas las ventas generan el mismo impacto sobre el ingreso de la empresa y distinguir de forma automática entre transacciones de alto y bajo valor económico abre una posibilidad enorme para la optimización operacional, la personalización del servicio y la asignación eficiente de recursos.

Tomando en consideración la gestión de ingresos, la distribución del revenue en plataformas de e-commerce tiende a ser desproporcionada debido a un valor bajo de las transacciones concentra una proporción desequilibrada del ingreso total.


Identificar estos intercambios de alto valor no es únicamente un ejercicio analítico, sino un apoyo operacional directo. Una empresa que conoce en tiempo real cuáles son sus transacciones que pertenecen a este segmento, puede actuar sobre ellas para priorizar su procesamiento logístico y asignar equipos especializados de soporte posventa, aplicar controles de calidad y diseñar estrategias de fidelización orientadas a los clientes que generan mayor rentabilidad.

Desde una perspectiva analítica, permite explorar las relaciones entre variables transaccionales como por ejemplo, categoría de producto, precio unitario, cantidad adquirida, ciudad de compra y temporalidad ademas del valor económico generado.

Este análisis ayuda a que los negocio vayan más allá de la predicción, permitiendo analizar qué productos, mercados y períodos estacionales concentran el mayor potencial de revenue, pudiendo así tomar decisiones de expansión de catálogo, efoque del marketing y planificación de inventarios.

La relevancia del problema se sostiene también en su viabilidad técnica, ya que todas las variables necesarias para construir el modelo predictivo están disponibles en el momento de la transacción, lo que hace que el sistema sea posible implementarlo en entornos de producción.

### 1.2 Problema a Resolver

El problema principal se basa en un dilema para poder predecir las ventas en vase a un modelo de aprendizaje automatico, a partir de las características en cada registro transaccional, se busca predecir si una venta corresponderá a una venta de alto valor económico o no, en base a lo anterior, se construirá una variable objetivo denominada "High_Value_Sale", proveneite del revenue de cada transacción:

```
Revenue = Quantity × Price
```

Las transacciones cuyo revenue supere la mediana del conjunto de datos completo serán etiquetadas como clase positiva "High_Value_Sale = 1", el resto como clase negativa "High_Value_Sale = 0". El uso de la mediana nos demuestra una partición equilibrada lo que simplifica el proceso al reducir la necesidad de técnicas de rebalanceo de clases en caso de que se encuentre un sesgo dentro de los datos debido a la diferencia de los datos.

### 1.3 Objetivo General

Desarrollar, entrenar y evaluar comparativamente tres modelos de clasificación supervisada como la regresión logística, árbol de decisión y random forest para la predicción de ventas de alto valor en un dataset de e-commerce, identificando el modelo que demuestre el mejor equilibrio segun su capacidad de predicción, estabilidad ante el sobreajuste y la interpretación de los datos de resultados.

### 1.4 Objetivos Específicos

1. Realizar un análisis de datos que permita caracterizar la distribución de las variables, identificando patrones relevantes, junto con las relaciones entre los predictores disponibles y la variable objetivo.

2. Construir la variable objetivo "High_Value_Sale", empleando la mediana del revenue como umbral de clasificación y verificando el balance resultante de clases.

3. Seleccionar las características más relevantes mediante técnicas estadísticas y métricas.

4. Entrenar los tres modelos seleccionados bajo una estrategia controlada de partición Train/Test, controlando el sobreajuste mediante la regulación de hiperparámetros y validación cruzada.

5. Evaluar el desempeño de cada modelo mediante un conjunto comprehensivo de métricas como accuracy, precision, recall, F1-Score, ROC-AUC y matriz de confusión y comparar sus resultados para fundamentar la selección del modelo.

## 2. Plan de Acción

### 2.1 Descripción del Dataset

El dataset utilizado corresponde a un registro transaccional de ventas de una plataforma de comercio electrónico compuesto por 1.000 registros, cada uno representando una orden de compra única e irrepetible. La información abarca operaciones realizadas en 10 ciudades que son Ahmedabad, Bangalore, Chennai, Delhi, Hyderabad, Jaipur, Kolkata, Mumbai, Pune y Surat y cubre 14 tipos de productos agrupados en 5 categorías: Electronics, Accessories, Home Appliances, Fashion y Books.

Las variables disponibles en el dataset son las siguientes:

**Order_ID**: Variable categórica nominal, funciana como identificador único de la orden.

**Product**: Variable categórica nominal, equivale a los Artículo vendido.

**Category**: Variable categórica nominal, realiza la agrupación de productos en 5 categorías comerciales.

**Quantity**: Variable numérica discreta, hace referencia a las unidades adquiridas por orden y su rango va de 1 a 5.

**Price**: Variable numérica continua, equivale al precio unitario del producto en unidades monetarias.

**City**: Variable categórica nominal, hace referencia a las ciudad donde se realizó la compra.

**Date**: Variable fecha  Fecha de la transacción. 

La alta variabilidad del precio unitario es el indicador más relevante del dataset, y constituye el principal valor predictivo esperado para la variable objetivo. La combinación de variables categóricas (Product, Category, City) con variables numéricas de alta dispersión (Price) y una dimensión temporal (Date),  esto demuestra que el dataset  contiene un potencial considerable para el entrenamiento de modelos de clasificación supervisada.

Desde el punto de vista de la calidad de los datos, el dataset no presenta valores nulos en ninguna de sus variables, lo que simplifica la etapa de preprocesamiento.

### 2.2 Metodología Propuesta

#### Exploración Inicial de Datos

Se procederá a la carga del dataset, inspección de tipos de datos, verificación de valores nulos y duplicados ademas de la generación de estadísticas descriptivas básicas para todas las variables, en este apartado tiene por objetivo detectar anomalías estructurales en los datos antes de comenzar el análisis.

#### Análisis Exploratorio de Datos (EDA)

Se realizará un análisis distribucional de las variables numéricas mediante histogramas y diagramas de caja, y un análisis de frecuencias para las variables categóricas, en conjunto se explorarán las distribuciones del revenue por categoría, producto y ciudad, y se analizarán patrones estacionales derivados de la variable "Date", El EDA permitirá identificar relaciones entre los predictores y la variable objetivo, así como detectar valores atípicos que puedan afectar el entrenamiento.

#### Limpieza y Preparación de Datos

Se verificará la consistencia de los registros, se tratarán los valores atípicos según el criterio del EDA, y se asegurará la coherencia de tipos de datos para cada columna. Se excluirá la variable "Order_ID" por falta de un valor predictivo.

#### Feature Engineering

Se extraerán características temporales de la variable "Date": mes (Month), día de la semana (DayOfWeek), trimestre (Quarter) e indicador de fin de semana (Is_Weekend). Se aplicará una transformación logarítmica sobre "Price" mediante "Log_Price = log(1 + Price)" para reducir la asimetría positiva de la distribución. Se construirá una variable de frecuencia para "Product" (Product_Freq) que capture la popularidad relativa de cada artículo en el conjunto de datos.

#### Construcción de la Variable Objetivo

Se calculará "Revenue = Quantity × Price" para cada registro. Se determinará la mediana del revenue sobre el conjunto completo de 1.000 observaciones y se etiquetará cada registro: "High_Value_Sale = 1" si "Revenue > mediana", "0" en caso contrario. Se verificará el balance de clases resultante.

#### Transformación de Variables Categóricas

Se aplicará "One-Hot Encoding" sobre las variables "Category" y "City", dado que son variables nominales sin orden y con cardinalidad moderada. Para la variable "Product", se utilizará la codificación por frecuencia construida en la etapa de Feature Engineering.

#### División Train/Test

El dataset se dividirá en un conjunto de entrenamiento y un conjunto de prueba, aplicando estratificación sobre la variable objetivo para garantizar que la proporción de clases se preserve en ambos subconjuntos. Se fijará una semilla aleatoria para asegurar la reproducibilidad del experimento.

#### Entrenamiento de Modelos

Se entrenarán los tres modelos seleccionados sobre el conjunto de entrenamiento. Para la Regresión Logística, se aplicará previamente un escalado StandardScaler, dado que este modelo es sensible a la escala de los predictores. Los hiperparámetros de cada modelo serán ajustados con criterio técnico para controlar el sobreajuste, aplicando validación cruzada donde corresponda.

#### Evaluación de Desempeño

Se calcularán todas las métricas de evaluación definidas sobre el conjunto de prueba para cada modelo, ademas se generarán las matrices de confusión correspondientes y las curvas ROC.

#### Comparación de Resultados

Se construirá una tabla comparativa de métricas entre los tres modelos y se visualizarán las curvas ROC en un mismo gráfico para facilitar la comparación en conjunto, se analizará la importancia de variables del Random Forest y se documentarán las conclusiones del proceso.

### 2.3 Estrategia de Evaluación

La evaluación del desempeño de los modelos se realizará mediante las siguientes métricas, cada una aportando una dimensión de análisis diferente:

**Accuracy**: Proporción de predicciones correctas sobre el total de instancias,lo que permite ver la información en datasets balanceados como el presente, pero insuficiente por sí sola para capturar el comportamiento diferencial de los errores.

**Precision**: De todas las transacciones que el modelo clasifica como alto valor, qué fracción lo es efectivamente, esta métrica implica que el modelo genera pocos falsos positivos que son relevante cuando el costo de clasificar erróneamente una venta normal como de alto valor es significativo.

**Recall**: De todas las transacciones que realmente son de alto valor se debe verificar qué fracción del modelo identifica correctamente, considerando que un elevado Recall implica que el modelo no omite ventas de alto valor, lo que resulta relevante cuando el costo de no detectar una venta de alto valor es alto.

**F1-Score**: Media armónica entre Precision y Recall, a través de esta metrica podemos analizar que ofrece un balance equilibrado entre ambas métricas y constituye la métrica principal de comparación entre modelos, especialmente útil cuando existe tensión entre Precision y Recall.

**ROC-AUC**: Área bajo la curva ROC, mide la capacidad discriminativa global del modelo a través de todos los posibles umbrales de clasificación. Un valor de 1.0 indica discriminación perfecta, demostrando que un 0.5 equivale a un clasificador aleatorio, esta es la métrica más robusta para comparar la calidad de los modelos independientemente del umbral elegido.

**Matriz de Confusión**: Desglosa las predicciones en verdaderos positivos (TP), verdaderos negativos (TN), falsos positivos (FP) y falsos negativos (FN), permitiendo un diagnóstico detallado del tipo de errores que comete cada modelo y orientando ajustes en el umbral de clasificación según los costos del negocio.

**Detección de Overfitting**: Se comparará sistemáticamente el desempeño de cada modelo sobre el conjunto de entrenamiento versus el conjunto de prueba, el analisis de esta matrica dice que una brecha significativa entre ambos será interpretada como evidencia de sobreajuste. Para controlarlo, se regularán los hiperparámetros relevantes de cada modelo y se complementará con validación cruzada estratificada de 5 particiones.

## 3. Justificación del Modelo

La selección de los tres modelos responde principalmete a una planificación que cubre una serie de complejidades desde el modelado lineal hasta los métodos de ensemble, logrando permitir una comparación detallada valiosa en base a los datos para demostrar la eficiencia en cada uno de los modelos al momento de realizar las prediccione, permitiento analizar rigurosamente futuras tomas de dicisiones en cuanto a la ventas de los productos.

### 3.1 Regresión Logística

#### Funcionamiento

La Regresión Logística modela la probabilidad de pertenencia a la clase positiva mediante la función sigmoide aplicada sobre una combinación lineal ponderada de los predictores:

$$P(Y=1 \mid X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \cdots + \beta_n X_n)}}$$

Los coeficientes $\beta$ son estimados mediante la maximización de la verosimilitud del modelo sobre los datos de entrenamiento. La predicción de clase se obtiene comparando la probabilidad estimada contra un umbral en el cual se desarrolle típicamente 0.5. El modelo incorpora regularización L2 (Ridge) controlada por el hiperparámetro "C", que penaliza la magnitud de los coeficientes para prevenir el sobreajuste.

Su principal fortaleza reside en la interpretabilidad directa de los coeficientes, cada $\beta_i$ cuantifica el cambio en el log-odds de la variable objetivo ante un incremento unitario en el predictor correspondiente, facilitando la comunicación de resultados. Computacionalmente es muy eficiente, produce estimaciones de probabilidad calibradas y su regularización  ofrece control del sobreajuste.

El modelo asume que la relación entre los predictores y la variable objetivo es lineal en el espacio logit, esta restricción limita su capacidad para capturar interacciones no lineales entre variables como la combinación entre categoría de producto y rango de precio, en presencia de relaciones complejas, su poder predictivo puede resultar inferior al de métodos basados en árboles.

En el contexto de la predicción de ventas de alto valor, existe una relación esencialmente lineal entre el precio unitario y la probabilidad de superar el umbral de revenue, tomando en cuenta que a mayor precio, mayor revenue esperado. Esta relación dominante hace que un modelo lineal capture una fracción significativa de la varianza explicable, posicionándolo competitivo y técnicamente sólido. Adicionalmente, los coeficientes estimados proveerán evidencia cuantitativa sobre la influencia relativa de cada variable transaccional sobre la probabilidad de una venta de alto valor.

### 3.2 Decision Tree

#### Funcionamiento

El Árbol de Decisión construye un modelo de clasificación mediante la partición recursiva del espacio de características en regiones cada vez más homogéneas respecto a la variable objetivo. En cada nodo interno, el algoritmo selecciona la variable y el umbral de corte que maximizan la reducción de impureza, el proceso continúa hasta alcanzar un criterio de parada definido, El árbol resultante puede representarse como una cadena de reglas "IF-THEN" que documentan el proceso de decisión aprendido.

Su principal activo es la interpretabilidad dado la visualización del árbol permite entender de forma intuitiva qué variables y qué umbrales determinan la clasificación de cada instancia, facilitando la validación por parte de expertos del dominio. No requiere normalización de variables ni codificación especial en muchas implementaciones, ademas de que captura de forma nativa relaciones no lineales e interacciones entre variables, y es robusto frente a valores atípicos en los predictores, dado que las particiones binarias no se ven afectadas por valores extremos de la misma forma que los modelos lineales.

Sin restricciones adecuadas a su crecimiento, el árbol tiende fuertemente al sobreajuste, ya que puede memorizar el conjunto de entrenamiento generando fronteras de decisión excesivamente complejas que no generalizan ante datos nuevos. Los árboles individuales presentan además alta varianza debido a pequeñas perturbaciones en los datos de entrenamiento pueden producir estructuras de árbol significativamente distintas. Esta inestabilidad es la principal motivación para los métodos construidos sobre árboles.

Las relaciones que determinan si una venta es de alto valor son naturalmente expresables en términos de reglas con umbrales, un producto de la categoría Electronics con precio unitario superior a cierto valor tiene alta probabilidad de generar un revenue que supere la mediana, La estructura permite descubrir estas reglas de forma automática a partir de los datos. La visualización del árbol resultante constituye además un hallazgo analítico de valor independiente, al explicitar las condiciones que el modelo considera determinantes para clasificar una venta como de alto valor.

### 3.3 Random Forest

#### Funcionamiento

Random Forest implementa el principio mediante la técnica de "bagging" aplicada sobre Árboles de Decisión con selección aleatoria de características. Para cada uno de los $B$ árboles del ensemble, el algoritmo extrae una submuestra aleatoria con reemplazo del conjunto de entrenamiento y en cada nodo de cada árbol, considera únicamente un subconjunto aleatorio de $\sqrt{p}$ características como candidatas para la partición, donde $p$ es el número total de predictores. La predicción final se obtiene por votación mayoritaria entre los $B$ árboles. Este doble mecanismo reduce drásticamente la correlación entre los árboles individuales, disminuyendo la varianza del ensemble respecto a cualquier árbol aislado y mejorando significativamente la capacidad de generalización.

Random Forest ofrece una medida de importancia de variables basada en la reducción media de impureza, que resulta valiosa tanto para la interpretación del modelo como para la selección de características. Su mecanismo de out-of-bag error proporciona una estimación interna del error de generalización sin necesidad de un conjunto de validación separado, ademas se puede decir que es robusto frente a valores atípicos, variables irrelevantes y diferencias de escala entre predictores.

La combinación de múltiples árboles produce un modelo de mayor opacidad respecto a los anteriores, debido a que las predicciones son difíciles de explicar a nivel de instancia individual sin recurrir a herramientas adicionales de interpretabilidad, el ajuste de sus hiperparámetros requiere más esfuerzo computacional que los modelos anteriores, aunque esta limitación es poco relevante para el tamaño del dataset del proyecto.

La predicción de ventas de alto valor involucra interacciones multidimensionales entre variables que un modelo lineal no puede capturar sin especificación explícita, ya que la combinación entre categoría, producto, precio y cantidad genera fronteras de decisión no lineales. Adicionalmente, la importancia de variables que genera el ensemble permitirá identificar qué características son realmente determinantes para discriminar ventas de alto valor, favoreciendo el análisis con evidencia basada en el modelo entrenado.
