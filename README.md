# Clasificación de Categorías de Productos en E-commerce mediante Machine Learning

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

Las plataformas de comercio electrónico registran diariamente miles de transacciones que contienen información sobre productos, cantidades compradas, precios, ubicación geográfica y fecha de compra, pero además en otros contextos, la categoría de un producto puede no encontrarse disponible de manera explícita o puede requerir validación para mejorar procesos de clasificación, recomendación y análisis comercial.

Debido a lo anterior, el problema consiste en desarrollar un modelo de Machine Learning capaz de predecir la categoría a la que pertenece un producto utilizando información transaccional disponible en cada orden de compra.

La variable objetivo del proyecto será "Category", la cual contiene clases como "Electronics", "Accessories", "Home Appliances", "Fashion", "Books".

Por lo tanto, el problema corresponde a una tarea de clasificación supervisada multiclase.

### 1.3 Objetivo General

Desarrollar, entrenar y evaluar comparativamente modelos de clasificación supervisada capaces de predecir la categoría de un producto dentro de una plataforma de comercio electrónico, identificando el algoritmo que presente el mejor desempeño predictivo y capacidad de generalización.

### 1.4 Objetivos Específicos

1. Realizar un análisis exploratorio de los datos para identificar patrones y relaciones entre las variables disponibles.

2. Preparar y transformar los datos mediante técnicas de preprocesamiento y Feature Engineering.

3. Seleccionar las variables más relevantes para la predicción de la categoría del producto.

4. Entrenar modelos de clasificación supervisada utilizando Regresión Logística, Árboles de Decisión y Random Forest.

5. Evaluar y comparar el desempeño de los modelos mediante métricas de clasificación multiclase.

6. Identificar el modelo que mejor represente para la clasificación automática de categorías de productos.

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

Se extraerán características temporales de la variable "Date": mes (Month), día de la semana (DayOfWeek), trimestre (Quarter) e indicador de fin de semana (Is_Weekend). Se aplicará una transformación logarítmica sobre "Price" mediante "Log_Price = log(1 + Price)" para reducir la asimetría positiva de la distribución. También se construirá una variable de frecuencia para "Product" denominada "Product_Freq", con el objetivo de capturar la aprobación de cada producto dentro del conjunto de datos.

#### Definición de la Variable Objetivo

La variable objetivo utilizada será "Category", la cual representa la categoría comercial a la que pertenece cada producto. Esta variable contiene cinco clases distintas ya antes mencionadas y será utilizada como objetivo de clasificación supervisada multiclase.

#### Transformación de Variables Categóricas

Se aplicará One-Hot Encoding sobre las variables categóricas relevantes para convertirlas en representaciones numéricas que puedan ser posibles y compatibles con los algoritmos de Machine Learning.

Dependiendo del análisis exploratorio, se evaluará el tratamiento más adecuado para la variable Product, considerando codificación por frecuencia o exclusión en caso de representar directamente la categoría objetivo.

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

**Accuracy**: Proporción de predicciones correctas sobre el total de instancias,lo que permite ver la información en datasets balanceados como el presente, pero insuficiente por sí sola para capturar el comportamiento de los errores.

**Precision**: De todas las transacciones que el modelo clasifica como categorías de productos, qué fracción lo es efectivamente, esta métrica implica que el modelo genera pocos falsos positivos que son relevante cuando el costo de clasificar de forma deficiente una venta normal como de alto valor es significativo.

**Recall**: De todas las transacciones que realmente son categorías de productos se debe verificar qué fracción del modelo identifica correctamente, considerando que un elevado Recall implica que el modelo no omite categorías de productos, lo que resulta relevante cuando el costo de no detectar una venta de alto valor es alto.

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

Los coeficientes $\beta$ son estimados mediante la maximización de la verosimilitud del modelo sobre los datos de entrenamiento. La predicción de clase se obtiene comparando la probabilidad estimada contra un umbral en el cual se desarrolla típicamente en 0.5. El modelo incorpora regularización L2 (Ridge) controlada por el hiperparámetro "C", que penaliza la magnitud de los coeficientes para prevenir el sobreajuste.

Su principal fortaleza reside en la interpretabilidad directa de los coeficientes, cada $\beta_i$ cuantifica el cambio en el log-odds de la variable objetivo ante un incremento en el predictor correspondiente, facilitando la comunicación de resultados. Computacionalmente es muy eficiente, ya que produce estimaciones de probabilidad calibradas y su regularización  ofrece control del sobreajuste.

El modelo asume que la relación entre los predictores y la variable objetivo es lineal en el espacio logit, esta restricción limita su capacidad para capturar interacciones no lineales entre variables como la combinación entre categoría de producto y rango de precio, en presencia de relaciones complejas, su poder predictivo puede resultar inferior al de métodos basados en árboles.

En el contexto de la clasificación de categorías de productos, existe una relación esencialmente lineal entre el precio unitario y la probabilidad de superar el umbral de revenue, tomando en cuenta que a mayor precio, mayor revenue esperado. Esta relación dominante hace que un modelo lineal capture una fracción significativa de la varianza explicable, posicionándolo competitivo y técnicamente sólido. Adicionalmente, los coeficientes estimados proveerán evidencia cuantitativa sobre la influencia relativa de cada variable transaccional sobre la probabilidad de una clasificación de categorías de productos.

### 3.2 Decision Tree

#### Funcionamiento

El Árbol de Decisión construye un modelo de clasificación mediante la partición recursiva del espacio de características en regiones cada vez más homogéneas respecto a la variable objetivo. En cada nodo interno, el algoritmo selecciona la variable y el umbral de corte que maximizan la reducción de impureza, el proceso continúa hasta alcanzar un criterio de parada definido, El árbol resultante puede representarse como una cadena de reglas "IF-THEN" que documentan el proceso de decisión aprendido.

Su principal activo es la interpretabilidad dado la visualización del árbol permite entender de forma intuitiva qué variables y qué umbrales determinan la clasificación de cada instancia, facilitando la validación por parte de expertos del dominio. No requiere normalización de variables ni codificación especial en muchas implementaciones, ademas de que captura de forma nativa relaciones no lineales e interacciones entre variables, y es robusto frente a valores atípicos en los predictores, dado que las particiones binarias no se ven afectadas por valores extremos de la misma forma que los modelos lineales.

Sin restricciones adecuadas a su crecimiento, el árbol tiende fuertemente al sobreajuste, ya que puede memorizar el conjunto de entrenamiento generando fronteras de decisión excesivamente complejas que no generalizan ante datos nuevos. Los árboles individuales presentan además alta varianza debido a pequeñas perturbaciones en los datos de entrenamiento pueden producir estructuras de árbol significativamente distintas. Esta inestabilidad es la principal motivación para los métodos construidos sobre árboles.

Las relaciones que determinan si una clasificación de categorías de productos son naturalmente expresables en términos de reglas con umbrales, un producto de la categoría Electronics con precio unitario superior a cierto valor tiene alta probabilidad de generar un revenue que supere la mediana, La estructura permite descubrir estas reglas de forma automática a partir de los datos. La visualización del árbol resultante constituye además un hallazgo analítico de valor independiente, al explicitar las condiciones que el modelo considera determinantes para clasificar una clasificación de categorías de productos.

### 3.3 Random Forest

#### Funcionamiento

Random Forest implementa el principio mediante la técnica de "bagging" aplicada sobre Árboles de Decisión con selección aleatoria de características. Para cada uno de los $B$ árboles del ensemble, el algoritmo extrae una submuestra aleatoria con reemplazo del conjunto de entrenamiento y en cada nodo de cada árbol, considera únicamente un subconjunto aleatorio de $\sqrt{p}$ características como candidatas para la partición, donde $p$ es el número total de predictores. La predicción final se obtiene por votación mayoritaria entre los $B$ árboles. Este doble mecanismo reduce drásticamente la correlación entre los árboles individuales, disminuyendo la varianza del ensemble respecto a cualquier árbol aislado y mejorando significativamente la capacidad de generalización.

Random Forest ofrece una medida de importancia de variables basada en la reducción media de impureza, que resulta valiosa tanto para la interpretación del modelo como para la selección de características. Su mecanismo de out-of-bag error proporciona una estimación interna del error de generalización sin necesidad de un conjunto de validación separado, ademas se puede decir que es robusto frente a valores atípicos, variables irrelevantes y diferencias de escala entre predictores.

La combinación de múltiples árboles produce un modelo de mayor opacidad respecto a los anteriores, debido a que las clasificaciones son difíciles de explicar a nivel de instancia individual sin recurrir a herramientas adicionales de interpretabilidad, el ajuste de sus hiperparámetros requiere más esfuerzo computacional que los modelos anteriores, aunque esta limitación es poco relevante para el tamaño del dataset del proyecto.

La clasificación de categorías de productos involucra interacciones multidimensionales entre variables que un modelo lineal no puede capturar sin especificación explícita, ya que la combinación entre categoría, producto, precio y cantidad genera fronteras de decisión no lineales. Adicionalmente, la importancia de variables que genera el ensemble permitirá identificar qué características son realmente determinantes para discriminar ventas de alto valor, favoreciendo el análisis con evidencia basada en el modelo entrenado.

### 4. Implementación

La implementación completa en base al dataset mensionado anteriormente con los 1.000 datos registrados.

Se cargó el dataset y se convirtió la columna Date a formato datetime. Las estadísticas descriptivas confirman que Quantity varía entre 1 y 5 unidades con una media de 3.04 y que Price presenta una dispersión considerable, entre 407.43 y 60.347,51, con una media de 8.940,49 y una desviación estándar de 14.229,46. 

Esta diferencia entre media y desviación estándar evidencia una distribución de precios fuertemente asimétrica hacia la derecha, lo que valida la decisión metodológica de aplicar una transformación logarítmica posteriormente. El periodo de transacciones cubre desde enero hasta finales de junio de 2026, y no se detectaron valores nulos.

### 4.1 Análisis Exploratorio de Datos (EDA)

Distribución de la variable objetivo, se confirma un escenario de clases moderadamente desbalanceadas como se demuestra en la tabla de distribución de categorios en el notebook Electronics concentra el 32.8% de las transacciones, Fashion el 22.6%, Home Appliances el 21.9%, Accessories el 16.1% y Books únicamente el 6.6%. Este desbalance, resulta relevante para la interpretación de las métricas por clase, particularmente para la clase minoritaria Books.



<img width="773" height="530" alt="image" src="https://github.com/user-attachments/assets/950bedda-64ea-43f7-99d0-d10057fe56cb" />

Se verificó que cada uno de los 14 productos del catálogo pertenece a una única categoría. Esta relación 1 a 1 confirma que Product constituye una fuga de información (data leakage) directa hacia la variable objetivo, por lo que fue excluida del entrenamiento. Esta decisión reemplaza la alternativa de codificación por frecuencia (Product_Freq).

<img width="1487" height="490" alt="image" src="https://github.com/user-attachments/assets/feedd562-ac62-42cd-93b8-2c989e211152" />

Los diagramas de caja muestran que Electronics es la categoría con mayor dispersión de precios y la presencia de valores atípicos hacia precios altos, mientras que Books concentra los precios más bajos y homogéneos, mientras tanto la variable Quantity, no muestra diferencias entre categorías, lo que anticipa un bajo poder predictivo para esta variable.

La distribución de transacciones por ciudad es homogénea ya que se encuentra entre 87 y 125 transacciones por ciudad, sobre 10 ciudades, sin una concentración geográfica marcada. 

<img width="1489" height="490" alt="image" src="https://github.com/user-attachments/assets/4be0f689-4418-43b0-b74e-16251befa0b5" />

El análisis mensual por categoría muestra que Electronics se mantiene consistentemente como la categoría líder en volumen a lo largo de los seis meses, mientras que Books permanece estable como la categoría de menor volumen.

### 4.2 Feature Engineering

A partir de Date se generaron las variables Month, DayOfWeek, Quarter e Is_Weekend. Sobre Price se aplicó la transformación Log_Price = log(1 + Price) para reducir la asimetría, tomando en consideración que no se esta contemplando la variaple Product, ya que esto produce una desviación menor.

Las variables finalmente utilizadas como predictoras son Quantity, Price, Log_Price, City, Month, DayOfWeek, Quarter e Is_Weekend, ademad de dos columnas diferenciadoras, una con StandardScaler sobre las variables numéricas para la Regresión Logística, y otro con passthrough para los modelos basados en árboles Decision Tree y Random Forest, que no requieren escalado.

La división train/test se realizó en proporción 80/20 con estratificación sobre Category y semilla fija con "random_state=42", obteniendo 800 registros de entrenamiento y 200 de prueba, ademas se verificó que la distribución de clases se conserva casi exactamente entre ambos conjuntos con una diferencias menores a 0.5 puntos porcentuales por clase.

### 4.3 Entrenamiento y Control de Overfitting

Directamente de la codificación se realizaron los 3 modelos definidos, a través de hiperparametros guiados explícitamente al los controles de sobreajuste.

| Modelo              | Accuracy Entrenamiento | Accuracy Prueba |
| ------------------- | ---------------------- | --------------- |
| Regresión Logística | 67,38%                 | 67,50%          |
| Decision Tree       | 90,62%                 | 87,50%          |
| Random Forest       | 92,75%                 | 87,50%          |

La tabla refleja que la Regresión Logística obtuvo el menor desempeño, alcanzando una exactitud cercana al 67% tanto en entrenamiento como en prueba, lo que nos indica ue el modelo posee una capacidad limitada para capturar las relaciones existentes entre las variables y la categoría del producto.

Por otro lado, el modelo basado en arboles presentaron un desempeño considerablemente superior, el Árbol de Decisión alcanzó una exactitud del 90,62% en entrenamiento y 87,50% en prueba, mientras que Random Forest obtuvo un 92,75% en entrenamiento y 87,50% en prueba.

### 4.4 Control de Overfitting

Uno de los aspectos más relevantes durante el entrenamiento fue evaluar la capacidad de generalización de los modelos y detectar posibles problemas de sobreajuste "overfitting".

| Modelo              | Overfitting Gap |
| ------------------- | --------------- |
| Regresión Logística | -0,0013         |
| Decision Tree       | 0,0312          |
| Random Forest       | 0,0525          |

La Regresión Logística presenta un Overfitting Gap prácticamente nulo, lo que representa que el comportamiento del modelo es muy similar en entrenamiento y prueba. Sin embargo, este resultado no necesariamente implica un mejor modelo, sino que demuestra una menor complejidad y una capacidad limitada para capturar patrones más sofisticados presentes en los datos.

El Árbol de Decisión presenta una diferencia de aproximadamente 3,1 puntos porcentuales entre entrenamiento y prueba. Esta diferencia es relativamente pequeña y sugiere que el modelo mantiene una adecuada capacidad, sin evidencias importantes de sobreajuste.

Y por ultimo, el Random Forest presenta la mayor diferencia entre entrenamiento y prueba, alcanzando aproximadamente 5,3 puntos porcentuales, esta diferencia sigue siendo moderada y se encuentra dentro de rangos aceptables para modelos de ensamblaje y el rendimiento en prueba se mantiene elevado, lo que permite concluir que el modelo conserva una buena capacidad

### 5.Resultados obtenidos.

### 5.1 Comparación de Métricas

| Modelo | Train Accuracy | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | ROC-AUC OVR (Weighted) | Overfitting Gap |
|----------|----------|----------|----------|----------|----------|----------|----------|
| Regresión Logística | 0.6738 | 0.6750 | 0.6791 | 0.6750 | 0.6423 | 0.8809 | -0.0013 |
| Decision Tree | 0.9062 | 0.8750 | 0.8873 | 0.8750 | 0.8762 | 0.9836 | 0.0312 |
| Random Forest | 0.9275 | 0.8750 | 0.8889 | 0.8750 | 0.8769 | 0.9857 | 0.0525 |

Los resultados muestran que los modelos basados en árboles superan ampliamente a la Regresión Logística en todas las métricas evaluadas. 

La Regresión Logística obtuvo una exactitud cercana al 67,5%, lo que indica que las relaciones presentes en los datos no son completamente lineales.

Por otro lado, Decision Tree y Random Forest alcanzaron una exactitud del 87,5% sobre el conjunto de prueba, lo que demuestra una mejor capacidad para capturar patrones complejos entre las variables predictoras y la categoría del producto.

### 5.2 Matrices de Confusión

La Regresión Logística concentra casi todos sus errores en la clase Accessories, que es  confundida con Fashion y Home Appliances, mientras que Books y Home Appliances son bien identificadas. Decision Tree y Random Forest muestran patrones de error casi idénticos entre sí, ambos clasifican perfectamente Books, identifican con alta certeza Home Appliances, y concentran su confusión principal entre Electronics y Fashion, junto con cierta confusión menor entre Accessories y Fashion.

<img width="1530" height="532" alt="image" src="https://github.com/user-attachments/assets/cba76110-6955-48fc-a552-17e67b9487ef" />

### 5.3 Reporte por clase

El hallazgo más notorio es el desempeño de la Regresión Logística sobre Accessories con una precision 0.09, recall 0.03 y F1 0.05, lo que demuestra que el modelo prácticamente no logra identificar esta clase solo 1 de 32 instancias correctamente clasificadas, compensando artificialmente su accuracy global al clasificar correctamente clases más separables como Books y Home Appliances con un recall 0.98.

Por otro lado, Decision Tree y Random Forest logran un desempeño equilibrado en Accessories con precision=0.79, recall=0.84, F1=0.82 y muestran un patrón en Electronics considerando una precision de 1.00 pero recall de solo 0.80, es decir, cuando el modelo predice Electronics nunca se equivoca, pero deja sin identificar cerca de un 20% de las transacciones reales de esa categoría.

<img width="1489" height="490" alt="image" src="https://github.com/user-attachments/assets/85c73727-96b4-4ca9-997c-af1acb88aab8" />

El bajo desempeño de la Regresión Logística, combinado con su brecha de overfitting prácticamente nula (-0.0013), no indica buena generalización sino subajuste (underfitting), el modelo es demasiado simple para representar las fronteras de decisión reales entre categorías.

Decision Tree y Random Forest obtienen exactamente el mismo accuracy de test (0.875), pero Random Forest presenta una brecha de overfitting mayor a 0.0525 frente a 0.0312 . 

La clase minoritaria Books (6.6% de los datos) es clasificada perfectamente por los tres modelos, lo que indica que su separabilidad depende de un rango de precios claramente distintivo.

### 6.Conclusiones

El desarrollo del proyecto afirma que la categoria de un producto en el dicho dataset descrito anteriormente puede predecirse con un desempeño satisfactorio con un 87.5% de accuracy en test, F1 weighted 0.877, sin la necesidad de la variable product

Por otro lado, Random Forest y Decision Tree resultan ser los modelos recomendados para este problema, con un desempeño casi idéntico entre ambos,  dada su mayor interpretabilidad y menor brecha de overfitting, el Decision Tree con profundidad acotada constituye una alternativa igualmente competitiva y más simple que el ensemble. La Regresión Logística, si bien aporta un ROC-AUC de 0.88, no es adecuada como modelo final debido a su deficiencia de generalizar en la clase Accessories.
