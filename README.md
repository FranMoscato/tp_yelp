# tp_yelp

Para descargar el dataset, usar el siguiente link: https://drive.google.com/drive/folders/184GClZ2W_wJVtyN8Vrzdgp7yz3-UY1CH?usp=drive_link

## Explicación del código

### `EDA_yelp.ipynb`
- Carga los dumps en formato JSON línea por línea para no sobrecargar memoria y arma `DataFrame` de negocios, reviews y usuarios.
- Identifica las ciudades con más registros, filtra las observaciones de Philadelphia y guarda copias reducidas (`business_philadelphia.json`, `review_philadelphia.json`, `user_philadelphia.json`) que sirven como subconjunto de trabajo para realizar la POC.
- Cruza reviews con negocios de la ciudad para asegurar consistencia y calcula métricas básicas (shapes, conteos por ciudad, histograma de estrellas) para entender la distribución de calificaciones.
- Genera un histograma en porcentaje de la columna `stars` para visualizar el desbalance de clases que enfrentan los modelos.

### `undersampling.ipynb`
- Parte del dataset completo de negocios para explorar su distribución geográfica y de calificaciones, separando segmentos útiles (negocios con una sola review vs. múltiples reviews).
- Grafica ciudades y estados con más negocios, desagregando por cantidad de estrellas y mostrando proporciones para detectar regiones dominantes.
- Limpia y vectoriza las categorías de negocio mediante `MultiLabelBinarizer` para cuantificar presencia, reviews acumuladas y reputación media de cada categoría o subcategoría (Restaurants, Food, etc.).
- Construye un mapa interactivo con Folium para inspeccionar la dispersión geográfica y detectar áreas densas antes de hacer muestreos o recortes del dataset.

### `grafos_script.ipynb`
- Crea un subconjunto balanceado de reviews de Philadelphia, elimina campos irrelevantes y genera embeddings semánticos por review con `SentenceTransformer (all-MiniLM-L6-v2)`.
- Usa los embeddings como atributos de aristas en un grafo heterogéneo (`user` ↔ `place`) con `torch_geometric`, donde los nodos representan usuarios y negocios, y las aristas representan reviews etiquetadas (buena reseña si `stars >= 4`).
- Mapea ids string a índices enteros, arma `edge_index`, `edge_attr` y `edge_label`, y aplica `RandomLinkSplit` para obtener folds de train/val/test que alimentarán modelos de recomendación o link prediction.

### `notebook_prueba.ipynb`
- Carga las reviews filtradas, divide en train/test estratificado y construye un vocabulario propio (min `freq=5`) para tokenizar texto simple por palabras.
- Implementa un `Dataset` + `DataLoader` con padding dinámico y truncado a `MAX_LEN=200` tokens para alimentar una arquitectura `Embedding → LSTM → Linear` (no bidireccional) que produce logits sobre 5 clases (0-4 estrellas).
- Entrena y evalúa con `CrossEntropyLoss`, registra accuracies por época, grafica la curva de aprendizaje y ejecuta una búsqueda manual de hiperparámetros (dimensión de embedding y hidden size) para comparar resultados.
- La elección de una LSTM unidireccional y compacta responde a que las reseñas tienen longitud variable y el dataset reducido exige un modelo eficiente que capture dependencias temporales básicas sin sobreajustar; el truncado/padding a 200 tokens mantiene el costo computacional bajo y permite mini-batches uniformes.

Estos notebooks se complementan: primero se explora y recorta la data (`EDA_yelp`, `undersampling`), luego se generan representaciones para modelos de texto o grafos (`grafos_script`, `notebook_prueba`) sobre el subconjunto reducido que facilita experimentar sin necesidad de procesar el dataset completo de Yelp.
