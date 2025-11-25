# tp_yelp

Sistema de recomendacion binaria para Yelp utilizando un grafo heterogeneo y un modelo HGT (Heterogeneous Graph Transformer). El proyecto incluye notebooks para explorar datos, preparar el grafo, entrenar y evaluar el modelo, y checkpoints preentrenados para evitar tiempos largos de computo.

## Estructura del repo
- grafos_script.ipynb: notebook principal. Carga datos, arma el grafo heterogeneo, entrena el modelo HGT y evalua. 
- EDA_yelp.ipynb: analisis exploratorio.
- notebook_prueba.ipynb: pruebas varias.
- modelo_yelp_philly.pth: checkpoint principal preentrenado (Filadelfia).
- modelo_yelp.pth, modelo_yelp2.pth: otros checkpoints.
- yelp_dataset/: carpeta esperada para los datos. Colocar aqui los archivos:
  - business_philadelphia.json
  - review_philadelphia.json
  - user_philadelphia.json
  - reviews_con_embeddings.pkl (embeddings de resenas ya calculados; evita recalcular con SentenceTransformer).
- Yelp_DS_Documentation.pdf: documentacion del dataset.

## Requisitos
- Python 3.10+ recomendado.
- Paquetes clave: torch, torch_geometric, sentence_transformers, pandas, numpy, scikit-learn, matplotlib.
- GPU opcional pero recomendable para entrenamiento completo; evaluacion corre en CPU.

## Dataset
1. Descarga desde el link de Drive: https://drive.google.com/drive/folders/184GClZ2W_wJVtyN8Vrzdgp7yz3-UY1CH?usp=drive_link
2. Copia los cuatro archivos listados arriba dentro de yelp_dataset/ en la raiz del repo.
3. Si no usas reviews_con_embeddings.pkl, el notebook puede recalcular embeddings con SentenceTransformer("all-MiniLM-L6-v2"), pero el tiempo de corrida es considerablemente superior.

## Como correr usando el modelo preentrenado (rapido)
1. Abre grafos_script.ipynb.
2. Ejecuta las celdas de carga y preparacion de datos (hasta antes del loop de entrenamiento). Esto arma el grafo y las etiquetas.
3. Salta el loop de entrenamiento (80 epocas).
4. Utiliza el modelo preentrenado modelo_yelp_philly.pth (Con eso podes obtener resultados finales sin reentrenar)
   - Carga el checkpoint modelo_yelp_philly.pth. Se podrian calcular metricas como Accuracy, Recall positivo/negativo, AUC, Recall@K en train/val/test.


## Como entrenar desde cero (completo)
1. Asegura que yelp_dataset/ contiene los archivos JSON y el reviews_con_embeddings.pkl (o recalcula embeddings si lo eliminas).
2. Abre grafos_script.ipynb y ejecuta secuencialmente todas las celdas hasta el loop de entrenamiento.
3. Ejecuta el loop de entrenamiento (80 epocas) para obtener performance_acc y performance_recall_neg y, si lo deseas, volver a guardar un nuevo checkpoint (modelo_yelp_philly.pth).


## Notas
- El loop de entrenamiento es lo mas costoso; si quieres curvas por epoca, debes ejecutarlo (puedes reducir epocas para pruebas).
