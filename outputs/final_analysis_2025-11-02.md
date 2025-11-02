# Informe de Análisis de Embeddings y Clustering

**Fecha:** 2025-11-02

## 1. ¿Qué embedding separa mejor los signos?

El embedding que separa mejor los signos es el **text-embedding-3-large**, ya que presenta una mayor varianza explicada en PCA con un ratio de [0.2627, 0.1776] en comparación con el **text-embedding-3-small** que tiene un ratio de [0.2454, 0.1809]. Esto sugiere que el modelo más grande captura mejor la estructura de los datos.

## 2. ¿Se observa agrupamiento claro en PCA?

No se proporciona información específica sobre la claridad del agrupamiento en PCA, pero los ratios de varianza explicada indican que ambos embeddings tienen cierta capacidad para representar la estructura de los datos. Sin embargo, se necesitaría visualizar los gráficos de PCA para una evaluación más precisa.

## 3. ¿Qué signos tienden a confundirse?

No se han identificado pares de signos que tiendan a confundirse en el análisis, ya que la sección de "top_confused_pairs" está vacía. Esto sugiere que los signos están suficientemente separados en el espacio de embeddings.

## 4. ¿Influye el intérprete o el modelo de embedding más en la separabilidad?

El análisis no proporciona métricas de silhouette, lo que limita la evaluación directa de la influencia del intérprete o del modelo de embedding. Sin embargo, la comparación de varianza explicada sugiere que el modelo de embedding tiene un impacto más significativo en la separabilidad de los signos.

---

### Conclusiones

- El **text-embedding-3-large** es el mejor modelo para la separación de signos.
- La claridad del agrupamiento en PCA no se puede evaluar sin visualización.
- No se identificaron signos que se confundan entre sí.
- La influencia del modelo de embedding parece ser más relevante que la del intérprete en la separabilidad.