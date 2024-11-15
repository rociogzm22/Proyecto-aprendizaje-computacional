library(randomForest) # Random Forest
library(caret)
library(e1071) # SVM
library(adabag) # Bagging
library(rpart.plot)
library(kernlab)
library(rattle)
library(RColorBrewer)

# En este ejercicio de clasificación de dígitos, se han implementado varios algoritmos de aprendizaje automático. 
# Se incluyen enfoques basados en árboles de decisión, bosques aleatorios, SVM, Bagging y Boosting. 
# El rendimiento de cada modelo se evalúa utilizando matrices de confusión y se calcula su precisión.

# Determinar ubicación del directorio: getwd()
# Cargar datos previamente guardados:
# load("resultados_guardados.RData")

# Importar los conjuntos de datos
entrenamiento <- read.csv("C:/Users/Usuario_UMA/Desktop/Machine/train.csv")
prueba <- read.csv("C:/Users/Usuario_UMA/Desktop/Machine/test.csv")

# Visualizar la proporción de etiquetas en el conjunto inicial
prop.table(table(entrenamiento$label)) * 100

# Convertir las etiquetas a factores
entrenamiento$label <- factor(entrenamiento$label)

## División de datos y validación cruzada

# Separar datos en entrenamiento (80%) y prueba (20%)
n_filas <- nrow(entrenamiento) # Número total de filas
tam_prueba <- ceiling(0.2 * n_filas) # 20% del total
indices_aleatorios <- sample(1:n_filas, n_filas, replace = FALSE) 
indices_prueba <- indices_aleatorios[1:tam_prueba]

entrenamiento_final <- entrenamiento[-indices_prueba,] # Conjunto de entrenamiento
prueba_final <- entrenamiento[indices_prueba,] # Conjunto de prueba

set.seed(4321)

###################### ÁRBOL DECISIÓN ########################
modelo_arbol <- rpart(label~., data = entrenamiento_final, method = "class")

matriz_conf_arbol <- table(predict(modelo_arbol, newdata = prueba_final, type = "class"), prueba_final$label)
precision_arbol <- sum(diag(matriz_conf_arbol)) / sum(matriz_conf_arbol)
precision_arbol # Ejemplo: 0.6416

fancyRpartPlot(modelo_arbol)

barplot(modelo_arbol$variable.importance) # Ejemplo: pixel 433

####################### RANDOM FOREST ########################
bosque_10 <- randomForest(label ~ ., data = entrenamiento_final, ntree = 10, nodesize = 50)

matriz_conf_rf10 <- table(predict(bosque_10, newdata = prueba_final), prueba_final$label)
precision_rf10 <- sum(diag(matriz_conf_rf10)) / sum(matriz_conf_rf10)
precision_rf10 # Ejemplo: 0.9165

######################### SVM ################################
svm_kernel <- ksvm(label ~ ., data = entrenamiento_final, kernel = "polydot", kpar = list(degree = 3), cross = 3)

matriz_conf_svm <- table(predict(svm_kernel, prueba_final), prueba_final$label)
precision_svm <- sum(diag(matriz_conf_svm)) / sum(matriz_conf_svm)
precision_svm # Ejemplo: 0.97333

####################### BAGGING #############################
modelo_bag <- bagging(label~., 
                      data=entrenamiento_final, 
                      na.action = na.omit,
                      mfinal=9,
                      control=rpart.control(cp = 0.001, minsplit=7))

matriz_conf_bag <- table(prueba_final[, "label"], predict(modelo_bag, newdata = prueba_final, type = "class")$class)
precision_bag <- sum(diag(matriz_conf_bag)) / sum(matriz_conf_bag)
precision_bag # Ejemplo: 0.85714

######################## BOOSTING ###########################
modelo_boost <- boosting(label ~ ., data = entrenamiento_final, boos=TRUE, mfinal = 10, coeflearn = 'Breiman')

summary(modelo_boost) 
errorevol(modelo_boost, newdata = entrenamiento) 

matriz_conf_boost <- table(predict(modelo_boost, newdata = prueba_final, type="class")$class, Actual = prueba_final$label)
precision_boost <- sum(diag(matriz_conf_boost)) / sum(matriz_conf_boost)
precision_boost # Ejemplo: 0.7894

arbol_individual <- modelo_boost$trees[[1]]

fancyRpartPlot(arbol_individual)

# Visualización del dígito
rotar_imagen <- function(imagen) {
  t(apply(imagen, 2, rev))
}

indice_prueba <- sample(1:nrow(prueba_final), 1)
fila_prueba <- prueba_final[indice_prueba, ]
etiqueta <- fila_prueba$label
fila_sin_label <- fila_prueba[, -1]

prediccion <- predict(svm_kernel, fila_sin_label)

matriz <- matrix(as.vector(fila_sin_label), nrow = 28, ncol = 28, byrow = TRUE) 
imagen <- apply(matriz, 1:2, as.numeric)

image(rotar_imagen(imagen), main = paste("Etiqueta:", etiqueta, "Predicción:", prediccion))

cat("Etiqueta Real:", as.character(etiqueta), "\n")
cat("Predicción:", as.character(prediccion), "\n")

# Guardar resultados
# save(entrenamiento, prueba, entrenamiento_final, prueba_final, modelo_arbol, matriz_conf_arbol, precision_arbol, 
#     bosque_10, matriz_conf_rf10, precision_rf10, svm_kernel, matriz_conf_svm, precision_svm, modelo_bag, matriz_conf_bag, precision_bag, 
#     modelo_boost, matriz_conf_boost, precision_boost, arbol_individual, file = "nuevos_resultados.RData")
