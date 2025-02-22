import numpy as np
from multiprocessing import Pool

def multiplicar_bloque_matriz(args):
    bloque_A, B, fila_inicio, tam_bloque = args
    resultado_bloque = np.dot(bloque_A, B)
    return (fila_inicio, resultado_bloque)

def multiplicacion_matriz_paralela(A, B):
    
    #Partición: Dividimos la matriz A en bloques para paralelizar el trabajo
    num_bloques = 4 
    tam_bloque = A.shape[0] // num_bloques  
    
    #Mapeo: Creamos un Pool de procesos para distribuir las tareas entre los núcleos
    pool = Pool(processes=num_bloques)
    
    #Preparamos las tareas para cada bloque de la matriz A
    tareas = [(A[i*tam_bloque:(i+1)*tam_bloque, :], B, i*tam_bloque, tam_bloque) for i in range(num_bloques)]
    
    #Comunicación: Ejecutamos las tareas en paralelo y recogemos los resultados
    resultados = pool.map(multiplicar_bloque_matriz, tareas)

    pool.close()
    pool.join()

    #Aglomeración: Colocamos los resultados parciales en la matriz resultante C
    C = np.zeros((A.shape[0], B.shape[1])) 
    for fila_inicio, resultado_bloque in resultados:
        C[fila_inicio:fila_inicio+tam_bloque, :] = resultado_bloque 
    return C

if __name__ == '__main__':

    A = np.random.randint(-10, 11, (2000, 1000))  
    B = np.random.randint(-10, 11, (1000, 2000)) 

    C = multiplicacion_matriz_paralela(A, B) 

    print(C)
