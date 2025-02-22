import numpy as np
from multiprocessing import Pool

def multiplicar_bloque_matriz(args):
    bloque_A, B, fila_inicio, tam_bloque = args

    resultado_bloque = np.dot(bloque_A, B)
    return (fila_inicio, resultado_bloque)

def multiplicacion_matriz_paralela(A, B):
    
    num_bloques = 4  
    tam_bloque = A.shape[0] // num_bloques

    pool = Pool(processes=num_bloques)
    
    tareas = [(A[i*tam_bloque:(i+1)*tam_bloque, :], B, i*tam_bloque, tam_bloque) for i in range(num_bloques)]
    
    resultados = pool.map(multiplicar_bloque_matriz, tareas)
    
    pool.close()
    pool.join()

    C = np.zeros((A.shape[0], B.shape[1]))
    for fila_inicio, resultado_bloque in resultados:
        C[fila_inicio:fila_inicio+tam_bloque, :] = resultado_bloque
    return C

if __name__ == '__main__':
    A = np.random.randint(-10, 11, (2000, 1000))  
    B = np.random.randint(-10, 11, (1000, 2000))  
    C = multiplicacion_matriz_paralela(A, B) 
    print(C)
