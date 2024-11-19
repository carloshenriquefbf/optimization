import tqdm
import numpy as np
from sympy import diff, hessian
from utils import (
    substitui_variaveis_funcao,
    substitui_variaveis_gradiente,
    substitui_variaveis_hessiana,
)


# Vetor gradiente
def calcula_gradiente(funcao, variaveis):
    return [diff(funcao, variavel) for variavel in variaveis]


# Matrzi Hessiana
def calcula_hessiana(funcao, variaveis):
    return hessian(funcao, variaveis)


# Busca de Armijo
def busca_de_armijo(
    funcao: any,
    gradiente: any,
    variaveis: list,
    ponto_inicial: np.array,
    d: np.array,
    gamma: float,
    eta: float,
):
    """
    Parametros:
      - funcao: função a ser minimizada
      - gradiente: gradiente da função a ser minimizada
      - variaveis: lista de variáveis da função
      - ponto_inicial: ponto inicial
      - d: direção de descida
      - gamma: fator de redução do passo
      - eta: fator de ???     # TODO: O que é o eta?

    Retorna:
      - t: passo ótimo
      - iteracoes: número de iterações
    """
    t = 1
    iteracoes = 1
    while substitui_variaveis_funcao(
        funcao, variaveis, ponto_inicial + t * d
    ) > substitui_variaveis_funcao(funcao, variaveis, ponto_inicial) + eta * t * np.dot(
        substitui_variaveis_gradiente(gradiente, variaveis, ponto_inicial), d
    ):
        t = gamma * t
        iteracoes += 1

    return t, iteracoes


# Método do Gradiente
def metodo_do_gradiente(
    funcao: any,
    gradiente: any,
    variaveis: list,
    ponto_inicial: np.array,
    gamma: float,
    eta: float,
    maximo_iteracoes: int = 1000,
    valor_minimo: float = 1e-6,
):
    """
    Parametros:
      - funcao: função a ser minimizada
      - gradiente: gradiente da função a ser minimizada
      - variaveis: lista de variáveis da função
      - ponto_inicial: ponto inicial
      - gamma: fator de redução do passo
      - eta: fator de ??? # TODO: O que é o eta?
      - maximo_iteracoes: número máximo de iterações
      - valor_minimo: valor mínimo da função para parar a execução

    Retorna:
      - ponto: ponto ótimo
      - iteracoes: número de iterações
      - iteracoes_armijo: número de iterações da busca de armijo
    """
    pbar = tqdm.tqdm(total=maximo_iteracoes, desc="Método do Gradiente")
    iteracoes = 0
    iteracoes_armijo = 0
    ponto = ponto_inicial
    while (
        np.linalg.norm(substitui_variaveis_gradiente(gradiente, variaveis, ponto))
        > valor_minimo
    ):
        pbar.update(1)
        d = -substitui_variaveis_gradiente(gradiente, variaveis, ponto)
        t, iteracoes_armijo_tmp = busca_de_armijo(
            funcao, gradiente, variaveis, ponto, d, gamma, eta
        )
        iteracoes_armijo += iteracoes_armijo_tmp
        ponto = ponto + t * d

        iteracoes += 1
        if iteracoes >= maximo_iteracoes:
            break

    return ponto, iteracoes, iteracoes_armijo


# Método de Newton
def metodo_de_newton(
    funcao: any,
    gradiente: any,
    hessiana: any,
    variaveis: list,
    ponto_inicial: np.array,
    gamma: float,
    eta: float,
    maximo_iteracoes: int = 1000,
    valor_minimo: float = 1e-6,
):
    """
    Parametros:
      - funcao: função a ser minimizada
      - gradiente: gradiente da função a ser minimizada
      - hessiana: hessiana da função a ser minimizada
      - variaveis: lista de variáveis da função
      - ponto_inicial: ponto inicial
      - gamma: fator de redução do passo
      - eta: fator de ???     # TODO: O que é o eta?
      - maximo_iteracoes: número máximo de iterações
      - valor_minimo: valor mínimo da função para parar a execução

    Retorna:
      - ponto: ponto ótimo
      - iteracoes: número de iterações
      - iteracoes_armijo: número de iterações da busca de armijo
    """

    pbar = tqdm.tqdm(total=maximo_iteracoes, desc="Método de Newton")
    iteracoes = 0
    iteracoes_armijo = 0
    ponto = ponto_inicial

    while (
        np.linalg.norm(substitui_variaveis_gradiente(gradiente, variaveis, ponto))
        > valor_minimo
    ):
        pbar.update(1)
        d = -np.linalg.inv(
            substitui_variaveis_hessiana(hessiana, variaveis, ponto)
        ) @ substitui_variaveis_gradiente(gradiente, variaveis, ponto)
        t, iteracoes_armijo_tmp = busca_de_armijo(
            funcao, gradiente, variaveis, ponto, d, gamma, eta
        )
        iteracoes_armijo += iteracoes_armijo_tmp
        ponto = ponto + t * d

        iteracoes += 1
        if iteracoes >= maximo_iteracoes:
            break

    return ponto, iteracoes, iteracoes_armijo


# Método de Quase Newton
def metodo_de_quase_newton(
    funcao: any,
    n_variaveis: int,
    gradiente: any,
    variaveis: list,
    ponto_inicial: np.array,
    gamma: float,
    eta: float,
    maximo_iteracoes: int = 1000,
    valor_minimo: float = 1e-6,
    metodo: str = "dfp",
):
    """
    Parametros:
      - funcao: função a ser minimizada
      - gradiente: gradiente da função a ser minimizada
      - hessiana: hessiana da função a ser minimizada
      - variaveis: lista de variáveis da função
      - ponto_inicial: ponto inicial
      - gamma: fator de redução do passo
      - eta: fator de ???     # TODO: O que é o eta?
      - maximo_iteracoes: número máximo de iterações
      - valor_minimo: valor mínimo da função para parar a execução
      - metodo: método de atualização

    Retorna:
      - ponto: ponto ótimo
      - iteracoes: número de iterações
      - iteracoes_armijo: número de iterações da busca de armijo
    """

    pbar = tqdm.tqdm(total=maximo_iteracoes, desc=f"Método de Quase Newton ({metodo})")
    iteracoes = 0
    iteracoes_armijo = 0
    ponto = ponto_inicial
    H = np.eye(n_variaveis)

    while (
        np.linalg.norm(substitui_variaveis_gradiente(gradiente, variaveis, ponto))
        > valor_minimo
    ):
        pbar.update(1)
        d = -H @ substitui_variaveis_gradiente(gradiente, variaveis, ponto)
        t, iteracoes_armijo_tmp = busca_de_armijo(
            funcao, gradiente, variaveis, ponto, d, gamma, eta
        )
        iteracoes_armijo += iteracoes_armijo_tmp

        proximo_ponto = ponto + t * d
        p = proximo_ponto - ponto
        q = substitui_variaveis_gradiente(
            gradiente, variaveis, proximo_ponto
        ) - substitui_variaveis_gradiente(gradiente, variaveis, ponto)
        ponto = proximo_ponto

        iteracoes += 1
        if iteracoes >= maximo_iteracoes:
            break

        H = calcula_dfp(H, p, q) if metodo == "dfp" else calcula_bfgs(H, p, q)

    return ponto, iteracoes, iteracoes_armijo


# BFGS para Quase Newton
def calcula_bfgs(H, p, q):
    p = p.reshape(p.size, 1)
    q = q.reshape(q.size, 1)

    termo_2_1 = 1 + np.nan_to_num((((q.T @ H) @ q) / (p.T @ q)))
    termo_2_2 = np.nan_to_num((p @ p.T) / (p.T @ q))
    termo_2 = termo_2_1 * termo_2_2
    termo_3 = ((p @ (q.T @ H)) + ((H @ q) @ p.T)) / (p.T @ q)
    return H + termo_2 - np.nan_to_num(termo_3)


# DFP para Quase Newton
def calcula_dfp(H, p, q):
    p = p.reshape(p.size, 1)
    q = q.reshape(q.size, 1)

    termo_2 = (p.T @ p) / (p.T @ q)
    termo_3 = ((H @ q) @ (q.T @ H)) / ((q.T @ H) @ q)
    return H + np.nan_to_num(termo_2) - np.nan_to_num(termo_3)
