import numpy as np


def substitui_variaveis_funcao(funcao, variaveis, ponto):
    return funcao.subs(dict(zip(variaveis, ponto)))


def substitui_variaveis_gradiente(gradiente, variaveis, ponto):
    return np.array(
        [gradiente_i.subs(dict(zip(variaveis, ponto))) for gradiente_i in gradiente]
    ).astype(np.float64)


def substitui_variaveis_hessiana(hessiana, variaveis, ponto):
    return np.array(hessiana.subs(dict(zip(variaveis, ponto)))).astype(np.float64)
